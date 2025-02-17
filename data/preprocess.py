import numpy as np
import torch
from typing import Dict, Tuple, Any
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import open3d as o3d
from open3d.geometry import PointCloud
import random

class ImageMaskTransforms:
    def __init__(self, image: np.ndarray, mask: np.ndarray, target: Tuple = (224, 224)) -> None:
        self.image = image
        self.target = target
        self.mask = mask.transpose(1, 2, 0) # Shape (H, W, num_objs)

    def __call__(self) -> torch.Tensor:
        image_transforms = A.Compose([
            A.Resize(self.target, self.target, interpolation=cv2.INTER_NEAREST, mask_interpolation=cv2.INTER_NEAREST),  
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]), # Normalize RGB values to [-1,1]
            ToTensorV2() 
        ])
        self.image = self.image
        image_mask_transformed = image_transforms(image=self.image, mask=self.mask) 
        return image_mask_transformed


class PointCloudTransforms:
    def __init__(self, point_cloud: np.ndarray, voxel_size: float = 0.01, num_points: int = 100000) -> None:
        self.point_cloud = point_cloud
        self.voxel_size = voxel_size
        self.num_points = num_points
    
    def __call__(self) -> torch.Tensor:
        # Reshape (3, H, W) to (N, 3)
        H, W = self.point_cloud.shape[1], self.point_cloud.shape[2]
        points = self.point_cloud.reshape(3, -1).T  # Shape (N, 3)

        if points.shape[0] > self.num_points:
            sampled_indices = np.random.choice(points.shape[0], self.num_points, replace=False)
            processed_points = points[sampled_indices]
        point_cloud_norm, centroid, max_dist = self.normalize_pc(processed_points)
        point_cloud_tensor = torch.tensor(point_cloud_norm, dtype=torch.float32)
        return point_cloud_tensor, centroid, max_dist
    
    def normalize_pc(self, points):
        centroid = np.mean(points, axis=0)
        points -= centroid
        max_dist = np.max(np.sqrt(np.sum(abs(points)**2,axis=-1)))
        points /= max_dist
        return points, centroid, max_dist
    
    def convert_points_to_voxel(self, points: np.ndarray) -> PointCloud:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        processed_pcd = pcd.voxel_down_sample(voxel_size=self.voxel_size)
        processed_points = np.asarray(pcd.points)
        return processed_points

class BBox3DTransforms:
    def __init__(self, bbox_3d: np.ndarray, pc_centroid: np.ndarray, max_dist: float, max_objects:int = 30, parameterized=True) -> None:
        self.bbox_3d = bbox_3d
        self.pc_centroid = pc_centroid
        self.max_dist = max_dist
        self.max_objects = max_objects
        self.num_objects = self.bbox_3d.shape[0]
        self.num_corners = self.bbox_3d.shape[1] 
        self.dim = self.bbox_3d.shape[2] 
        self.parameterized = parameterized

    def __call__(self) -> torch.Tensor:
        bbox_3d_norm = self.normalize_bbox(self.bbox_3d)
        if self.parameterized:
            params_bboxes = self.corners_to_params(bbox_3d_norm)
            padded_boxes = np.zeros((self.max_objects, params_bboxes.shape[1]))
            padded_boxes[:self.num_objects] = params_bboxes
        else:
            padded_boxes = np.zeros((self.max_objects, self.num_corners, self.dim))
            padded_boxes[:self.num_objects] = bbox_3d_norm
        confidence_bboxes = np.zeros((self.max_objects, 1))
        confidence_bboxes[:self.num_objects] = 1.0

        gt_bboxes = torch.tensor(padded_boxes, dtype=torch.float32)
        gt_confidence = torch.tensor(confidence_bboxes, dtype=torch.float32)

        return {
            "gt_bboxes": gt_bboxes,
            "gt_confidence": gt_confidence,
        }
    
    def normalize_bbox(self, bbox_3d: np.ndarray) -> np.ndarray:
        bbox_3d_norm = bbox_3d - self.pc_centroid
        bbox_3d_norm /= self.max_dist
        return bbox_3d_norm
    
    def corners_to_params(self, corners: np.ndarray) -> np.ndarray:
        num_objects = corners.shape[0]
        parameterized_bboxes = []
        for i in range(num_objects):
            center = np.mean(corners[i], axis=0)
            min_xyz = np.min(corners[i], axis=0)
            max_xyz = np.max(corners[i], axis=0)
            size = max_xyz - min_xyz 
            yaw = 0
            bbox_tensor = np.array(
                [center[0], center[1], center[2],
                 size[0], size[1], size[2],
                 yaw], dtype=np.float32
            )
            parameterized_bboxes.append(bbox_tensor)
        parameterized_bboxes = np.stack(parameterized_bboxes)
        return parameterized_bboxes

   

class BBox3DTransforms2:
    """
    Convert 3D corners representation (num_objects, 8, 3) to (center_x, center_y, center_z, width, height, depth, yaw)
    """
    def __init__(self, bbox_3d: np.ndarray) -> None:
        self.bbox_3d = bbox_3d

    def __call__(self) -> torch.Tensor:

        num_objects = self.bbox_3d.shape[0]
        transformed_bboxes = []

        for i in range(num_objects):
            corners = self.bbox_3d[i]
            bbox_center = self.compute_bbox_center(corners)
            bbox_size = self.compute_bbox_size(corners)
            orientation = 0 # TODO: It can compute to standard orientation
            bbox_tensor = torch.tensor(
                [bbox_center[0], bbox_center[1], bbox_center[2],
                 bbox_size[0], bbox_size[1], bbox_size[2],
                 orientation], dtype=torch.float32
            )
            transformed_bboxes.append(bbox_tensor)
        
        transformed_bboxes = torch.stack(transformed_bboxes) # (num_objects, 7)
        return transformed_bboxes
    
    def compute_bbox_center(self, corners: np.ndarray) -> np.ndarray:
        return np.mean(corners, axis=0) 

    def compute_bbox_size(self, corners: np.ndarray):
        min_xyz = np.min(corners, axis=0)
        max_xyz = np.max(corners, axis=0)
        size = max_xyz - min_xyz  
        return size

    
