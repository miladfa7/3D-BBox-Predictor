from torchvision import transforms
from torch.utils.data import Dataset
from glob import glob 
import os
import cv2
import numpy as np
from typing import List, Dict, Any, Tuple
import os
import numpy as np
import torch

from data.preprocess import (
    ImageMaskTransforms,
    PointCloudTransforms,
    BBox3DTransforms
)

class SereactDataLoader(Dataset):
    def __init__(self, data_config: Dict[str, Any], indices: List[str]) -> None:
        self.data_config = data_config
        self.transform = data_config.get("transforms", {})
        self.indices = indices
        self.data_splitter = DataSpliter(data_config)

        self.image_files = self.data_splitter.image_files
        self.point_files = self.data_splitter.point_files
        self.bbox_3d_files = self.data_splitter.bbox_3d_files
        self.mask_files = self.data_splitter.mask_files

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        idx = self.indices[index]     
        # Load data 
        image = cv2.imread(self.image_files[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # height, width, 3
        point_cloud = np.load(self.point_files[idx]) # 3, height, width
        bboxes_3d = np.load(self.bbox_3d_files[idx]) # num_objects, 8, 3
        mask = np.load(self.mask_files[idx]).astype(np.int8) # num_objects, height, width

        if self.transform.get("image_transform") and self.transform.get("mask_transform"):
            image_mask_transformed = ImageMaskTransforms(image, mask, target=self.data_config.get("image_target"))()
            image = image_mask_transformed["image"]
            mask = image_mask_transformed["mask"]

        if self.transform.get("pc_transform"):
            point_cloud, pc_centroid, pc_max_dist = PointCloudTransforms(point_cloud)()

        if self.transform.get("bbox_3d_transform"):
            bboxes_3d = BBox3DTransforms(bboxes_3d, pc_centroid, pc_max_dist)()
        else:
            bboxes_3d = torch.tensor(bboxes_3d)

        return {
            'image': image,
            'point_cloud': point_cloud,
            'labels': bboxes_3d,
        }
 
class DataSpliter:
    def __init__(self, data_config: Dict) -> None:
        self.data_config = data_config
        self.load_data(data_config)

    def load_files(self, root_path: str, sub_path: str, format: str) -> List[str]:
        directory = os.path.join(root_path, sub_path)
        files = glob(f"{directory}/*{format}")
        files = sorted(files, key=lambda x: os.path.splitext(os.path.basename(x))[0]) 
        return files

    def load_data(self, data_config) -> None:
        root_path = data_config.get('root_path')
        image_path = data_config.get('image_path')
        point_cloud_path = data_config.get('point_cloud_path')
        mask_path = data_config.get('mask_path')
        bboxes_3d_path = data_config.get('bboxes_3d')

        self.image_files = self.load_files(root_path, image_path, format='.jpg')
        self.point_files = self.load_files(root_path, point_cloud_path, format='.npy')
        self.bbox_3d_files = self.load_files(root_path, bboxes_3d_path, format='.npy')
        self.mask_files = self.load_files(root_path, mask_path, format='.npy')

    def shuffle_samples(self, sample_count: int) -> np.ndarray:
        indices = np.arange(sample_count)
        np.random.shuffle(indices)
        return indices

    def train_test_indices(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Splits data to train and test indices
        """
        sample_count = len(self.image_files)
        indices = self.shuffle_samples(sample_count)
        train_split = int(np.floor(self.data_config['train_split'] * sample_count))
        train_indices, test_indices = indices[:train_split], indices[train_split:]
        return train_indices, test_indices
       
