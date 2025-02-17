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

from data.preprocess import BBox3DTransforms


class PillarsDataLoader(Dataset):
    def __init__(self, data_config: Dict[str, Any], indices: List[str]) -> None:
        self.data_config = data_config
        self.transform = data_config.get("transforms", {})
        self.indices = indices
        self.total_file = 0
        self.data_splitter = DataSpliter(data_config)
        self.points_files = self.data_splitter.points_files
        self.bbox_3d_files = self.data_splitter.bbox_3d_files
        

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        idx = self.indices[index]     
        # Load data
        point_cloud = np.load(self.points_files[idx]) # 3, height, width
        points = point_cloud.reshape(3, -1).T # N, 3
        # points = np.hstack([points, np.zeros((points.shape[0], 1))])
        bboxes_3d = np.load(self.bbox_3d_files[idx]) # num_objects, 8, 3
        bbox_transform = BBox3DTransforms(bboxes_3d, 0, 0)
        bboxes_3d = bbox_transform.corners_to_params(bboxes_3d)
        gt_labels = np.array([0])

        # Apply data augmentaion
        return {
            'pts': points,
            'gt_bboxes_3d': bboxes_3d,
            "gt_labels": gt_labels,
            "gt_names": ["unknonw"]
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
        point_cloud_path = data_config.get('point_cloud_path')
        bboxes_3d_path = data_config.get('bboxes_3d')

        self.points_files = self.load_files(root_path, point_cloud_path, format='.npy')
        self.bbox_3d_files = self.load_files(root_path, bboxes_3d_path, format='.npy')

        self.total_file = len(self.points_files)

    def shuffle_samples(self, sample_count: int) -> np.ndarray:
        indices = np.arange(sample_count)
        np.random.shuffle(indices)
        return indices

    def train_test_indices(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Splits data to train and test indices
        """
        sample_count = self.total_file
        indices = self.shuffle_samples(sample_count)
        train_split = int(np.floor(self.data_config['train_split'] * sample_count))
        train_indices, test_indices = indices[:train_split], indices[train_split:]
        return train_indices, test_indices
       


# if __name__ == '__main__':
    
#     import yaml
#     import argparse

#     parser = argparse.ArgumentParser(description='3D Bounding Box Prediction')
#     parser.add_argument('--config', default='../configs/config.yaml', help='Path to config file')
#     args = parser.parse_args()

#     with open(args.config, 'r') as f:
#         config = yaml.safe_load(f)

#     data_config = config.get('data', {})
#     data_splitter = DataSpliter(data_config)
#     train_indices, test_indices = data_splitter.train_test_indices()

#     pillars_data = PillarsDataLoader(data_config, train_indices)
#     print(pillars_data.__getitem__(9))

