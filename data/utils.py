from torch.utils.data.dataloader import default_collate

import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from functools import partial


def multimodal_collate(batch):
    batch = list(filter(lambda x : x is not None, batch))
    images = [item['image'] for item in batch]
    point_clouds = [item['point_cloud'] for item in batch]
    bbox_3d = [item['bbox_3d'] for item in batch]
    images = default_collate(images)
    point_clouds = default_collate(point_clouds)
    # masks = default_collate(masks)
    
    return {
        'image': images,
        'point_cloud': point_clouds,
        'bbox_3d': bbox_3d
    }

def collate_fn(list_data):
    batched_pts_list, batched_gt_bboxes_list = [], []
    batched_labels_list, batched_names_list = [], []

    for data_dict in list_data:
        pts, gt_bboxes_3d = data_dict['pts'], data_dict['gt_bboxes_3d']
        gt_labels, gt_names = data_dict['gt_labels'], data_dict['gt_names']
        num_objects = gt_bboxes_3d.shape[0]
        batched_pts_list.append(torch.from_numpy(pts).contiguous())
        batched_gt_bboxes_list.append(torch.from_numpy(gt_bboxes_3d))
        batched_labels_list.append(
            torch.zeros(num_objects, )
        )
        batched_names_list.append(gt_names) 

    rt_data_dict = dict(
        batched_pts=batched_pts_list,
        batched_gt_bboxes=batched_gt_bboxes_list,
        batched_labels=batched_labels_list,
        batched_names=batched_names_list,
    )

    return rt_data_dict