import argparse
import os
import numpy as np
import torch
from models import build_model
from datasets import build_dataset
from utils.dist import init_distributed, is_distributed, is_primary
from utils.misc import my_worker_init_fn
from engine import evaluate
from torch.utils.data import DataLoader
from utils.dist import (
    all_gather_dict,
    all_reduce_average,
    is_primary,
    reduce_dict,
    barrier,
)
from utils.ap_calculator import parse_predictions

import shutil

def load_model(args, dataset_config):
    """Load the trained model checkpoint."""
    model, _ = build_model(args, dataset_config)
    model.cuda()
    checkpoint = torch.load(args.test_ckpt, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model

def run_inference(model, inputs):
    with torch.no_grad():
        outputs = model(inputs)
    return outputs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_ckpt", required=True, type=str, help="Path to model checkpoint")
    parser.add_argument("--dataset_name", required=True, type=str, choices=["scannet", "sunrgbd"], help="Dataset name")
    parser.add_argument("--use_color", default=False, action="store_true")
    parser.add_argument(
        "--model_name",
        default="3detr",
        type=str,
        help="Name of the model",
        choices=["3detr"],
    )
    parser.add_argument(
        "--dataset_root_dir",
        type=str,
        default=None,
        help="Root directory containing the dataset files. \
              If None, default values from scannet.py/sunrgbd.py are used",
    )
        ### Encoder
    parser.add_argument(
        "--enc_type", default="vanilla", choices=["masked", "maskedv2", "vanilla"]
    )
    # Below options are only valid for vanilla encoder
    parser.add_argument("--enc_nlayers", default=3, type=int)
    parser.add_argument("--enc_dim", default=256, type=int)
    parser.add_argument("--enc_ffn_dim", default=128, type=int)
    parser.add_argument("--enc_dropout", default=0.1, type=float)
    parser.add_argument("--enc_nhead", default=4, type=int)
    parser.add_argument("--enc_pos_embed", default=None, type=str)
    parser.add_argument("--enc_activation", default="relu", type=str)

    ### Decoder
    parser.add_argument("--dec_nlayers", default=8, type=int)
    parser.add_argument("--dec_dim", default=256, type=int)
    parser.add_argument("--dec_ffn_dim", default=256, type=int)
    parser.add_argument("--dec_dropout", default=0.1, type=float)
    parser.add_argument("--dec_nhead", default=4, type=int)

    ### MLP heads for predicting bounding boxes
    parser.add_argument("--mlp_dropout", default=0.3, type=float)
    parser.add_argument(
        "--nsemcls",
        default=-1,
        type=int,
        help="Number of semantic object classes. Can be inferred from dataset",
    )

    ### Other model params
    parser.add_argument("--preenc_npoints", default=2048, type=int)
    parser.add_argument(
        "--pos_embed", default="fourier", type=str, choices=["fourier", "sine"]
    )
    parser.add_argument("--nqueries", default=256, type=int)

    ##### Set Loss #####
    ### Matcher
    parser.add_argument("--matcher_giou_cost", default=2, type=float)
    parser.add_argument("--matcher_cls_cost", default=1, type=float)
    parser.add_argument("--matcher_center_cost", default=0, type=float)
    parser.add_argument("--matcher_objectness_cost", default=0, type=float)

    ### Loss Weights
    parser.add_argument("--loss_giou_weight", default=0, type=float)
    parser.add_argument("--loss_sem_cls_weight", default=1, type=float)
    parser.add_argument(
        "--loss_no_object_weight", default=0.2, type=float
    )  # "no object" or "background" class for detection
    parser.add_argument("--loss_angle_cls_weight", default=0.1, type=float)
    parser.add_argument("--loss_angle_reg_weight", default=0.5, type=float)
    parser.add_argument("--loss_center_weight", default=5.0, type=float)
    parser.add_argument("--loss_size_weight", default=1.0, type=float)
    parser.add_argument("--batchsize_per_gpu", default=1, type=int)
    parser.add_argument("--dataset_num_workers", default=4, type=int)
    parser.add_argument("--ngpus", default=1, type=int)
    args = parser.parse_args()

    # Load dataset configuration
    datasets, dataset_config = build_dataset(args)
    # Load model
    model = load_model(args, dataset_config)
    config_dict = {
        'remove_empty_box': True,
        'use_3d_nms': True,
        'nms_iou': 0.25,
        'use_old_type_nms': False,
        'cls_nms': True,
        'per_class_proposal': True,
        'use_cls_confidence_only': False,
        'conf_thresh': 0.25,
        'no_nms': False,
        'dataset_config': dataset_config,
    }


    dataloaders = {}
    split = "test"
    dataset_splits = ["test"]
    sampler = torch.utils.data.SequentialSampler(datasets[split])
    dataloaders[split] = DataLoader(
            datasets[split],
            sampler=sampler,
            batch_size=args.batchsize_per_gpu,
            num_workers=args.dataset_num_workers,
            worker_init_fn=my_worker_init_fn,
        )
    dataloaders[split + "_sampler"] = sampler

    net_device = next(model.parameters()).device
  
    dataset_loader = dataloaders[split]
    num_batches = len(dataset_loader)
    for batch_idx, batch_data_label in enumerate(dataset_loader):
        for key in batch_data_label:
            batch_data_label[key] = batch_data_label[key].to(net_device)

        inputs = {
            "point_clouds": batch_data_label["point_clouds"],
            "point_cloud_dims_min": batch_data_label["point_cloud_dims_min"],
            "point_cloud_dims_max": batch_data_label["point_cloud_dims_max"],
        }
        # Run inference
        outputs = run_inference(model, inputs)
        outputs["outputs"] = all_gather_dict(outputs["outputs"])

        # Details 
        predicted_boxes = outputs['outputs']['box_corners']
        sem_cls_probs = outputs['outputs']['sem_cls_prob']
        objectness_probs = outputs['outputs']['objectness_prob']
        point_cloud = batch_data_label["point_clouds"]
        point_cloud_dims_min = batch_data_label["point_cloud_dims_min"]
        point_cloud_dims_max = batch_data_label["point_cloud_dims_max"]
        gt_bounding_boxes = batch_data_label["gt_box_corners"]

        batch_pred_map_cls = parse_predictions(
                predicted_boxes, sem_cls_probs, objectness_probs, point_cloud, config_dict
        )  
        name_folder = "outputs/test"
       
        element_dir = os.path.join(name_folder, f'element_{batch_idx}')
        os.makedirs(element_dir, exist_ok=True)

        GT = os.path.join(element_dir, 'GT')
        if os.path.exists(GT):
            shutil.rmtree(GT)
        os.makedirs(GT)
        
        all_bboxes = []
        all_pc_data = []
        
        for i in range(point_cloud.size(0)):
            bboxes = [pred[1] for pred in batch_pred_map_cls[i]]  # Extract box parameters
            all_bboxes.append(np.array(bboxes, dtype=np.float32))
            
            pc_data = point_cloud[i].cpu().numpy()
            all_pc_data.append(pc_data)

        np.save(os.path.join(element_dir, 'bboxes.npy'), np.array(all_bboxes))
        np.save(os.path.join(element_dir, 'point_cloud.npy'), np.array(all_pc_data))
        

        print(f"-------------{batch_idx}--------------")
 

if __name__ == "__main__":
    main()