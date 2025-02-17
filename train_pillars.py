import argparse
import yaml
import os
import torch
from tqdm import tqdm
import wandb
import numpy as np
from torch.utils.data import DataLoader
import torch.optim as optim
from typing import List, Dict, Tuple, Any
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Any, Tuple
from tqdm import tqdm

# Data Loader
from data.pillars_loader import PillarsDataLoader, DataSpliter
from data.utils import collate_fn

# Models 
from models.PointPillars.pointpillars.model import PointPillars
from models.PointPillars.pointpillars.loss import Loss
from models.optimizer import build_optimizer


class Sereact3DBoxPredictor:
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.data_config = config['data']
        self.training_config = config['training']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize WandB
        self.wandb_logger = wandb.init(project="3d-bounding-box-pillars-points-2", config=config)
        wandb.config.update(config)

        # Point Model -> PointPillars
        self.pointpillars = PointPillars(nclasses=1, 
                                         voxel_size=self.data_config["voxels"]["voxel_size"],
                                         point_cloud_range=self.data_config["voxels"]["point_cloud_range"],
                                         max_num_points=self.data_config["voxels"]["max_num_points"],
                                         max_voxels=self.data_config["voxels"]["max_voxels"]).to(self.device)
                                         
        self.loss_func = Loss()
        
        # Optimizer and Scheduler
        self.init_lr = config['training']["learning_rate"]
        optim_config = {
            "lr": self.training_config["learning_rate"],
            "weight_decay": self.training_config["weight_decay"],
            "betas": (0.95, 0.99)
        }
        self.optimizer = build_optimizer(optimizer_name=self.training_config["optimizer"], 
                                         parms=self.pointpillars.parameters(),
                                         config=optim_config)
        
        self.scheduler = optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=self.init_lr * 10,
                                                       total_steps=4320, pct_start=0.4, anneal_strategy='cos',
                                                       cycle_momentum=True, base_momentum=0.95 * 0.895,
                                                       max_momentum=0.95, div_factor=10)
        
        self.epochs = config['training']["num_epochs"]
    
    @property
    def get_models_parameters(self):
        pointpillars_params = sum(p.numel() for p in self.pointpillars.parameters())
        return {
            "pointpillars_params": f"{pointpillars_params:,}",
        }
    
    def load_data(self) -> Tuple[DataLoader, DataLoader]:
        """Load training and validing data."""
        data_config = self.data_config
        data_splitter = DataSpliter(data_config)
        train_indices, valid_indices = data_splitter.train_valid_indices()
        
        train_loader = DataLoader(PillarsDataLoader(data_config, train_indices),
                                  batch_size=data_config["batch_size"],
                                  shuffle=True, num_workers=data_config["num_workers"],
                                  collate_fn=collate_fn)
        
        val_loader = DataLoader(PillarsDataLoader(data_config, valid_indices),
                                 batch_size=data_config["batch_size"],
                                 shuffle=False, num_workers=data_config["num_workers"],
                                 collate_fn=collate_fn)
        
        return train_loader, val_loader
    
    def process_batch(self, data_dict: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        processed_dict = {}
        for key, value in data_dict.items():
            if isinstance(value, list):
                processed_dict[key] = [
                    item.to(self.device) if torch.is_tensor(item) else item
                    for item in value
                ]
            elif torch.is_tensor(value):
                processed_dict[key] = value.to(self.device)
            else:
                processed_dict[key] = value
        return processed_dict
    
    def train_one_epoch(self, train_loader: DataLoader, epoch: int) -> float:
        """Train for one epoch."""
        epoch_loss = 0.0
        nclasses = 1
        
        for batch_id, data_dict in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch+1}")):
            data_dict = self.process_batch(data_dict)
            self.optimizer.zero_grad()
            bbox_cls_pred, bbox_pred, bbox_dir_cls_pred, anchor_target_dict = \
                self.pointpillars(batched_pts=data_dict['batched_pts'], mode='train',
                                  batched_gt_bboxes=data_dict['batched_gt_bboxes'],
                                  batched_gt_labels=data_dict['batched_labels'])

            
            bbox_cls_pred = bbox_cls_pred.permute(0, 2, 3, 1).reshape(-1, nclasses)
            bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 7)
            bbox_dir_cls_pred = bbox_dir_cls_pred.permute(0, 2, 3, 1).reshape(-1, 2)
            batched_bbox_reg = anchor_target_dict['batched_bbox_reg'].reshape(-1, 7)

            loss_dict = self.loss_func(bbox_pred=bbox_pred, batched_bbox_reg=batched_bbox_reg)
            loss = loss_dict['loss']
            
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            
            epoch_loss += loss.item()
        
        return epoch_loss / len(train_loader)
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> None:
        """Train the model."""
        self.pointpillars.train()
        # Watch point pillors training logs as real-time
        self.wandb_logger.watch(self.pointpillars)

        for epoch in range(self.epochs):
            print(f"\n{'=' * 10} Epoch {epoch + 1}/{self.epochs} {'=' * 10}")
            train_loss = self.train_one_epoch(train_loader, epoch)
            eval_loss = self.evaluate(val_loader)
            print(f"Epoch {epoch + 1} \n{'=' * 10} Train Loss: {train_loss:.4f} \n{'=' * 10} Eval Loss: {eval_loss:.4f}")

            # Log to WandB
            self.wandb_logger.log({
                "train_smoothL1_loss": train_loss,
                "eval_smoothL1_loss": eval_loss
            })
        self.save_model(self.epochs)


    def evaluate(self, val_loader: DataLoader) -> float:

        self.pointpillars.eval()
        total_loss = 0     
        nclasses = 1       
        with torch.no_grad():
            for batch_id, data_dict in enumerate(tqdm(val_loader, desc="Evaluating")):
                data_dict = self.process_batch(data_dict)
                
                batched_pts = data_dict['batched_pts']
                batched_gt_bboxes = data_dict['batched_gt_bboxes']
                batched_labels = data_dict['batched_labels']
                bbox_cls_pred, bbox_pred, bbox_dir_cls_pred, anchor_target_dict = \
                    self.pointpillars(batched_pts=batched_pts, 
                                mode='train',
                                batched_gt_bboxes=batched_gt_bboxes, 
                                batched_gt_labels=batched_labels)

                bbox_cls_pred = bbox_cls_pred.permute(0, 2, 3, 1).reshape(-1, nclasses)
                bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 7)
                bbox_dir_cls_pred = bbox_dir_cls_pred.permute(0, 2, 3, 1).reshape(-1, 2)
                batched_bbox_reg = anchor_target_dict['batched_bbox_reg'].reshape(-1, 7)

                loss_dict = self.loss_func(bbox_pred=bbox_pred, batched_bbox_reg=batched_bbox_reg)
                loss = loss_dict["loss"]
                # Accumulate metrics
                total_loss += loss.item()
        
        self.pointpillars.train()       
        return total_loss / len(val_loader)

    def save_model(self, last_epoch: int) -> None:
        """Save the trained model."""
        save_model_path = self.training_config["save_model_path"]
        model_name = self.training_config["model_name"]
        save_path = f"{save_model_path}/{model_name}_{last_epoch}.pth"
        torch.save(self.pointpillars.state_dict(), save_path)
        print(f"Model saved at {save_path}")

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='3D Bounding Box Prediction')
    parser.add_argument('--config', default='configs/pillar_config.yaml', help='Path to config file')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    predictor = Sereact3DBoxPredictor(config)

    models_params = predictor.get_models_parameters
    print("-------------------- Summary of models parameters ---------------------------------")
    print(f"PillarPoints Parameters: {models_params['pointpillars_params']}")
    # print()
    print("---------------------Load dataset--------------------------------")
    
    # Load train and valid dataloader
    train_loader, val_loader = predictor.load_data()
    train_size = len(train_loader.dataset)
    print(f"Total train samples: {train_size}")
    valid_size = len(val_loader.dataset)
    print(f"Total valid samples: {valid_size}")

    # Training
    print("-------------------- Training and Evaluating Model ---------------------------------")
    predictor.train(train_loader, val_loader)

