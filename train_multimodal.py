import argparse
import yaml
import os
import torch
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import wandb

# Custom Moduels
from data.data_loader import SereactDataLoader, DataSpliter
from data.utils import multimodal_collate
from typing import List, Dict, Tuple, Any

# Models
# from models.pointnet.pointnet2_model import PointNet2
from models.PointNet.pointnet2_model import PointNetEncoder
from models.ImageNet.models import build_model
from models.multimodal import SereactMultimodelModel
from models.loss_functions import SmoothL1Loss
from models.optimizer import build_optimizer

class Sereact3DBoxPredictor:
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.training_config = config['training']
        self.valid_config = config['validation']
        self.device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')

        # Initialize WandB
        self.wandb_logger = wandb.init(project="3d-bounding-box-multimodal", config=config)
        wandb.config.update(config)

        self.pc_feature_extractor = PointNetEncoder()
        self.image_feature_extractor = build_model(model_name=self.training_config["backbone"]["image"])
        self.multimodal = SereactMultimodelModel()
        self.loss_func = SmoothL1Loss(beta=1.0 / 9.0, loss_weight=2.0)

        if torch.cuda.is_available():
            self.pc_feature_extractor = self.pc_feature_extractor.to(self.device)
            self.image_feature_extractor = self.image_feature_extractor.to(self.device)
            self.multimodal = self.multimodal.to(self.device)
            self.loss_func = self.loss_func.to(self.device)
        
        optim_config = {
            "lr": self.training_config["learning_rate"],
            "weight_decay": self.training_config["weight_decay"],
            "betas": (0.95, 0.99)
        }
        self.optimizer = build_optimizer(optimizer_name=self.training_config["optimizer"], 
                                         parms=self.multimodal.parameters(),
                                         config=optim_config)

    @property
    def get_models_parameters(self):
        image_params = sum(p.numel() for p in self.image_feature_extractor.parameters())
        pointnet2_params = sum(p.numel() for p in self.pc_feature_extractor.parameters())
        multimodal_params = sum(p.numel() for p in self.multimodal.parameters())
        return {
            "image_model_params": f"{image_params:,}",
            "point_cloud_model_params": f"{pointnet2_params:,}",
            "multimodal_model_params": f"{multimodal_params:,}",
            "total_params": f"{image_params + pointnet2_params + multimodal_params:,}"
        }

    def load_data(self) -> Tuple[DataLoader, DataLoader]:
        """Load training and valid data."""
        data_config = self.config['data']
        data_splitter = DataSpliter(data_config)
        train_indices, val_indices = data_splitter.train_test_indices()

        # DataLoader for training
        train_dataset = SereactDataLoader(data_config, train_indices)
        train_loader = DataLoader(train_dataset, batch_size=data_config["batch_size"], 
                                      shuffle=True, num_workers=data_config["num_workers"])

        # DataLoader for val
        val_dataset = SereactDataLoader(data_config, val_indices)
        val_loader = DataLoader(val_dataset, batch_size=data_config["batch_size"],
                                     shuffle=False, num_workers=data_config["num_workers"])

        return train_loader, val_loader
    
    def train_one_epoch(self, train_loader: DataLoader) -> float:
        epoch_loss = 0 
        for batch_id, batch in enumerate(train_loader):
            self.optimizer.zero_grad()
            images = batch['image']
            point_clouds = batch['point_cloud'].transpose(2, 1) 
            labels = batch['labels']

            if torch.cuda.is_available():
                images = images.to(self.device)
                point_clouds = point_clouds.to(self.device)
                gt_bboxes = labels["gt_bboxes"].to(self.device)
                gt_confidence = labels["gt_confidence"].to(self.device)

            point_feats = self.pc_feature_extractor(point_clouds)
            img_feats = self.image_feature_extractor(images)
            pred_corners = self.multimodal(img_feats, point_feats)
            pred_corners = pred_corners.to(self.device)
            loss = self.loss_func(pred_corners, gt_bboxes)
           
            # Backpropagation
            loss.backward()
            self.optimizer.step()
            # self.scheduler.step()

            epoch_loss += loss.item()

        return epoch_loss
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> None:
        """Train the model."""
        self.pc_feature_extractor.train()
        self.image_feature_extractor.train()
        self.multimodal.train()
        # Watch point multimodal training logs as real-time
        self.wandb_logger.watch(self.multimodal)

        epochs = self.training_config["num_epochs"]
        for epoch in range(epochs):
            train_loss = self.train_one_epoch(train_loader)
            eval_loss = self.evaluate(val_loader)
            print(f"Epoch {epoch + 1} \n{'=' * 10} Train Loss: {train_loss:.4f} \n{'=' * 10} Eval Loss: {eval_loss:.4f}")

            # Log to WandB
            self.wandb_logger.log({
                "train_smoothL1_loss": train_loss,
                "eval_smoothL1_loss": eval_loss
            })
        self.save_model(epochs)

    def save_model(self, last_epoch: int):
        """Save model weights to specified path"""
        save_model_path = self.training_config["save_model_path"]
        model_name = self.training_config["model_name"]
        save_path = f"{save_model_path}/{model_name}_{last_epoch}.pth"
        torch.save({
            'pointnet': self.pc_feature_extractor.state_dict(),
            'imagenet': self.image_feature_extractor.state_dict(),
            'multimodal': self.multimodal.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, save_path)
        print(f"Model saved to {save_path}")

    def evaluate(self, val_loader):
        self.pc_feature_extractor.eval()
        self.image_feature_extractor.eval()
        self.multimodal.eval()
        
        total_loss = 0.0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Evaluating"):
                images = batch['image']
                point_clouds = batch['point_cloud'].transpose(2, 1)
                labels = batch['labels']
                
                if torch.cuda.is_available():
                    images = images.to(self.device)
                    point_clouds = point_clouds.to(self.device)
                    gt_bboxes = labels["gt_bboxes"].to(self.device)
                    gt_confidence = labels["gt_confidence"].to(self.device)

                # Forward pass
                point_feats = self.pc_feature_extractor(point_clouds)
                img_feats = self.image_feature_extractor(images)
                pred_corners = self.multimodal(img_feats, point_feats)
                
                # Calculate loss
                loss = self.loss_func(pred_corners, gt_bboxes)
                total_loss += loss.item()
                

        self.pc_feature_extractor.train()
        self.image_feature_extractor.train()
        self.multimodal.train()

        return total_loss

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='3D Bounding Box Prediction')
    parser.add_argument('--config', default='configs/pillar_config.yaml', help='Path to config file')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    predictor = Sereact3DBoxPredictor(config)

    # Show models parameters
    models_params = predictor.get_models_parameters
    print("-------------------- Summary of models parameters ---------------------------------")
    print(f"Image Model Parameters: {models_params['image_model_params']}")
    print(f"Point Cloud Model Parameters: {models_params['point_cloud_model_params']}")
    print(f"Multimodal Model Parameters: {models_params['multimodal_model_params']}")
    print(f"Total Parameters: {models_params['total_params']}")
    print()
    print("---------------------Load dataset--------------------------------")
    # Load train and val dataloader

    train_loader, val_loader = predictor.load_data()
    # Load train and valid dataloader
    train_loader, val_loader = predictor.load_data()
    train_size = len(train_loader.dataset)
    print(f"Total train samples: {train_size}")
    valid_size = len(val_loader.dataset)
    print(f"Total valid samples: {valid_size}")

    # Training
    print("-------------------- Training and Validation Model ---------------------------------")
    predictor.train(train_loader, val_loader)



