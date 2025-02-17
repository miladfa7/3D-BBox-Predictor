import torch
import torch.nn as nn
from models.fusion_layers import FusionLayer
import numpy as np 


class SereactMultimodelModel(nn.Module):
    def __init__(self, max_num_objects=30):
        super(SereactMultimodelModel, self).__init__()
        self.max_num_objects = max_num_objects 

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        
        # MLP layers for fused features
        self.fc1 = nn.Linear(3072, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 128)
        
        self.object_proposal = nn.Linear(128, 128 * self.max_num_objects)
        
        # Regression head
        self.regression_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            # nn.Linear(64, 8*3)  # Cubue 8 × 3 bounding box shape
            nn.Linear(64, 7)  # 7 Parameterize bounding box
        )
        
        # Confidence head
        self.confidence_head = nn.Sequential(
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        self.fusion_dropout = nn.Dropout(p=0.4)
        self.relu = nn.ReLU()

        
    def forward(self, img_feats, point_feats):
        # fusion layer
        img_feats = img_feats.squeeze(1)
        point_feats = point_feats.squeeze(1)

        fused_feats = torch.cat([img_feats, point_feats], 1)  # [batch, 1, 3072]
        # Feature processing
        x = self.relu(self.fc1(fused_feats))
        x = self.fusion_dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fusion_dropout(x)
        x = self.relu(self.fc3(x))  
        
        object_feats = self.object_proposal(x)  # [batch, 128*num_objects]
        object_feats = object_feats.view(-1, self.max_num_objects, 128)  # [batch, num_objects, 128]
        
        # Predict confidence scores
        confidence = self.confidence_head(object_feats) 
        
        # Predict 3D bounding boxes
        bboxes_3d = self.regression_head(object_feats) 
        # corners = corners.view(-1, self.max_num_objects, 8, 3)  # [batch, num_objects, 8, 3]
        bboxes_3d = bboxes_3d.view(-1, self.max_num_objects, 7)  # [batch, num_objects, 7]
        return bboxes_3d

        
       
