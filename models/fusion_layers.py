import torch
import torch.nn as nn



class FusionLayer(nn.Module):
    def __init__(self, input_dim1: int = 1024, input_dim2: int = 1024, output_dim: int = 1024):
        super(FusionLayer, self).__init__()
        self.fc = nn.Linear(input_dim1 + input_dim2, output_dim)

    def forward(self, point_cloud_features, image_features):
        fused_features = torch.cat((point_cloud_features, image_features), dim=1)
        return self.fc(fused_features)