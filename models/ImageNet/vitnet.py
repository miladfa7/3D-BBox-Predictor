import timm
import torch
import torch.nn as nn

class ViTModel(nn.Module):
    def __init__(self, model_name='vit_base_patch16_224', pretrained=True, output_dim=2024):
        super(ViTModel, self).__init__()
        self.output_dim = output_dim
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0 )
        self.fc = nn.Linear(in_features=6144, out_features=output_dim)
        
    def forward(self, x):
        batch = x.size(0)
        x = self.model(x)  
        x = x.view(batch, -1) 
        self.fc = nn.Linear(in_features=x.shape[1], out_features=self.output_dim)
        x = self.fc(x)  
        return x