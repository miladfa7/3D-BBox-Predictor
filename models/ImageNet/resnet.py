import timm
import torch
import torch.nn as nn

class ResNetModel(nn.Module):
    def __init__(self, model_name='resnet50', pretrained=True, output_dim=1024):
        super(ResNetModel, self).__init__()

        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        # self.fc = nn.Linear(2048, output_dim)

    def forward(self, x):
        batch = x.size(0)
        x = self.model(x)
        x = x.squeeze()
        x = x.view(batch, 1, 2048)
        return x