import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class Loss(nn.Module):
    def __init__(self, reg_w=1.0, dir_w=0.2, beta=1/9):
        super().__init__()
        self.reg_w = reg_w
        self.dir_w = dir_w
        self.smooth_l1_loss = nn.SmoothL1Loss(reduction='mean', beta=beta)
        self.dir_cls = nn.CrossEntropyLoss()

    def forward(self, bbox_pred, batched_bbox_reg):
        # print("batched_bbox_reg: ",  torch.max(batched_bbox_reg).item())
        reg_loss = self.smooth_l1_loss(bbox_pred, batched_bbox_reg)
        loss = self.reg_w * reg_loss 
        loss_dict = {'loss': loss}
        return loss_dict

    