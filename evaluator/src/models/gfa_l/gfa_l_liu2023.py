import torch
import torch.nn as nn
import torch.nn.functional as F

import src.models.pytorch_utils as pt_utils


class GFA(nn.Module):
    def __init__(self, d_in, d_out, config):        
        super().__init__()
        
        self.fc_a = pt_utils.Conv2d(d_in, 1, kernel_size=(1, 1), activation=None)
        self.fc_b = pt_utils.Conv2d(d_in, 1, kernel_size=(1, 1), activation=None)
        
        self.mlp = pt_utils.Conv2d(d_in, d_in, kernel_size=(1, 1), bn=True, activation=None)

    def forward(self, feature):
        
        a = self.fc_a(feature).squeeze(3)
        b = self.fc_b(feature).squeeze(3)
        att = torch.matmul(a.permute((0, 2, 1)), b)
        att = F.softmax(att, dim=2)
        
        f = self.mlp(feature).squeeze(3).permute((0, 2, 1))
        f = torch.matmul(att, f)
        f = f.permute((0, 2, 1)).unsqueeze(3)
        
        f = feature + f
        
        return f


