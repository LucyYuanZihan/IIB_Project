import torch
import torch.nn as nn
import torch.nn.functional as F

import src.models.pytorch_utils as pt_utils


class GFA(nn.Module):
    def __init__(self, d_in, d_out, config):        
        super().__init__()
        
        self.fam = FAM(d_in, d_out, config)

    def forward(self, feature):
        
        f = self.fam(feature)

        return f


class FAM(nn.Module):
    def __init__(self, d_in, d_out, config):        
        super().__init__()
        
        self.fc_a = pt_utils.Conv2d(d_in, d_in, kernel_size=(1, 1), activation=None)
        self.fc_b = pt_utils.Conv2d(d_in, d_in, kernel_size=(1, 1), activation=None)
        
        self.mlp = pt_utils.Conv2d(d_in, d_in, kernel_size=(1, 1), bn=True, activation=None)
        
        self.beta = torch.nn.Parameter(torch.tensor([0.0]), requires_grad=True)
        

    def forward(self, feature):
        
        a = self.fc_a(feature).squeeze(3)
        b = self.fc_b(feature).squeeze(3)
        att = torch.matmul(a, b.permute((0, 2, 1)))
        att = F.softmax(att, dim=2)
        
        f = self.mlp(feature).squeeze(3)
        f = torch.matmul(att, f)
        f = f.unsqueeze(3)
        
        f = feature + self.beta * f
        
        return f