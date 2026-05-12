import torch
import torch.nn as nn
import torch.nn.functional as F

import src.models.pytorch_utils as pt_utils


class GFA(nn.Module):
    def __init__(self, d_in, d_out, config):        
        super().__init__()
        
        self.fc = pt_utils.Conv2d(d_in, 1, kernel_size=(1, 1), activation=None)
        
        self.mlp = pt_utils.Conv2d(d_in, d_in, kernel_size=(1, 1), bn=True, activation=None)
        
        self.ln = nn.LayerNorm(d_in)
        

    def forward(self, feature):
        
        att = self.fc(feature).squeeze(3)
        att = F.softmax(att, dim=2)
        
        f = self.mlp(feature).squeeze(3).permute((0, 2, 1))
        f = torch.matmul(att, f)
        
        f = self.ln(f)
        f = F.relu(f)
        f = f.permute((0, 2, 1)).unsqueeze(3)
        
        f = feature + f

        return f