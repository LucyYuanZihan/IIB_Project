import torch
import torch.nn as nn
import torch.nn.functional as F

import src.models.pytorch_utils as pt_utils


class GFA(nn.Module):
    def __init__(self, d_in, d_out, config):        
        super().__init__()
        
        self.blo_0 = Blo_0(d_in, d_out, config)
        self.blo_1 = Blo_1(d_in, d_out, config)
        self.mlp = pt_utils.Conv2d(3 * d_out, d_out, kernel_size=(1, 1), activation=None)

    def forward(self, feature):
        
        f0 = self.blo_0(feature)
        f1 = self.blo_1(feature)
        f = torch.cat((f0, f1, feature), dim=1)
        f = self.mlp(f)
        
        return f


class Blo_0(nn.Module):
    def __init__(self, d_in, d_out, config):        
        super().__init__()
        
        self.mlp0 = pt_utils.Conv2d(d_in, 1, kernel_size=(1, 1), activation=None)
        self.mlp1 = pt_utils.Conv2d(d_in, d_in, kernel_size=(1, 1), activation=None)
        
        self.mlp2 = pt_utils.Conv2d(d_in, d_in, kernel_size=(1, 1), activation=None)
        self.ln = nn.LayerNorm(d_in)
        
        self.alpha = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.beta = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.alpha.data.fill_(0)
        self.beta.data.fill_(0)

    def forward(self, feature):
        
        bb = self.mlp0(feature)
        dd = torch.bmm(feature.squeeze(-1), bb.squeeze(-1).permute((0, 2, 1)))
        dd = F.softmax(dd, dim=2)
        cc = self.mlp1(feature)
        ee = torch.bmm(cc.squeeze(-1).permute((0, 2, 1)), dd).permute((0, 2, 1)).unsqueeze(-1)
        ee = F.softmax(ee, dim=2)
        
        dd = dd.unsqueeze(-1)
        dd = self.mlp2(dd)
        dd = F.relu(self.ln(dd.permute(0, 2, 3, 1))).permute((0, 3, 1, 2))
        
        f = self.alpha * ee + self.beta * dd + feature
        
        return f
    
    
class Blo_1(nn.Module):
    def __init__(self, d_in, d_out, config):        
        super().__init__()
        
        self.param = config.gfa_s_param
        self.mlp0 = pt_utils.Conv2d(d_in, d_in, kernel_size=(3, 1), activation=None)

    def forward(self, feature):
        
        oo = torch.bmm(feature.squeeze(-1), feature.squeeze(-1).permute((0, 2, 1)))
        oo = F.softmax(oo, dim=1)
        pp = torch.bmm(feature.squeeze(-1).permute((0, 2, 1)), oo)
        pp = pp.permute((0, 2, 1)).unsqueeze(-1)
        
        f = self.param * pp + feature
        
        return f