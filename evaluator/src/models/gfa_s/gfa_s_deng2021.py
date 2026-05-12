import torch
import torch.nn as nn
import torch.nn.functional as F

import src.models.pytorch_utils as pt_utils


class GFA(nn.Module):
    def __init__(self, d_in, d_out, config):        
        super().__init__()
        
        self.gcb = Global_Context_Block(d_in, d_out, config)
        self.mlp = pt_utils.Conv2d(d_in, d_out, kernel_size=(1, 1), bn=True)
        
        self.nlb1 = Non_Local_Block(d_in, d_out, config)
        self.nlb2 = Non_Local_Block(d_in, d_out, config)

    def forward(self, feature):
        
        g_f = self.gcb(feature)
        
        f = self.mlp(feature * g_f)
        f = self.nlb1(f)
        f = self.nlb2(f)
        
        return f


class Global_Context_Block(nn.Module):
    def __init__(self, d_in, d_out, config):        
        super().__init__()
        
        self.mlp1 = pt_utils.Conv2d(d_in, 1, kernel_size=(1, 1), bn=True)
        self.fc1 = nn.Linear(d_in, d_in, bias=False)
        self.ln = nn.LayerNorm(d_in)
        self.fc2 = nn.Linear(d_in, d_in, bias=False)
        
    def forward(self, feature):
        
        num_points = feature.shape[2]
        
        att = self.mlp1(feature)
        att = F.softmax(att, dim=2)
        
        g_f = torch.matmul(feature.squeeze(3), att.squeeze(3).permute((0, 2, 1)))
        g_f = F.relu(self.ln(self.fc1(g_f.squeeze(2))))
        g_f = self.fc2(g_f)
        g_f = g_f.unsqueeze(2).unsqueeze(3).repeat(1, 1, num_points, 1)
        
        return g_f


class Non_Local_Block(nn.Module):
    def __init__(self, d_in, d_out, config):        
        super().__init__()
        
        self.num_slice = config.gfa_s_param
        
        self.mlp_q1 = pt_utils.Conv2d(d_in, d_out, kernel_size=(1, 1), bn=True)
        self.mlp_k1 = pt_utils.Conv2d(d_in, d_out, kernel_size=(1, 1), bn=True)
        self.mlp_v1 = pt_utils.Conv2d(d_in, d_out, kernel_size=(1, 1), bn=True)
        
        self.mlp_q2 = pt_utils.Conv2d(d_in, d_out, kernel_size=(1, 1), bn=True)
        self.mlp_k2 = pt_utils.Conv2d(d_in, d_out, kernel_size=(1, 1), bn=True)
        self.mlp_v2 = pt_utils.Conv2d(d_in, d_out, kernel_size=(1, 1), bn=True)
    
    def forward(self, feature):
        
        batch_size = feature.shape[0]
        d = feature.shape[1]
        num_points = feature.shape[2]
        
        f = feature.reshape(batch_size, d, num_points // self.num_slice, self.num_slice)
        q1 = self.mlp_q1(f).permute(0, 2, 3, 1)
        k1 = self.mlp_k1(f).permute(0, 2, 3, 1)
        v1 = self.mlp_v1(f).permute(0, 2, 3, 1)
        w1 = torch.matmul(q1, k1.permute(0, 1, 3, 2))
        w1 = F.softmax(w1, dim=-1)
        
        f = torch.matmul(w1, v1)
        
        f = f.permute((0, 3, 2, 1))
        q2 = self.mlp_q2(f).permute(0, 2, 3, 1)
        k2 = self.mlp_k2(f).permute(0, 2, 3, 1)
        v2 = self.mlp_v2(f).permute(0, 2, 3, 1)
        w2 = torch.matmul(q2, k2.permute(0, 1, 3, 2))
        w2 = F.softmax(w2, dim=-1)
        
        f = torch.matmul(w2, v2)
        
        f = f.reshape(batch_size, num_points, 1, d).permute((0, 3, 1, 2))
        
        return f
    