import torch
import torch.nn as nn

import src.models.pytorch_utils as pt_utils


class RFA(nn.Module):
    
    def __init__(self, d_in, d_out, config):
        super().__init__()
        
        self.param = config.rfa_param
        self.pooling = config.rfa_pooling
        
        self.tnet = TNet()
        
        self.mlp1 = pt_utils.Conv2d(3, 3, kernel_size=(1, 1), bn=True)
        self.mlp2 = pt_utils.Conv2d(d_in, d_in // self.param, kernel_size=(1, 1), bn=True)
        self.mlp3 = pt_utils.Conv2d(d_in // self.param + 3, d_in // self.param, kernel_size=(1, 1), bn=True)
        self.mlp4 = pt_utils.Conv2d(2 * d_in // self.param, d_in, kernel_size=(1, 1), bn=True)

    def forward(self, feature, xyz, neigh_idx):
        
        t = self.tnet(xyz)
        f_xyz = torch.bmm(xyz, t)
        
        f_xyz = f_xyz.permute((0, 2, 1)).unsqueeze(-1)
        f_xyz = self.mlp1(f_xyz)
        f_xyz = self.gather_neighbour(f_xyz.squeeze(-1).permute((0, 2, 1)), neigh_idx)
        f_xyz = f_xyz.permute((0, 3, 1, 2))
        
        f = self.mlp2(feature)
        f_neighbours = self.gather_neighbour(f.squeeze(-1).permute((0, 2, 1)), neigh_idx)
        f_neighbours = f_neighbours.permute((0, 3, 1, 2))
        
        f_concat = torch.cat([f_neighbours, f_xyz], dim=1)
        f_concat = self.mlp3(f_concat)
        
        if self.pooling == 'max':
            f_concat, _ = torch.max(f_concat, dim=3, keepdim=True)
        elif self.pooling == 'mean':
            f_concat = torch.mean(f_concat, dim=3, keepdim=True)
        
        f = torch.cat([f, f_concat], dim=1)
        f = self.mlp4(f)
        
        return f
    
    @staticmethod
    def gather_neighbour(pc, neighbor_idx):

        batch_size = pc.shape[0]
        num_points = pc.shape[1]
        d = pc.shape[2]
        index_input = neighbor_idx.reshape(batch_size, -1)
        features = torch.gather(pc, 1, index_input.unsqueeze(-1).repeat(1, 1, d))
        features = features.reshape(batch_size, num_points, neighbor_idx.shape[-1], d)
        
        return features


class TNet(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.conv1 = pt_utils.Conv2d(3, 16, kernel_size=(1, 1), bn=True)
        self.conv2 = pt_utils.Conv2d(16, 64, kernel_size=(1, 1), bn=True)
        
        self.fc1 = pt_utils.Conv2d(64, 16, kernel_size=(1, 1), bn=True)
        self.fc2 = pt_utils.Conv2d(16, 9, kernel_size=(1, 1), activation=None)
        
        self.dia = nn.Parameter(
            torch.FloatTensor([1, 0, 0, 0, 1, 0, 0, 0, 1]).unsqueeze(0).unsqueeze(-1).unsqueeze(-1),
            requires_grad=False
            )
    
    def forward(self, xyz):
        
        t = xyz.unsqueeze(2).permute((0, 3, 1, 2))
        t = self.conv1(t)
        t = self.conv2(t)
        t, _ = torch.max(t, dim=2, keepdim=True)
        t = self.fc1(t)
        t = self.fc2(t)
        t = t + self.dia
        
        t = t.squeeze(-1).squeeze(-1).reshape(-1, 3, 3)
        
        return t
        
        
    