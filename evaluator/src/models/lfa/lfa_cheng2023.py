import src.models.pytorch_utils as pt_utils
import torch
import torch.nn as nn
import torch.nn.functional as F


class LFA(nn.Module):
    
    def __init__(self, d_in, d_out, config):
        super().__init__()
        
        self.config = config
        
        self.mlp1 = pt_utils.Conv2d(d_in, d_out // 2, kernel_size=(1, 1), bn=True)
        self.block1 = Block(d_out // 2, d_out // 2, self.config)
        self.block2 = Block(d_out // 2, d_out, self.config)
        self.mlp2 = pt_utils.Conv2d(d_out, 2 * d_out, kernel_size=(1, 1), bn=True)
        self.shortcut = pt_utils.Conv2d(d_in, 2 * d_out, kernel_size=(1, 1), bn=True)

    def forward(self, feature, xyz, neigh_idx):
        
        f = self.mlp1(feature)
        f = self.block1(xyz, f, neigh_idx)
        f = self.block2(xyz, f, neigh_idx)
        f = self.mlp2(f)
        
        shortcut = self.shortcut(feature)
        
        return F.leaky_relu(f + shortcut, negative_slope=0.2)


class Block(nn.Module):
    
    def __init__(self, d_in, d_out, config):
        super().__init__()
        
        self.config = config
        
        self.mlp1 = pt_utils.Conv2d(3, d_in, kernel_size=(1, 1), bn=True)
        self.mlp2 = pt_utils.Conv2d(d_in, d_out, kernel_size=(1, 1), bn=True)

        
    def forward(self, xyz, feature, neigh_idx):
        
        f_xyz = self.relative_pos_encoding(xyz, neigh_idx)
        f_xyz = f_xyz.permute((0, 3, 1, 2))
        f_xyz = self.mlp1(f_xyz)
        
        f_neighbours = self.gather_neighbour(feature.squeeze(-1).permute((0, 2, 1)), neigh_idx)
        f_neighbours = f_neighbours.permute((0, 3, 1, 2))
        
        f = f_xyz * f_neighbours
        f = torch.mean(f, dim=-1, keepdim=True)
        f = self.mlp2(f)
        
        return f
        
    def relative_pos_encoding(self, xyz, neigh_idx):
        
        neighbor_xyz = self.gather_neighbour(xyz, neigh_idx)
        xyz_tile = xyz.unsqueeze(2).repeat(1, 1, neigh_idx.shape[-1], 1)
        relative_xyz = xyz_tile - neighbor_xyz
        
        return relative_xyz

    @staticmethod
    def gather_neighbour(pc, neighbor_idx):

        batch_size = pc.shape[0]
        num_points = pc.shape[1]
        d = pc.shape[2]
        index_input = neighbor_idx.reshape(batch_size, -1)
        features = torch.gather(pc, 1, index_input.unsqueeze(-1).repeat(1, 1, d))
        features = features.reshape(batch_size, num_points, neighbor_idx.shape[-1], d)
        
        return features