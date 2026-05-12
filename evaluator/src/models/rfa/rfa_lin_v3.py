import torch
import torch.nn as nn

import src.models.pytorch_utils as pt_utils


class RFA(nn.Module):
    
    def __init__(self, d_in, d_out, config):
        super().__init__()
        
        self.pooling = config.rfa_pooling
        
        self.mlp1 = pt_utils.Conv2d(10, d_in, kernel_size=(1, 1), bn=True)
        self.mlp2 = pt_utils.Conv2d(d_in, d_in, kernel_size=(1, 1), bn=True)
        self.mlp3 = pt_utils.Conv2d(2 * d_in , d_in, kernel_size=(1, 1), bn=True)
        self.mlp4 = pt_utils.Conv2d(d_in, d_in, kernel_size=(1, 1), bn=True)

    def forward(self, feature, xyz, neigh_idx):
        
        f_xyz = self.relative_pos_encoding(xyz, neigh_idx)
        f_xyz = f_xyz.permute((0, 3, 1, 2))
        f_xyz = self.mlp1(f_xyz)
        
        f = self.mlp2(feature)
        f_neighbours = self.gather_neighbour(f.squeeze(-1).permute((0, 2, 1)), neigh_idx)
        f_neighbours = f_neighbours.permute((0, 3, 1, 2))
        
        f_concat = torch.cat([f_neighbours, f_xyz], dim=1)
        f_concat = self.mlp3(f_concat)
        
        if self.pooling == 'max':
            f_concat, _ = torch.max(f_concat, dim=3, keepdim=True)
        elif self.pooling == 'mean':
            f_concat = torch.mean(f_concat, dim=3, keepdim=True)
        
        f = feature + f_concat
        f = self.mlp4(f)
        
        return f
    
    def relative_pos_encoding(self, xyz, neigh_idx):
        
        neighbor_xyz = self.gather_neighbour(xyz, neigh_idx)

        xyz_tile = xyz.unsqueeze(2).repeat(1, 1, neigh_idx.shape[-1], 1)
        relative_xyz = xyz_tile - neighbor_xyz

        relative_dis = torch.sqrt(torch.sum(torch.pow(relative_xyz, 2), dim=-1, keepdim=True))

        relative_feature = torch.cat([relative_dis, relative_xyz, xyz_tile, neighbor_xyz], dim=-1)
        
        return relative_feature
    
    @staticmethod
    def gather_neighbour(pc, neighbor_idx):

        batch_size = pc.shape[0]
        num_points = pc.shape[1]
        d = pc.shape[2]
        index_input = neighbor_idx.reshape(batch_size, -1)
        features = torch.gather(pc, 1, index_input.unsqueeze(-1).repeat(1, 1, d))
        features = features.reshape(batch_size, num_points, neighbor_idx.shape[-1], d)
        
        return features