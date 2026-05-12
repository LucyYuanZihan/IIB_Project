import src.models.pytorch_utils as pt_utils
import torch
import torch.nn as nn


class LFA(nn.Module):
    
    def __init__(self, d_in, d_out, config):
        super().__init__()
        
        self.block1 = Block(d_in, d_out, config)
        
        d_in = d_out
        d_out = 2 * d_out
        
        self.block2 = Block(d_in, d_out, config)

    def forward(self, feature, xyz, neigh_idx):
        
        feature = self.block1(feature, xyz, neigh_idx)
        feature = self.block2(feature, xyz, neigh_idx)
        
        return feature


class Block(nn.Module):
    
    def __init__(self, d_in, d_out, config):
        super().__init__()
        
        self.d_geo = 10
        self.m = config.lfa_param
        self.k_n = config.k_n
        
        self.mlp1 = pt_utils.Conv2d(self.d_geo, d_out // self.m, kernel_size=(1, 1), bn=True)
        self.mlp2 = pt_utils.Conv2d(d_in, d_out // self.m, kernel_size=(1, 1), bn=True)
        self.mlp3 = pt_utils.Conv2d(d_out // self.m * 2, d_out // self.m, kernel_size=(1, 1), bn=True)
        
        self.fc = nn.Linear(self.k_n, 1)
        
        self.mlp4 = pt_utils.Conv2d(d_out // self.m, d_out, kernel_size=(1, 1), bn=True)
        
        self.shortcut = pt_utils.Conv2d(d_in, d_out, kernel_size=(1, 1), bn=True)

    def forward(self, feature, xyz, neigh_idx):
        
        f_xyz = self.relative_pos_encoding(xyz, neigh_idx)
        f_xyz = f_xyz.permute((0, 3, 1, 2))
        f_xyz = self.mlp1(f_xyz)
        
        f_pc = self.mlp2(feature).squeeze(-1).permute((0, 2, 1))
        f_pc = self.gather_neighbour(f_pc, neigh_idx)
        f_pc = f_pc.permute((0, 3, 1, 2))
        
        f_concat = torch.cat([f_pc, f_xyz], dim=1)
        f_concat = self.mlp3(f_concat)
        
        f_concat = f_concat.permute((0, 2, 3, 1))
        B, N, k, d = f_concat.shape
        f_concat_score = f_concat.reshape(B * N, k, d).permute((0, 2, 1))
        f_concat_score = self.fc(f_concat_score).reshape(B, N, d)
        f_concat_score = torch.softmax(f_concat_score, dim=-1)
        
        f_concat, _ = torch.max(f_concat, dim=2, keepdim=True)
        
        f_concat_weighted = f_concat * f_concat_score.unsqueeze(2)

        f_concat = f_concat + f_concat_weighted
        f_concat = f_concat.permute((0, 3, 1, 2))
        
        f_concat = self.mlp4(f_concat)
        
        shortcut = self.shortcut(feature)
        f_concat = f_concat + shortcut
        
        return f_concat
        
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