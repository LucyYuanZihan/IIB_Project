import src.models.pytorch_utils as pt_utils
import torch
import torch.nn as nn


class LFA(nn.Module):
    
    def __init__(self, d_in, d_out, config):
        super().__init__()
        
        self.config = config
        
        self.block1 = Block(d_in, d_out, self.config)
        self.block2 = Block(d_out, 2 * d_out, self.config)

    def forward(self, feature, xyz, neigh_idx):
        
        f = self.block1(xyz, feature, neigh_idx)
        f = self.block2(xyz, f, neigh_idx)
        
        return f


class Block(nn.Module):
    
    def __init__(self, d_in, d_out, config):
        super().__init__()
        
        self.config = config
        
        self.mlp1 = pt_utils.Conv2d(d_in, d_out // 2, kernel_size=(1, 1), bn=True)
        self.fa_res = FA_Res(d_out // 2, self.config)
        self.mlp2 = pt_utils.Conv2d(d_out // 2, d_out, kernel_size=(1, 1), bn=True)
        
        self.shortcut = pt_utils.Conv2d(d_in, d_out, kernel_size=(1, 1), bn=True)
        
    def forward(self, xyz, feature, neigh_idx):
        
        f = self.mlp1(feature)
        f = self.fa_res(xyz, f, neigh_idx)
        f = self.mlp2(f)
        
        shortcut = self.shortcut(feature)
        f = shortcut + f
        
        return f
        
        
class FA_Res(nn.Module):
    
    def __init__(self, d, config):
        super().__init__()
        
        self.config = config
        self.n = self.config.lfa_param
        
        self.mlp1 = pt_utils.Conv2d(10, d, kernel_size=(1, 1), bn=True)
        
        self.mlp = pt_utils.Conv2d(2 * d, 2 * d, kernel_size=(1, 1), bn=True)
        
        self.mlp2 = pt_utils.Conv2d(3 * d, d, kernel_size=(1, 1), bn=True)
        
        self.mlp3 = nn.ModuleList()
        for i in range(self.n):
            self.mlp3.append(pt_utils.Conv2d(d, d, kernel_size=(1, 1), bn=True))
        self.mlp4 = nn.ModuleList()
        for i in range(self.n):
            self.mlp4.append(pt_utils.Conv2d(d, d, kernel_size=(1, 1), bn=True))

    def forward(self, xyz, feature, neigh_idx):
        
        f_xyz = self.relative_pos_encoding(xyz, neigh_idx)
        f_xyz = f_xyz.permute((0, 3, 1, 2))
        f_xyz = self.mlp1(f_xyz)
        
        f_neighbours = self.gather_neighbour(feature.squeeze(-1).permute((0, 2, 1)), neigh_idx)
        f_neighbours = f_neighbours.permute((0, 3, 1, 2))
        
        f_cat = torch.cat([f_neighbours, f_xyz], dim=1)
        
        f_cat = self.mlp(f_cat)
        
        f_tile = feature.repeat(1, 1, 1, neigh_idx.shape[-1])
        f_cat = torch.cat([f_cat, f_tile], dim=1)
        f = self.mlp2(f_cat)
        
        for i in range(self.n):
            f_res = self.mlp3[i](f)
            f = f + f_res
        
        pool_f = f.max(dim=3, keepdim=True)[0]
        
        for i in range(self.n):
            pool_f_res = self.mlp4[i](pool_f)
            pool_f = pool_f + pool_f_res
        
        return pool_f
    
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
        features = torch.gather(pc, 1, index_input.unsqueeze(-1).repeat(1, 1, pc.shape[2]))
        features = features.reshape(batch_size, num_points, neighbor_idx.shape[-1], d)
        
        return features