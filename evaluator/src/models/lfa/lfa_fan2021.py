import src.models.pytorch_utils as pt_utils
import torch
import torch.nn as nn


class LFA(nn.Module):
    
    def __init__(self, d_in, d_out, config):
        super().__init__()
        
        self.mlp1 = pt_utils.Conv2d(d_in, d_out // 2, kernel_size=(1, 1), bn=True)
        self.local_context_learning = Local_Context_Learning(d_out // 2, d_out, config)
        self.mlp2 = pt_utils.Conv2d(d_out, d_out * 2, kernel_size=(1, 1), bn=True, activation=None)
        self.shortcut = pt_utils.Conv2d(d_in, d_out * 2, kernel_size=(1, 1), bn=True, activation=None)
        self.mlp3 = pt_utils.Conv2d(4, d_out * 2, kernel_size=(1, 1), bn=True, activation=None)
        self.mlp4 = pt_utils.Conv2d(d_out * 4, d_out * 2, kernel_size=(1, 1), bn=True)

    def forward(self, feature, xyz, neigh_idx):
        
        f_pc = self.mlp1(feature)
        f_lc, lg_volume_ratio = self.local_context_learning(xyz, f_pc, neigh_idx)
        f_lc = self.mlp2(f_lc)
        shortcut = self.shortcut(feature)
        
        f_gc = torch.cat([xyz, lg_volume_ratio], dim=-1).permute((0, 2, 1)).unsqueeze(-1)
        f_gc = self.mlp3(f_gc)
        
        f = self.mlp4(torch.cat([f_lc + shortcut, f_gc], dim=1))
        
        return f


class Local_Context_Learning(nn.Module):

    def __init__(self, d_in, d_out, config):
        super().__init__()
        
        self.mlp1 = pt_utils.Conv2d(9, d_in, kernel_size=(1, 1), bn=True)
        self.dualdis_att_pool_1 = Dualdis_Att_Pool(d_in * 2, d_out // 2, config)
        
        self.mlp2 = pt_utils.Conv2d(d_in, d_out // 2, kernel_size=(1, 1), bn=True)
        self.dualdis_att_pool_2 = Dualdis_Att_Pool(d_out, d_out, config)
        
    def forward(self, xyz, feature, neigh_idx): 

        local_rep, g_dis, lg_volume_ratio = self.local_polar_representation(xyz, neigh_idx)
        g_dis = g_dis.permute((0, 3, 1, 2))
        
        local_rep = self.mlp1(local_rep.permute((0, 3, 1, 2)))
        f_neighbours = self.gather_neighbour(feature.squeeze(-1).permute((0, 2, 1)), neigh_idx)
        f_neighbours = f_neighbours.permute((0, 3, 1, 2))
        f_cat = torch.cat([f_neighbours, local_rep], dim=1)
        f_dis = self.cal_feature_dis(feature.squeeze(-1), f_neighbours)
        f_lc = self.dualdis_att_pool_1(f_cat, f_dis, g_dis)

        local_rep = self.mlp2(local_rep)
        f_neighbours = self.gather_neighbour(f_lc.squeeze(-1).permute((0, 2, 1)), neigh_idx)
        f_neighbours = f_neighbours.permute((0, 3, 1, 2))
        
        f_cat = torch.cat([f_neighbours, local_rep], dim=1)
        f_dis = self.cal_feature_dis(f_lc.squeeze(-1), f_neighbours)
        f_lc = self.dualdis_att_pool_2(f_cat, f_dis, g_dis)

        return f_lc, lg_volume_ratio
    
    def local_polar_representation(self, xyz, neigh_idx):
        
        neighbor_xyz = self.gather_neighbour(xyz, neigh_idx)
        
        relative_info, relative_alpha, relative_beta, geometric_dis, local_volume = self.relative_pos_transforming(xyz, neigh_idx, neighbor_xyz)
        
        neighbor_mean = torch.mean(neighbor_xyz, dim=-2)
        direction = xyz - neighbor_mean
        direction_tile = direction.unsqueeze(2).repeat(1, 1, neigh_idx.shape[-1], 1)
    
        direction_alpha = torch.atan2(direction_tile[:, :, :, 1], direction_tile[:, :, :, 0]).unsqueeze(-1)
        direction_xydis = torch.sqrt(torch.sum(torch.pow(direction_tile[:, :, :, :2], 2), dim=-1))
        direction_beta = torch.atan2(direction_tile[:, :, :, 2], direction_xydis).unsqueeze(-1)
    
        angle_alpha = relative_alpha - direction_alpha
        angle_beta = relative_beta - direction_beta
        angle_updated = torch.cat([angle_alpha, angle_beta], dim=-1)
    
        local_rep = torch.cat([angle_updated, relative_info], dim=-1)
    
        global_dis = torch.sqrt(torch.sum(torch.square(xyz), dim=-1, keepdim=True))
        global_volume = torch.pow(torch.max(global_dis, dim=-1).values, 3)
        lg_volume_ratio = (local_volume / global_volume).unsqueeze(-1)
    
        return local_rep, geometric_dis, lg_volume_ratio
    
    def relative_pos_transforming(self, xyz, neigh_idx, neighbor_xyz):

        xyz_tile = xyz.unsqueeze(2).repeat(1, 1, neigh_idx.shape[-1], 1)
        relative_xyz = xyz_tile - neighbor_xyz
        relative_alpha = torch.atan2(relative_xyz[:, :, :, 1], relative_xyz[:, :, :, 0]).unsqueeze(-1)
        relative_xydis = torch.sqrt(torch.sum(torch.pow(relative_xyz[:, :, :, :2], 2), dim=-1))
        relative_beta = torch.atan2(relative_xyz[:, :, :, 2], relative_xydis).unsqueeze(-1)
        relative_dis = torch.sqrt(torch.sum(torch.pow(relative_xyz, 2), dim=-1, keepdim=True))
        
        relative_info = torch.cat([relative_dis, xyz_tile, neighbor_xyz], dim=-1)
        
        exp_dis = torch.exp(-relative_dis)
        
        local_volume = torch.pow(torch.max(torch.max(relative_dis, -1).values, -1).values, 3)
        
        return relative_info, relative_alpha, relative_beta, exp_dis, local_volume
    
    def cal_feature_dis(self, feature, f_neighbours):
        
        feature_tile = feature.unsqueeze(-1).repeat(1, 1, 1, f_neighbours.shape[-1])
        feature_dist = feature_tile - f_neighbours
        
        feature_dist = torch.mean(torch.abs(feature_dist), dim=1, keepdim=True)
        feature_dist = torch.exp(-feature_dist)
    
        return feature_dist
    
    @staticmethod
    def gather_neighbour(pc, neighbor_idx):

        batch_size = pc.shape[0]
        num_points = pc.shape[1]
        d = pc.shape[2]
        index_input = neighbor_idx.reshape(batch_size, -1)
        features = torch.gather(pc, 1, index_input.unsqueeze(-1).repeat(1, 1, d))
        features = features.reshape(batch_size, num_points, neighbor_idx.shape[-1], d)
        
        return features


class Dualdis_Att_Pool(nn.Module):

    def __init__(self, d_in, d_out, config):
        super().__init__()
        
        self.lamda = config.lfa_param
        
        self.fc = nn.Conv2d(d_in + 2, d_in, (1, 1), bias=False)
        self.mlp = pt_utils.Conv2d(d_in, d_out, kernel_size=(1, 1), bn=True)
        
    def forward(self, feature_set, f_dis, g_dis):    

        f_dis = f_dis * self.lamda
        
        concat = torch.cat([g_dis, f_dis, feature_set], dim=1)
        att_activation = self.fc(concat)
        att_scores = torch.softmax(att_activation, dim=3)
        f_lc = feature_set * att_scores
        f_lc = torch.sum(f_lc, dim=-1, keepdim=True)
        f_lc = self.mlp(f_lc)
    
        return f_lc