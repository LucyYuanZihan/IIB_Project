import src.models.pytorch_utils as pt_utils
import torch
import torch.nn as nn
import torch.nn.functional as F


class LFA(nn.Module):
    
    def __init__(self, d_in, d_out, config):
        super().__init__()
        
        self.config = config
        
        self.conv1 = pt_utils.Conv2d(d_in, 2 * d_out, kernel_size=(1, 1), bn=True)
        
        self.linear_q = pt_utils.Conv2d(2 * d_out, 2 * d_out, kernel_size=(1, 1), activation=None)
        self.linear_k = pt_utils.Conv2d(2 * d_out, 2 * d_out, kernel_size=(1, 1), activation=None)
        self.linear_v = pt_utils.Conv2d(2 * d_out, 2 * d_out, kernel_size=(1, 1), activation=None)
        
        self.linear_p = nn.Sequential(
            pt_utils.Conv2d(3, 3, kernel_size=(1, 1), bn=True),
            pt_utils.Conv2d(3, 2 * d_out, kernel_size=(1, 1), activation=None),
        )
        
        self.linear_w = nn.Sequential(
            nn.BatchNorm2d(2 * d_out),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            pt_utils.Conv2d(2 * d_out, 2 * d_out // 8, kernel_size=(1, 1), bn=True),
            pt_utils.Conv2d(2 * d_out // 8, 2 * d_out // 8, kernel_size=(1, 1), activation=None),
        )
        
        self.bn = nn.BatchNorm2d(2 * d_out)
        self.conv2 = pt_utils.Conv2d(2 * d_out, 2 * d_out, kernel_size=(1, 1), bn=True, activation=None)

    def forward(self, feature, xyz, neigh_idx):
        
        f = self.conv1(feature)
        x = f
        
        x_q = self.linear_q(x)
        x_k = self.gather_neighbour(self.linear_k(x).squeeze(3).permute((0, 2, 1)), neigh_idx).permute((0, 3, 1, 2))
        x_v = self.gather_neighbour(self.linear_v(x).squeeze(3).permute((0, 2, 1)), neigh_idx).permute((0, 3, 1, 2))
        
        p_r = self.gather_neighbour(xyz, neigh_idx).permute((0, 3, 1, 2))
        p_r = self.linear_p(p_r)
        
        r_qk = x_k - x_q + p_r
        
        w = self.linear_w(r_qk)
        w = F.softmax(w, dim=-1)
        
        batch_size = x_v.shape[0]
        d = x_v.shape[1]
        num_points = x_v.shape[2]
        k = x_v.shape[3]
        
        xp = (x_v + p_r).permute(0, 2, 3, 1).reshape(batch_size, num_points, k, 8, d // 8)
        w = w.permute(0, 2, 3, 1).unsqueeze(3)
        x = (xp * w).sum(2).reshape(batch_size, num_points, d)
        x = x.permute((0, 2, 1)).unsqueeze(-1)
        
        x = F.leaky_relu(self.bn(x), negative_slope=0.2)
        x = self.conv2(x)
        
        return F.leaky_relu(f + x, negative_slope=0.2)
        
    @staticmethod
    def gather_neighbour(pc, neighbor_idx):

        batch_size = pc.shape[0]
        num_points = pc.shape[1]
        d = pc.shape[2]
        index_input = neighbor_idx.reshape(batch_size, -1)
        features = torch.gather(pc, 1, index_input.unsqueeze(-1).repeat(1, 1, d))
        features = features.reshape(batch_size, num_points, neighbor_idx.shape[-1], d)
        
        return features