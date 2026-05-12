import torch
import torch.nn as nn
from typing import Optional


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.z_size = config['z_size']
        self.use_bias = config['model']['E']['use_bias']
        self.relu_slope = config['model']['E']['relu_slope']

        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=6, out_channels=64, kernel_size=1, bias=self.use_bias), 
            nn.ReLU(inplace=True),

            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=1, bias=self.use_bias),
            nn.ReLU(inplace=True),

            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=1, bias=self.use_bias),
            nn.ReLU(inplace=True),

            nn.Conv1d(in_channels=256, out_channels=512, kernel_size=1, bias=self.use_bias),
            nn.ReLU(inplace=True),

            nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, bias=self.use_bias),
        )

        self.fc = nn.Sequential(
            nn.Linear(512, 512, bias=True),
            nn.ReLU(inplace=True)
        )

        self.mu_layer = nn.Linear(512, self.z_size, bias=True)
        self.std_layer = nn.Linear(512, self.z_size, bias=True)

    def reparameterize(self, mu, logvar):
        std = torch.exp(logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, x):
        output = self.conv(x)
        output2 = output.max(dim=2)[0]
        logit = self.fc(output2)
        mu = self.mu_layer(logit)
        logvar = self.std_layer(logit)
        z = self.reparameterize(mu, logvar)
        return z, mu, torch.exp(logvar)

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False



import torch.nn.functional as F


# -------------------------
# Helpers: gather + grouping
# -------------------------

def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """
    points: (B, N, C)
    idx:    (B, S) or (B, S, K)
    return: (B, S, C) or (B, S, K, C)
    """
    B = points.shape[0]
    device = points.device

    if idx.dim() == 2:
        # (B,S)
        batch_indices = torch.arange(B, device=device).view(B, 1).expand_as(idx)
        return points[batch_indices, idx, :]  # (B,S,C)

    elif idx.dim() == 3:
        # (B,S,K)
        B, S, K = idx.shape
        batch_indices = torch.arange(B, device=device).view(B, 1, 1).expand(B, S, K)
        return points[batch_indices, idx, :]  # (B,S,K,C)

    else:
        raise ValueError(f"idx must be (B,S) or (B,S,K), got {idx.shape}")


def knn_group(xyz: torch.Tensor, new_xyz: torch.Tensor, K: int) -> torch.Tensor:
    """
    kNN grouping via cdist (pure PyTorch).
    xyz:     (B, N, 3)
    new_xyz: (B, S, 3) centroids
    return idx: (B, S, K) indices in [0..N-1]
    """
    # (B,S,N) pairwise distances
    dists = torch.cdist(new_xyz, xyz, p=2)  # can be heavy but OK for N~8192, S~2048
    idx = torch.topk(dists, k=K, dim=-1, largest=False, sorted=False).indices
    return idx


def fps_torch(xyz: torch.Tensor, npoint: int, random_start: bool = False) -> torch.Tensor:
    """
    Farthest Point Sampling (FPS) in pure PyTorch.
    xyz: (B, N, 3)
    return centroids_idx: (B, npoint)
    NOTE: O(B*N*npoint). With B small and N~8192 it's usually OK, but slower than PyTorch3D.
    """
    B, N, _ = xyz.shape
    device = xyz.device
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)

    # distances to selected set
    dist = torch.full((B, N), 1e10, device=device)

    if random_start:
        farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    else:
        farthest = torch.zeros((B,), dtype=torch.long, device=device)

    batch_indices = torch.arange(B, device=device, dtype=torch.long)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid_xyz = xyz[batch_indices, farthest, :].view(B, 1, 3)   # (B,1,3)
        d = torch.sum((xyz - centroid_xyz) ** 2, dim=-1)               # (B,N)
        dist = torch.minimum(dist, d)
        farthest = torch.max(dist, dim=-1).indices

    return centroids


# -------------------------
# PointNet++ blocks
# -------------------------

class SharedMLP2d(nn.Module):
    """
    Shared MLP implemented as 1x1 Conv2d stack.
    Input:  (B, Cin, S, K)
    Output: (B, Cout, S, K)
    """
    def __init__(self, in_ch: int, mlp_channels, use_bias=True):
        super().__init__()
        layers = []
        last = in_ch
        for out in mlp_channels:
            layers.append(nn.Conv2d(last, out, kernel_size=1, bias=use_bias))
            layers.append(nn.ReLU(inplace=True))
            last = out
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class SetAbstractionKNN(nn.Module):
    """
    PointNet++ Set Abstraction (single-scale grouping) using FPS + kNN.

    Inputs:
      xyz:  (B, N, 3)
      feat: (B, N, Cin) or None
    Outputs:
      new_xyz:  (B, S, 3)
      new_feat: (B, Cout, S)   (channels-first, convenient for later Conv1d/global pool)
    """
    def __init__(self, npoint: int, nsample: int, in_feat_ch: int, mlp_channels, use_bias=True,
                 use_pytorch3d_if_available=True):
        super().__init__()
        self.npoint = npoint
        self.nsample = nsample
        self.in_feat_ch = in_feat_ch

        # local input = [rel_xyz(3) + feat(Cin)]
        self.mlp = SharedMLP2d(in_ch=3 + in_feat_ch, mlp_channels=mlp_channels, use_bias=use_bias)

        self.use_pytorch3d = False
        self._p3d = None
        if use_pytorch3d_if_available:
            try:
                from pytorch3d.ops import sample_farthest_points, knn_points, knn_gather
                self.use_pytorch3d = True
                self._p3d = (sample_farthest_points, knn_points, knn_gather)
            except Exception:
                self.use_pytorch3d = False
                self._p3d = None

    def forward(self, xyz: torch.Tensor, feat: Optional[torch.Tensor]):
        B, N, _ = xyz.shape
        device = xyz.device

        # --- 1) Sampling (FPS) ---
        if self.use_pytorch3d:
            sample_farthest_points, _, _ = self._p3d
            new_xyz, fps_idx = sample_farthest_points(xyz, K=self.npoint, random_start_point=False)
            # fps_idx: (B, S)
        else:
            fps_idx = fps_torch(xyz, self.npoint, random_start=False)  # (B,S)
            new_xyz = index_points(xyz, fps_idx)                       # (B,S,3)

        # --- 2) Grouping (kNN) ---
        if self.use_pytorch3d:
            _, knn_points, _ = self._p3d
            knn = knn_points(new_xyz, xyz, K=self.nsample, return_nn=False)
            idx = knn.idx  # (B,S,K)
        else:
            idx = knn_group(xyz, new_xyz, K=self.nsample)  # (B,S,K)

        grouped_xyz = index_points(xyz, idx)                 # (B,S,K,3)
        grouped_xyz_rel = grouped_xyz - new_xyz.unsqueeze(2) # (B,S,K,3)

        if feat is None or self.in_feat_ch == 0:
            grouped_feat = torch.zeros((B, self.npoint, self.nsample, 0), device=device, dtype=xyz.dtype)
        else:
            grouped_feat = index_points(feat, idx)  # (B,S,K,Cin)

        local_input = torch.cat([grouped_xyz_rel, grouped_feat], dim=-1)  # (B,S,K,3+Cin)

        # --- 3) Local PointNet (shared MLP + max pool over neighbors) ---
        local_input = local_input.permute(0, 3, 1, 2).contiguous()  # (B, 3+Cin, S, K)
        local_feat = self.mlp(local_input)                          # (B, Cout, S, K)
        new_feat = torch.max(local_feat, dim=3)[0]                  # (B, Cout, S)

        return new_xyz, new_feat


# -------------------------
# Your requested encoder: PointNet2VAEEncoder
# -------------------------

class PointNet2VAEEncoder(nn.Module):
    """
    PointNet++-style encoder that drops into your existing VAE -> Hypernetwork pipeline.

    Contract matches your current Encoder:
      forward(x: (B,6,N)) -> (z, mu, std)
    """
    def __init__(self, config):
        super().__init__()

        self.z_size = config["z_size"]
        self.use_bias = config["model"]["E"].get("use_bias", True)
        self.relu_slope = config["model"]["E"].get("relu_slope", 0.0)  # not used (keeping for consistency)

        # Your current Encoder expects 6 channels per point: (x,y,z, f1,f2,f3)
        self.in_channels = 6
        self.in_feat_ch = max(self.in_channels - 3, 0)

        # Allow override from config if you want
        pn2_cfg = config.get("model", {}).get("E", {}).get("pn2", {})
        npoint1  = int(pn2_cfg.get("npoint1", 2048))
        npoint2  = int(pn2_cfg.get("npoint2", 512))
        npoint3  = int(pn2_cfg.get("npoint3", 128))
        nsample1 = int(pn2_cfg.get("nsample1", 32))
        nsample2 = int(pn2_cfg.get("nsample2", 32))
        nsample3 = int(pn2_cfg.get("nsample3", 32))

        # Set Abstraction hierarchy: 8192 -> 2048 -> 512 -> 128 -> global
        self.sa1 = SetAbstractionKNN(
            npoint=npoint1, nsample=nsample1,
            in_feat_ch=self.in_feat_ch,
            mlp_channels=[64, 64, 128],
            use_bias=self.use_bias,
            use_pytorch3d_if_available=True
        )
        self.sa2 = SetAbstractionKNN(
            npoint=npoint2, nsample=nsample2,
            in_feat_ch=128,
            mlp_channels=[128, 128, 256],
            use_bias=self.use_bias,
            use_pytorch3d_if_available=True
        )
        self.sa3 = SetAbstractionKNN(
            npoint=npoint3, nsample=nsample3,
            in_feat_ch=256,
            mlp_channels=[256, 512, 512],
            use_bias=self.use_bias,
            use_pytorch3d_if_available=True
        )

        # Optional 1x1 Conv on the last level before global pooling
        self.post = nn.Sequential(
            nn.Conv1d(512, 512, kernel_size=1, bias=self.use_bias),
            nn.ReLU(inplace=True),
        )

        # Same VAE head structure as your current Encoder
        self.fc = nn.Sequential(
            nn.Linear(512, 512, bias=True),
            nn.ReLU(inplace=True)
        )
        self.mu_layer  = nn.Linear(512, self.z_size, bias=True)
        self.std_layer = nn.Linear(512, self.z_size, bias=True)

    def reparameterize(self, mu, logvar):
        # Keep exactly the same behavior as your current Encoder for compatibility
        std = torch.exp(logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, x):
        """
        x: (B, 6, N)
        """
        # Convert to (B, N, C)
        x_bn = x.transpose(1, 2).contiguous()  # (B,N,6)
        xyz = x_bn[:, :, :3]                   # (B,N,3)
        feat = x_bn[:, :, 3:] if self.in_feat_ch > 0 else None  # (B,N,3) or None

        # PointNet++ hierarchy
        l1_xyz, l1_feat = self.sa1(xyz, feat)                          # l1_feat: (B,128,S1)
        l2_xyz, l2_feat = self.sa2(l1_xyz, l1_feat.transpose(1, 2))    # needs (B,S1,128)
        l3_xyz, l3_feat = self.sa3(l2_xyz, l2_feat.transpose(1, 2))    # l3_feat: (B,512,S3)

        # Global aggregation (still produces a single 512-D vector per point cloud)
        g = self.post(l3_feat)                 # (B,512,S3)
        g = torch.max(g, dim=2)[0]             # (B,512)

        # VAE head
        logit = self.fc(g)                     # (B,512)
        mu = self.mu_layer(logit)              # (B,z)
        logvar = self.std_layer(logit)         # (B,z)
        z = self.reparameterize(mu, logvar)    # (B,z)

        return z, mu, torch.exp(logvar)

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False
