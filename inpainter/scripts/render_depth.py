import numpy as np
import os
from os.path import join, exists
import matplotlib.pyplot as plt
import pandas as pd
import torch
import scipy.ndimage as ndimage
from torch.utils.data import DataLoader
from datetime import datetime
from src.render.nerf_helpers import *
from itertools import chain
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.utils import *

import trimesh, mcubes

from src.datasets.seg2tunnel import NeRFShapeNetDataset

from src.models.encoder import Encoder
from src.models.nerf import NeRF
from src.models.resnet import resnet18
from hypnettorch.hnets.chunked_mlp_hnet import ChunkedHMLP

import open3d as o3d

import argparse
import imageio
import json

#Needed for workers for dataloader
from torch.multiprocessing import Pool, Process, set_start_method
set_start_method('spawn', force=True)

import math

import os, re

def get_sample_id(obj_path, fallback: str):
    # obj_path might be ['...'], tensor([..]), or ''
    if isinstance(obj_path, (list, tuple)):
        obj_path = obj_path[0] if len(obj_path) > 0 else ""
    if isinstance(obj_path, torch.Tensor):
        obj_path = obj_path[0].item() if obj_path.numel() > 0 else ""
    s = str(obj_path)

    # strip directories + extension
    s = os.path.splitext(os.path.basename(s))[0]

    # sanitize (keep letters/numbers/._-)
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("_")

    return s if s else fallback
def cart2sph(x,y,z):
    XsqPlusYsq = x**2 + y**2
    r = math.sqrt(XsqPlusYsq + z**2)               # r
    elev = math.atan2(z,math.sqrt(XsqPlusYsq))     # theta
    az = math.atan2(y,x)                           # phi
    return r, elev, az


# -----------------------------------------------------------------------------
# Dataset-consistent camera helpers (match generate_fixed.py)
# -----------------------------------------------------------------------------
def look_at_c2w_np(
    origin: np.ndarray,
    target: np.ndarray,
    up_hint: np.ndarray = np.array([0, 1, 0], dtype=np.float32),
) -> np.ndarray:
    """Blender/NeRF-style camera-to-world (camera looks along -Z)."""
    origin = origin.astype(np.float32)
    target = target.astype(np.float32)
    up_hint = up_hint.astype(np.float32)

    forward = target - origin
    f = forward / (np.linalg.norm(forward) + 1e-8)

    r = np.cross(f, up_hint)
    r_norm = np.linalg.norm(r)
    if r_norm < 1e-6:
        up_hint = np.array([0, 0, 1], dtype=np.float32)
        r = np.cross(f, up_hint)
        r_norm = np.linalg.norm(r)
        if r_norm < 1e-6:
            up_hint = np.array([1, 0, 0], dtype=np.float32)
            r = np.cross(f, up_hint)
            r_norm = np.linalg.norm(r)
    r = r / (r_norm + 1e-8)

    u = np.cross(r, f)
    r = r / (np.linalg.norm(r) + 1e-8)
    u = u / (np.linalg.norm(u) + 1e-8)
    f = f / (np.linalg.norm(f) + 1e-8)

    c2w = np.eye(4, dtype=np.float32)
    c2w[0:3, 0] = r
    c2w[0:3, 1] = u
    c2w[0:3, 2] = -f  # camera looks along -Z
    c2w[0:3, 3] = origin
    return c2w


def random_sphere_np(n: int, rng: np.random.Generator) -> np.ndarray:
    """Generate n random points on a unit sphere."""
    u = rng.random(n, dtype=np.float32)
    v = rng.random(n, dtype=np.float32)
    theta = 2.0 * math.pi * u
    z = 2.0 * v - 1.0
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.stack([x, y, z], axis=1).astype(np.float32)

def generate_extraction_poses(center: np.ndarray, radius: float, num_cameras: int, device: str):
    """
    Generates camera poses exactly matching the training data distribution:
    Random points on a sphere of 'radius' distance, looking at 'center'.
    """
    rng = np.random.default_rng(42) # Fixed seed for reproducible point clouds
    directions = random_sphere_np(num_cameras, rng)
    
    poses = []
    for d in directions:
        origin = center + radius * d
        # Uses your existing look_at_c2w_np function
        c2w = look_at_c2w_np(origin.astype(np.float32), center.astype(np.float32))
        poses.append(c2w[:3, :4])
        
    return torch.from_numpy(np.stack(poses, axis=0)).to(device)


def _extract_points_xyz_np(entry, j: int) -> np.ndarray:
    """Return (N,3) float32 numpy array of world points for object j."""
    pts = entry['data'][j] if isinstance(entry.get('data', None), (list, tuple)) else entry['data']
    if torch.is_tensor(pts):
        # remove singleton batch/pose dims if present
        while pts.ndim > 2 and pts.shape[0] == 1:
            pts = pts[0]
        pts = pts[..., :3].reshape(-1, 3).detach().cpu().numpy()
    else:
        pts = np.asarray(pts)[..., :3].reshape(-1, 3)
    return pts.astype(np.float32)



# NOTE: GT pose extraction removed (inference uses only sparse point clouds)


def compute_scene_center_radius(entry, j: int, fov_x: float, radius_scale: float, zoom_out: float = 1.15, center_mode: str = 'bbox'):
    """Compute a turntable center + camera radius using ONLY the sparse point cloud.

    This is the inference-safe version: it does NOT use GT cam_poses / images / depth.
    We mimic the dataset generator's distance heuristic:

        center = (mins + maxs)/2
        half_diag = 0.5 * ||maxs - mins||
        distance = half_diag / tan(fov_x/2) * radius_scale

    But you can choose a more downsampling-stable center via center_mode.
    Returns: center (3,), radius (float), bbox_mins (3,), bbox_maxs (3,)
    """
    pts = _extract_points_xyz_np(entry, j)
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)

    if center_mode == 'mean':
        center = pts.mean(axis=0)
        mins_r, maxs_r = mins, maxs
    elif center_mode == 'median':
        center = np.median(pts, axis=0)
        mins_r, maxs_r = mins, maxs
    elif center_mode == 'qbbox':
        qmin = np.quantile(pts, 0.01, axis=0)
        qmax = np.quantile(pts, 0.99, axis=0)
        center = 0.5 * (qmin + qmax)
        mins_r, maxs_r = qmin, qmax
    else:  # 'bbox'
        center = 0.5 * (mins + maxs)
        mins_r, maxs_r = mins, maxs

    half_diag = 0.5 * np.linalg.norm(maxs_r - mins_r)
    half_diag = float(max(half_diag, 1e-6))

    radius = (half_diag / np.tan(0.5 * float(fov_x))) * float(radius_scale) * float(zoom_out)
    radius = float(max(radius, 1e-4))

    return center.astype(np.float32), radius, mins.astype(np.float32), maxs.astype(np.float32)



def make_turntable_poses(center: np.ndarray, radius: float, elev_deg: float, n: int, device: str):
    """Generate (n,3,4) poses that orbit 'center' at given elevation."""
    center = np.asarray(center, dtype=np.float32)
    elev = np.deg2rad(elev_deg)
    angles = np.linspace(-180.0, 180.0, n, endpoint=False)

    poses = []
    for a in angles:
        th = np.deg2rad(a)
        d = np.array([math.cos(th) * math.cos(elev),
                      math.sin(elev),
                      math.sin(th) * math.cos(elev)], dtype=np.float32)
        origin = center + radius * d
        c2w = look_at_c2w_np(origin, center)[:3, :4]
        poses.append(c2w)

    return torch.from_numpy(np.stack(poses, axis=0)).to(device)


def export_model(render_kwargs, focal, path, path_colored, N=256):
    width = 1.1
    with torch.no_grad():
        #Sample NeRF
        t = torch.linspace(-width, width, N+1)
        query_pts = torch.stack(torch.meshgrid(t, t, t), -1)
        print(query_pts.shape)
        sh = query_pts.shape
        flat = query_pts.reshape([-1,3])
        print(flat.shape)

        fn = lambda i0, i1 : render_kwargs['network_query_fn'](flat[i0:i1,None,:], viewdirs=None, network_fn=render_kwargs['network_fn'])
        chunk = 1024*16
        raw = torch.cat([fn(i, i+chunk) for i in range(0, flat.shape[0], chunk)], 0)
        raw = torch.reshape(raw, list(sh[:-1]) + [-1])
        sigma = torch.clamp(raw[...,-1], min=0.0)

        #Marching cubes
        threshold = 5
        vertices, triangles = mcubes.marching_cubes(sigma.cpu().numpy(), threshold)
        print('done', vertices.shape, triangles.shape)

        #Two meshes because colors tend to be misplaced on mesh_export
        mesh = trimesh.Trimesh((vertices / N) - 0.5, triangles)

        obj = trimesh.exchange.ply.export_ply(mesh)

        with open(path, "wb+") as f:
            f.write(obj)

        print("Saved uncolored model to", path)

        rgbs = []
        final = []
        vertex_colors = []
        radius = 0.05 # distance from camera to a vertex, theoretically it could be lower to properly capture its color

        H = 1
        W = 1
        K = np.array([
            [focal, 0, 0.5*W],
            [0, focal, 0.5*H],
            [0, 0, 1]
        ])

        for i, vert in enumerate(mesh.vertices): 
            coords = np.array(vert)

            coords = coords / np.linalg.norm(coords)
            r, phi, theta = cart2sph(*coords)
            theta += math.pi/2
            phi -= math.pi
            c2w = pose_spherical(theta * 180 / math.pi, phi * 180 / math.pi, r+radius)
            result = render(H, W, K, chunk=2048, c2w=c2w, **render_kwargs)
            rgb = np.clip(result[0].detach().cpu().numpy(),0,1).squeeze()
            rgbs.append(rgb)
            final.append([*vert, *rgb])
            mesh.visual.vertex_colors[i] = np.concatenate((rgb, [1]))*255

        obj = trimesh.exchange.ply.export_ply(mesh)

        with open(path_colored, "wb+") as f:
            f.write(obj)

        print("Saved colored model to", path_colored)

def extract_pointcloud_from_depth(render_kwargs, poses, focal, H, W, device='cuda'):
    """
    Extracts a dense point cloud by rendering depth maps from virtual cameras 
    and back-projecting them to 3D space.
    """
    print(f"Starting Depth Back-Projection from {len(poses)} virtual cameras...")
    
    K = np.array([
        [focal, 0, 0.5 * W],
        [0, focal, 0.5 * H],
        [0, 0, 1]
    ], dtype=np.float32)
    
    all_points = []
    all_colors = []
    
    with torch.no_grad():
        for i, c2w_t in enumerate(poses):
            if i % 10 == 0:
                print(f"  Rendering view {i}/{len(poses)}...")
            
            c2w_t = c2w_t.to(device).float()
            
            # Render the depth (disparity) and color
            img, disp, acc, _ = render(H, W, K, chunk=2048, c2w=c2w_t, retraw=True, **render_kwargs)
            
            disp_np = disp.detach().cpu().numpy()
            img_np = img.detach().cpu().numpy()
            acc_np = acc.detach().cpu().numpy()
            
            # Convert disparity to depth (with epsilon to prevent div/0)
            depth_np = 1.0 / (disp_np + 1e-6)
            
            # MASK: Only keep pixels where the NeRF accumulated high opacity 
            # (meaning it actually hit a solid surface)
            valid_mask = acc_np > 0.8
            
            u, v = np.meshgrid(np.arange(W), np.arange(H))
            u_valid = u[valid_mask]
            v_valid = v[valid_mask]
            Z = depth_np[valid_mask]
            colors = img_np[valid_mask]
            
            # Back-project pixels to 3D Camera Space
            X = (u_valid - 0.5 * W) * Z / focal
            Y = -(v_valid - 0.5 * H) * Z / focal
            pts_cam = np.stack([X, Y, Z], axis=-1)
            
            # Transform from Camera Space to World Space
            c2w_np = c2w_t.cpu().numpy()
            R = c2w_np[:3, :3]
            t = c2w_np[:3, 3]
            pts_world = (R @ pts_cam.T).T + t
            
            all_points.append(pts_world)
            all_colors.append(colors)

    # Combine all views
    final_points = np.concatenate(all_points, axis=0)
    final_colors = np.concatenate(all_colors, axis=0)
    
    # Create Open3D Point Cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(final_points)
    pcd.colors = o3d.utility.Vector3dVector(np.clip(final_colors, 0, 1))
    
    # Downsample slightly to clean up overlapping points from different cameras
    print("Downsampling and cleaning point cloud...")
    # Change voxel_size to control final point density. 0.02 is usually good for a 1-unit bbox.
    pcd = pcd.voxel_down_sample(voxel_size=0.01) 
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    
    return pcd




if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)

    dirname = os.path.dirname(__file__)

    parser = argparse.ArgumentParser(description='Start training HyperRF')
    parser.add_argument('config_path', type=str,
                        help='Relative config path')
    parser.add_argument('-o_anim_count', type=int, help='How many object animations')
    parser.add_argument('-g_anim_count', type=int, default=0, help='How many generated object animations')
    parser.add_argument('-i_anim_count', type=int, default=0, help='How many interpolation object animations')
    parser.add_argument('-train_ds', type=int, help="Use train dataset?", default=0)
    parser.add_argument('-epoch', type=int, help="Default epoch to use. Set 0 to use latest.", default=0)
    # Rendering-only settings (for inference: we assume ONLY the sparse point cloud is available)
    parser.add_argument('-img_hw', type=int, nargs=2, default=[200, 200],
                        help='Render image height and width (default 200 200).')
    parser.add_argument('-fov', type=float, default=0.6911112070083618,
                        help='Horizontal field-of-view in radians (default matches dataset generator).')
    parser.add_argument('-radius_scale', type=float, default=0.85,
                        help='Camera distance multiplier used with bbox size to set turntable radius.')
    parser.add_argument('-zoom_out', type=float, default=1.15,
                        help='Extra zoom-out factor for turntable camera.')
    parser.add_argument('-center_mode', type=str, default='bbox',
                        choices=['bbox', 'mean', 'median', 'qbbox'],
                        help="How to compute scene center from sparse points: "
                             "'bbox'=(min+max)/2, 'mean', 'median', 'qbbox' uses 1%%-99%% quantile bbox.")
    parser.add_argument('-mc_threshold', type=float, default=5.0,
                        help='Marching cubes iso-threshold on sigma for mesh extraction (higher => thinner).'
    )


    args = parser.parse_args()

    config = None
    with open(args.config_path) as f:
        config = json.load(f)
    assert config is not None

    print(config)

    set_seed(config['seed'])

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    dataset = NeRFShapeNetDataset(root_dir=config['data_dir'], classes=config['classes'], train=args.train_ds != 0)

    config['batch_size'] = 1

    dataloader = DataLoader(dataset, batch_size=config['batch_size'],
                                    shuffle=config['shuffle'],
                                    num_workers=2, drop_last=True,
                                    pin_memory=True, generator=torch.Generator(device='cuda'))

    embed_fn, config['model']['TN']['input_ch_embed'] = get_embedder(config['model']['TN']['multires'], config['model']['TN']['i_embed'])

    embeddirs_fn = None
    config['model']['TN']['input_ch_views_embed'] = 0
    if config['model']['TN']['use_viewdirs']:
        embeddirs_fn, config['model']['TN']['input_ch_views_embed']= get_embedder(config['model']['TN']['multires_views'], config['model']['TN']['i_embed'])

    # Create a NeRF network
    nerf = NeRF(config['model']['TN']['D'],config['model']['TN']['W'], 
                config['model']['TN']['input_ch_embed'], 
                config['model']['TN']['input_ch_views_embed'],
                config['model']['TN']['use_viewdirs']).to(device)

    #Hypernetwork
    hnet = ChunkedHMLP(nerf.param_shapes, uncond_in_size=config['z_size'], cond_in_size=0,
                layers=config['model']['HN']['arch'], chunk_size=config['model']['HN']['chunk_size'], cond_chunk_embs=False, use_bias=config['model']['HN']['use_bias']).to(device)

    #Create encoder: either Resnet or classic
    if config['resnet']==True:
        encoder = resnet18(num_classes=config['z_size']).to(device) 
    else:
        encoder = Encoder(config).to(device) 

    results_dir = config['results_dir']
    os.makedirs(join(dirname,results_dir), exist_ok=True)

    with open(join(results_dir, "config_eval.json"), "w") as file:
        json.dump(config, file, indent=4)


    print(args.epoch, "set as starting epoch")
    if args.epoch == 0:
        print("Loading \'latest\' models")
        try:
            hnet.load_state_dict(torch.load(join(results_dir, f"model_hn_latest.pt"))) 
            print("Loaded HNet")
            encoder.load_state_dict(torch.load(join(results_dir, f"model_e_latest.pt")))
            print("Loaded Encoder")
        except:
            print("Haven't loaded all previous models.")
    else:
        starting_epoch = args.epoch
        print("Starting epoch:", starting_epoch)

    if(starting_epoch>0):
        print("Loading weights")
        try:
            hnet.load_state_dict(torch.load(join(results_dir, f"model_hn_{starting_epoch}.pt"))) 
            print("Loaded HNet")
            encoder.load_state_dict(torch.load(join(results_dir, f"model_e_{starting_epoch}.pt")))
            print("Loaded Encoder")
        except:
            print("Haven't found all previous models.")

    results_dir = join(dirname, 'rendered_samples', config['classes'][0])
    os.makedirs(results_dir, exist_ok=True)
    results_dir_main = results_dir

    encoder.eval()
    hnet.eval()

    default_N = 1024
    render_iterations = 60 + 1
    render_fps = 30

    # -----------------------------------------------------------------
    # Rendering intrinsics for inference (no GT images available).
    # We use a fixed resolution and fixed horizontal FOV.
    # -----------------------------------------------------------------
    H, W = int(args.img_hw[0]), int(args.img_hw[1])
    fov_x = float(args.fov)
    focal = 0.5 * W / np.tan(0.5 * fov_x)
    K = np.array([
        [focal, 0, 0.5 * W],
        [0, focal, 0.5 * H],
        [0, 0, 1]
    ], dtype=np.float32)

    # Defaults used for "generated" renders (no GT poses/points).
    default_center = None
    default_radius = None
    default_bbox_corners = None


    for i, (entry, cat, obj_path) in enumerate(dataloader):
        if i > args.o_anim_count:
            break

        start_time = datetime.now()

        if config['resnet']:
            nerf_Ws = get_nerf_resnet(entry, encoder, hnet)
        else:
            nerf_Ws, mu, logvar = get_nerf(entry, encoder, hnet)

        #For batch size == 1 hnet doesn't return batch dimension...
        if config['batch_size'] == 1:
            nerf_Ws = [nerf_Ws]
    
        for j, target_w in enumerate(nerf_Ws):
            render_kwargs = get_render_kwargs(config, nerf, target_w, embed_fn, embeddirs_fn)
            render_kwargs['perturb'] = False
            render_kwargs['raw_noise_std'] = 0.

            print("Animation", i, obj_path)
            # -----------------------------------------------------------------
            # WORLD-aligned camera setup (match generate_fixed.py)
            # -----------------------------------------------------------------
            scene_center, scene_radius, bbox_mins, bbox_maxs = compute_scene_center_radius(entry, j, fov_x=fov_x, radius_scale=args.radius_scale, zoom_out=args.zoom_out, center_mode=args.center_mode)
            bbox_corners = np.stack([bbox_mins, bbox_maxs], axis=0)

            if default_center is None:
                default_center = scene_center.copy()
                default_radius = float(scene_radius)
                default_bbox_corners = bbox_corners.copy()

            # Ensure renderer bounds match world scale (avoid clipping)
            render_kwargs['ndc'] = False
            render_kwargs['near'] = 0.0
            render_kwargs['far'] = max(float(render_kwargs.get('far', 0.0)), scene_radius * 3.0)


            sample_id = get_sample_id(obj_path, fallback=f"idx{i:04d}")
            results_dir = join(results_dir_main, f"o_{sample_id}")
            os.makedirs(results_dir, exist_ok=True)

            torch.set_printoptions(threshold=100)
            
            #Render cloud of points
            """
            for el in [0,45,90,135, 180, 225, 270, 315]:
                for az in [0,45,90,135, 180, 225, 270, 315]:
                    fig = plt.figure(figsize=(8,8))
                    ax = fig.add_subplot(111, projection = '3d')
                    ax.view_init(elev=el, azim=az)
                    ax.scatter(entry['data'][j][:,0], entry['data'][j][:,1], entry['data'][j][:,2], c = entry['data'][j][:,3:])
                    ax.set_xlim3d(-1, 1)
                    ax.set_ylim3d(-1, 1)
                    ax.set_zlim3d(-1, 1)
                    plt.axis('off')
                    plt.grid(b=None)
                    plt.tight_layout()
                    plt.savefig(join(results_dir, f'pc_{el}_{az}.png'))
                    plt.close()
            """
            
                        # NOTE: In inference mode we assume ONLY the sparse point cloud is available.
            # We therefore do NOT use GT images / depth / cam_poses from the .npz.

            with torch.no_grad():
                # (A) Object animation block — orbit around the object's WORLD bbox center
                render_poses = make_turntable_poses(scene_center, scene_radius, elev_deg=-45,
                                                    n=render_iterations-1, device=device)
                frames = []
                for k, pose in enumerate(render_poses):

                    img, disp, acc, _ = render(H, W, K, chunk=config['model']['TN']['netchunk'], c2w=pose,
                                                verbose=True, retraw=True,
                                                **render_kwargs)
                    frames.append(to8b(img.detach().cpu().numpy()))

                    if k % 4 == 0:
                        imageio.imsave(join(results_dir, f'o_{sample_id}_{k}.png'), to8b(img.detach().cpu().numpy()))

            writer = imageio.get_writer(join(results_dir, f'an_{sample_id}.gif'), fps=30)
            for frame in frames:
                writer.append_data(frame)
            writer.close()

            with torch.no_grad():
                # (B) “Other elevations” block — same WORLD center/radius, different elevations
                render_poses = torch.cat([
                    make_turntable_poses(scene_center, scene_radius, elev_deg=-45, n=8, device=device),
                    make_turntable_poses(scene_center, scene_radius, elev_deg=-30, n=8, device=device),
                    make_turntable_poses(scene_center, scene_radius, elev_deg=-15, n=8, device=device),
                ], dim=0)
                for k, pose in enumerate(render_poses):

                    img, disp, acc, _ = render(H, W, K, chunk=config['model']['TN']['netchunk'], c2w=pose,
                                                verbose=True, retraw=True,
                                                **render_kwargs)

                    imageio.imsave(join(results_dir, f'o_other_{i}_{k}.png'), to8b(img.detach().cpu().numpy()))

            render_kwargs['near'] = 0.
            
            # --- DEPTH BACK-PROJECTION EXTRACTION ---
            # 1. Define how many views to use to build the cloud
            # 50 matches your training data, but 100-200 will give a denser cloud
            num_extraction_cameras = 200 
            
            # 2. Generate poses mimicking your training distribution
            extraction_poses = generate_extraction_poses(scene_center, scene_radius*1.25, num_extraction_cameras, device)
            
            # 3. Extract the point cloud!
            pcd = extract_pointcloud_from_depth(render_kwargs, extraction_poses, focal, H, W, device)
            
            # 4. Save the point cloud
            pc_path = join(results_dir, f'depth_extracted_pc_{sample_id}.ply')
            o3d.io.write_point_cloud(pc_path, pcd)
            print(f"Saved Depth Back-Projected point cloud to {pc_path}")

    for i in range(args.g_anim_count):
        start_time = datetime.now()
        sample = torch.normal(mean=torch.zeros(config["z_size"]), std=torch.full((config["z_size"],), fill_value=0.006))
        render_kwargs = get_render_kwargs(config, nerf, get_nerf_from_code(hnet, sample[None]), embed_fn, embeddirs_fn)
        render_kwargs['perturb'] = False
        render_kwargs['raw_noise_std'] = 0.

        # (C) Generated object animation — use the default WORLD camera (from first GT object)
        if default_center is None:
            gen_center = np.zeros(3, dtype=np.float32)
            gen_radius = 3.2
            gen_bbox_corners = np.stack([gen_center - 1.1, gen_center + 1.1], axis=0)
        else:
            gen_center = default_center
            gen_radius = float(default_radius)
            gen_bbox_corners = default_bbox_corners

        render_kwargs['ndc'] = False
        render_kwargs['near'] = 0.0
        render_kwargs['far'] = max(float(render_kwargs.get('far', 0.0)), gen_radius * 3.0)

        
        results_dir = join(results_dir_main, f'g{i}')
        os.makedirs(results_dir, exist_ok=True)

        print("Generated Object Animation", i)
        with torch.no_grad():
            render_poses = make_turntable_poses(gen_center, gen_radius, elev_deg=-45, n=render_iterations-1, device=device)
            frames = []
            for k, pose in enumerate(render_poses):

                img, disp, acc, _ = render(H, W, K, chunk=config['model']['TN']['netchunk'], c2w=pose,
                                                    verbose=True, retraw=True,
                                                    **render_kwargs)
                frames.append(to8b(img.detach().cpu().numpy()))

                if k%4==0:
                    imageio.imsave(join(results_dir, f'g_{i}_{k}.png'), to8b(img.detach().cpu().numpy()))

            writer = imageio.get_writer(join(results_dir, f'g_an_{i}.gif'), fps=render_fps)
            for frame in frames:
                writer.append_data(frame)
            writer.close()

            render_kwargs['near'] = 0. 

            # --- Depth Extraction for Generated Object ---
            gen_extraction_poses = generate_extraction_poses(gen_center, gen_radius, 50, device)
            pcd = extract_pointcloud_from_depth(render_kwargs, gen_extraction_poses, focal, H, W, device)
            
            pc_path = join(results_dir, f'g_depth_extracted_pc_{i}.ply')
            o3d.io.write_point_cloud(pc_path, pcd)
        print("Time:", round((datetime.now() - start_time).total_seconds(), 2))

    
    dl_iter = iter(dataloader)

    for i in range(args.i_anim_count):
        with torch.no_grad():

            results_dir = join(results_dir_main, f'i{i}')
            os.makedirs(results_dir, exist_ok=True)

            full_interpolations = None
            start_time = datetime.now()

            entry_1, cat_1, obj_path_1 = next(dl_iter)
            entry_2, cat_2, obj_path_2  = next(dl_iter)

            nerf_1_code = get_code(entry_1, encoder)
            nerf_2_code = get_code(entry_2, encoder)
            print("Generated Object Animation", i)
            print(obj_path_1)
            print(obj_path_2)
            
            kwargs_1 = get_render_kwargs(config, nerf, get_nerf_from_code(hnet, nerf_1_code), embed_fn, embeddirs_fn)
            kwargs_2 = get_render_kwargs(config, nerf, get_nerf_from_code(hnet, nerf_2_code), embed_fn, embeddirs_fn)

            kwargs_1['perturb'] = False
            kwargs_1['raw_noise_std'] = 0.

            kwargs_2['perturb'] = False
            kwargs_2['raw_noise_std'] = 0.

            # (D) Interpolation animation — build a WORLD-aligned camera that frames BOTH objects
            c1, r1, bmin1, bmax1 = compute_scene_center_radius(entry_1, 0, fov_x=fov_x, radius_scale=args.radius_scale, zoom_out=args.zoom_out, center_mode=args.center_mode)
            c2, r2, bmin2, bmax2 = compute_scene_center_radius(entry_2, 0, fov_x=fov_x, radius_scale=args.radius_scale, zoom_out=args.zoom_out, center_mode=args.center_mode)
            union_mins = np.minimum(bmin1, bmin2)
            union_maxs = np.maximum(bmax1, bmax2)
            interp_center = 0.5 * (union_mins + union_maxs)

            half1 = 0.5 * float(np.linalg.norm(bmax1 - bmin1))
            ratio = float(r1 / (half1 + 1e-6))
            half_u = 0.5 * float(np.linalg.norm(union_maxs - union_mins))
            interp_radius = float(ratio * half_u)
            interp_bbox_corners = np.stack([union_mins, union_maxs], axis=0)

            for _kw in (kwargs_1, kwargs_2):
                _kw['ndc'] = False
                _kw['near'] = 0.0
                _kw['far'] = max(float(_kw.get('far', 0.0)), interp_radius * 3.0)


            steps = render_iterations + 1

            # --- Depth Extraction for Interpolation Baselines ---
            i_poses = generate_extraction_poses(interp_center, interp_radius, 50, device)
            
            pcd_1 = extract_pointcloud_from_depth(kwargs_1, i_poses, focal, H, W, device)
            o3d.io.write_point_cloud(join(results_dir, f'i_1_depth_pc_{i}.ply'), pcd_1)
            
            pcd_2 = extract_pointcloud_from_depth(kwargs_2, i_poses, focal, H, W, device)
            o3d.io.write_point_cloud(join(results_dir, f'i_2_depth_pc_{i}.ply'), pcd_2)
        
            writer = imageio.get_writer(join(results_dir, f'i_an_{i}.gif'), fps=render_fps)
            render_poses = make_turntable_poses(interp_center, interp_radius, elev_deg=-45, n=steps-1, device=device)
            for k, pose in enumerate(render_poses):
                
                #c2w=pose for rotation
                img1, disp, acc, _ = render(H, W, K, chunk=config['model']['TN']['netchunk'], c2w=render_poses[-36],
                                        verbose=True, retraw=True,**kwargs_1)
                img2, disp, acc, _ = render(H, W, K, chunk=config['model']['TN']['netchunk'], c2w=render_poses[-36],
                                        verbose=True, retraw=True,**kwargs_2)

                nerf_3_code=torch.lerp(nerf_1_code, nerf_2_code, k/steps)
                
                kwargs_3 = get_render_kwargs(config, nerf, get_nerf_from_code(hnet, nerf_3_code), embed_fn, embeddirs_fn)
                kwargs_3['perturb'] = False
                kwargs_3['raw_noise_std'] = 0.
                kwargs_3['ndc'] = False
                kwargs_3['near'] = 0.0
                kwargs_3['far'] = max(float(kwargs_3.get('far', 0.0)), interp_radius * 3.0)

                

                img3, disp, acc, _ = render(H, W, K, chunk=config['model']['TN']['netchunk'], c2w=render_poses[-36],
                                        verbose=True, retraw=True,**kwargs_3)
                
                frame = torch.cat([img1,img3,img2], dim=1)
                
                if k % 5==0:
                    kwargs_3['near'] = 0.
                    
                    # --- Depth Extraction for Interpolated Frame ---
                    i_frame_poses = generate_extraction_poses(interp_center, interp_radius, 50, device)
                    pcd_interp = extract_pointcloud_from_depth(kwargs_3, i_frame_poses, focal, H, W, device)
                    o3d.io.write_point_cloud(join(results_dir, f'interpolated_depth_pc_{i}_{k}.ply'), pcd_interp)
                    
                    imageio.imsave(join(results_dir, f'ii_{i}_{k}.png'), to8b(img3.detach().cpu().numpy()))
                
                writer.append_data(to8b(frame.detach().cpu().numpy()))
            writer.close()


            print("Time:", round((datetime.now() - start_time).total_seconds(), 2))
