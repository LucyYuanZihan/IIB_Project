import numpy as np
import os
from os.path import join, exists
import matplotlib.pyplot as plt
import pandas as pd
import torch
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

from ChamferDistancePytorch.fscore import fscore

import open3d as o3d

import argparse

import ChamferDistancePytorch.chamfer_python as chfp
import os, re

#Needed for workers for dataloader
from torch.multiprocessing import Pool, Process, set_start_method
set_start_method('spawn', force=True)

def rot_x(angle):
    rx = torch.Tensor([ [1,0,0],
                        [0, math.cos(angle), -math.sin(angle)],
                        [0, math.sin(angle), math.cos(angle)]])
    return rx

def sanitize_id(s: str) -> str:
    s = str(s)
    s = os.path.splitext(os.path.basename(s))[0]  # drop path + extension if any
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("_")
    return s


def as_mesh(scene_or_mesh):
    if isinstance(scene_or_mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate([
            trimesh.Trimesh(vertices=m.vertices, faces=m.faces)
            for m in scene_or_mesh.geometry.values()])
    else:
        mesh = scene_or_mesh
    return mesh

def calculate_best_mesh_metrics(obj_path, render_kwargs, save_pc=True, name="1", thresholds=[1,2,3]):

    fscores = [calculate_mesh_metrics(obj_path, render_kwargs, save_pc, name+f'_{t}', t) for t in thresholds]
    return max(fscores, key=lambda x: x[0].item())

def calculate_mesh_metrics(obj_path, render_kwargs, save_pc=True, name="1", threshold = 3):

    with torch.no_grad():
        N = 128
        t = torch.linspace(-1.1, 1.1, N+1)

        query_pts = torch.stack(torch.meshgrid(t, t, t), -1)
        sh = query_pts.shape
        flat = query_pts.reshape([-1,3])

        def batchify(fn, chunk):
            if chunk is None:
                return fn
            def ret(inputs):
                return torch.cat([fn(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], 0)
            return ret

        fn = lambda i0, i1 : render_kwargs['network_query_fn'](flat[i0:i1,None,:], viewdirs=None, network_fn=render_kwargs['network_fn'])
        chunk = 1024*16
        raw = torch.cat([fn(i, i+chunk) for i in range(0, flat.shape[0], chunk)], 0)
        raw = torch.reshape(raw, list(sh[:-1]) + [-1])
        sigma = torch.maximum(raw[...,-1], torch.Tensor([0.]))

        vertices, triangles = mcubes.marching_cubes(sigma.cpu().numpy(), threshold)
        mesh = trimesh.Trimesh(vertices / N - .5, triangles)
        
        try:
            entry_mesh = trimesh.load_mesh(obj_path, force='mesh')
            entry_mesh = as_mesh(entry_mesh)

            entry_points = trimesh.sample.sample_surface(entry_mesh, 3000)
            entry_points = torch.from_numpy(entry_points[0]).to(device, dtype=torch.float)

            entry_points = rot_x(math.pi/2)@entry_points.T
            entry_points = entry_points.T
            entry_points = entry_points[None]

            sampled_points = trimesh.sample.sample_surface(mesh, 3000)
            sampled_points = torch.from_numpy(sampled_points[0])[None].to(device, dtype=torch.float)
            
            dist1, dist2, idx1, idx2 = chfp.distChamfer(entry_points, sampled_points)
            cd = (torch.mean(dist1)) + (torch.mean(dist2))
            f_score, precision, recall = fscore(dist1, dist2, 0.01)
        except Exception as e:
            print(e)
            f_score = torch.Tensor([0.0])
            cd = torch.Tensor([0.0])
        
        iou=0
        if save_pc:
            try:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(sampled_points.detach().cpu().numpy()[0])
                o3d.io.write_point_cloud(f"./results/pcs/{name}_sampled_points.ply", pcd)

                pcd = o3d.geometry.PointCloud()
                #print(entry_points.detach().cpu().numpy())
                #print(entry_points.detach().cpu().numpy().shape)
                pcd.points = o3d.utility.Vector3dVector(entry_points.detach().cpu().numpy()[0])
                o3d.io.write_point_cloud(f"./results/pcs/{name}_entry_points.ply", pcd)
            except Exception as e:
                print(e)
                print("something went wrong with saving point cloud!")
                raise e
        return f_score, cd #remember the threshold!

def calculate_mesh_metrics_worldaligned(points_xyz, render_kwargs, threshold=3.0, N=128, pad_frac=0.10):
    """
    points_xyz: torch tensor (Npts,3) in WORLD coordinates (same coords as cam_poses).
    """
    with torch.no_grad():
        # --- 1) Build world-space bounds from points ---
        pts = points_xyz.detach().cpu().numpy()
        mins = pts.min(axis=0)
        maxs = pts.max(axis=0)

        # pad the box a bit so the surface isn't clipped
        extent = maxs - mins
        mins = mins - pad_frac * extent
        maxs = maxs + pad_frac * extent

        # --- 2) Query sigma on a world-space grid ---
        xs = torch.linspace(mins[0], maxs[0], N, device=device)
        ys = torch.linspace(mins[1], maxs[1], N, device=device)
        zs = torch.linspace(mins[2], maxs[2], N, device=device)

        grid = torch.stack(torch.meshgrid(xs, ys, zs, indexing='ij'), dim=-1)  # (N,N,N,3)
        flat = grid.reshape(-1, 3)

        chunk = 1024 * 16
        fn = lambda i0, i1: render_kwargs['network_query_fn'](
            flat[i0:i1, None, :], viewdirs=None, network_fn=render_kwargs['network_fn']
        )

        raw = torch.cat([fn(i, i + chunk) for i in range(0, flat.shape[0], chunk)], 0)
        raw = raw.reshape(N, N, N, -1)
        sigma = torch.clamp(raw[..., -1], min=0.0)

        # --- 3) Marching cubes gives vertices in voxel index coordinates ---
        vertices, triangles = mcubes.marching_cubes(sigma.detach().cpu().numpy(), threshold)

        # Map voxel vertices -> world coordinates
        scale = (maxs - mins) / (N - 1)   # world units per voxel step
        verts_world = vertices * scale[None, :] + mins[None, :]

        mesh = trimesh.Trimesh(verts_world, triangles)
        return mesh



def calculate_image_metrics(entry, render_kwargs, metric_fn, j, count=5):
    x = []
    y = []
    with torch.no_grad():
        for c in range(count):
            img_i = np.random.choice(len(entry['images'][j]), 1)
            target = entry['images'][j][img_i][0].to(device)#entry['images'][j][img_i][0].to(device)
            target = torch.Tensor(target.float())
            pose = entry['cam_poses'][j][img_i, :3,:4][0].to(device)
            
            H = entry["images"][j].shape[1]
            W = entry["images"][j].shape[2]
            focal = .5 * W / np.tan(.5 * 0.6911112070083618) 

            K = np.array([
            [focal, 0, 0.5*W],
            [0, focal, 0.5*H],
            [0, 0, 1]
            ])

            img, _, _, _ = render(H, W, K, chunk=config['model']['TN']['netchunk'], c2w=pose,
                                                        verbose=True, retraw=True,
                                                        **render_kwargs)
            
            x.append(img)
            y.append(target)
        
        x = torch.stack(x)
        y = torch.stack(y)

        metric_val = metric_fn(y, x)

        return metric_val

def project_points_fullscene(
    pts: np.ndarray,
    intens: np.ndarray,
    c2w: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    img_w: int,
    img_h: int,
    depth_clip: [float, float] = (0.0, 1000.0),
):
    """
    Project the entire point cloud into the image plane with a Z-buffer.

    Returns:
      rgb:   (H,W) uint8 grayscale [0,255], bg = 0
      depth: (H,W) uint16 depth in mm, bg = 0
      hit:   (H,W) bool mask where we saw at least one point
    """
    # world->camera
    w2c = np.linalg.inv(c2w)
    Pw = np.concatenate([pts, np.ones((pts.shape[0], 1), dtype=pts.dtype)], axis=1)
    Pc = (w2c @ Pw.T).T[:, :3]
    z = -Pc[:, 2]
    valid = z > 1e-4
    if not np.any(valid):
        rgb = np.zeros((img_h, img_w), dtype=np.uint8)
        depth = np.zeros((img_h, img_w), dtype=np.uint16)
        hit = np.zeros((img_h, img_w), dtype=bool)
        return rgb, depth, hit

    Pc = Pc[valid]
    z = z[valid]
    I = intens[valid]

    u = fx * (Pc[:, 0] / z) + cx
    v = -fy * (Pc[:, 1] / z) + cy

    ui = np.round(u).astype(np.int32)
    vi = np.round(v).astype(np.int32)
    in_img = (ui >= 0) & (ui < img_w) & (vi >= 0) & (vi < img_h)
    if not np.any(in_img):
        rgb = np.zeros((img_h, img_w), dtype=np.uint8)
        depth = np.zeros((img_h, img_w), dtype=np.uint16)
        hit = np.zeros((img_h, img_w), dtype=bool)
        return rgb, depth, hit

    ui = ui[in_img]
    vi = vi[in_img]
    z = z[in_img]
    I = I[in_img]

    # Normalize intensity per view to [0,255]
    I_clamped = np.clip(I, 0.0, 1.0)
    Iu8 = np.round(I_clamped * 255.0).astype(np.uint8)

    H, W = img_h, img_w
    rgb = np.zeros((H, W), dtype=np.uint8)
    depth = np.zeros((H, W), dtype=np.uint16)
    hit = np.zeros((H, W), dtype=bool)

    zmin, zmax = depth_clip
    for px, py, pz, val in zip(ui, vi, z, Iu8):
        if not (zmin <= pz <= zmax):
            continue
        # Z-buffer: keep nearest
        if (not hit[py, px]) or pz < (depth[py, px] / 1000.0):
            hit[py, px] = True
            rgb[py, px] = val
            depth[py, px] = int(np.clip(round(pz * 1000.0), 0, 65535))  # mm

    return rgb, depth, hit
if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)

    dirname = os.path.dirname(__file__)

    parser = argparse.ArgumentParser(description='Start training HyperRF')
    parser.add_argument('config_path', type=str,
                        help='Relative config path')

    args = parser.parse_args()

    config = None
    with open(args.config_path) as f:
        config = json.load(f)
    assert config is not None

    print(config)

    set_seed(config['seed'])

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    # ---- NEW: read optional geometry flag/path (defaults keep old behavior) ----
    compute_geometry = config.get("compute_geometry", True)
    shapenet_dir = config.get("shapenet_dir", None)
    if not compute_geometry:
        print("Skipping geometry metrics (compute_geometry=false). Running image metrics only.")

    #config['classes'] = ['cars']

    dataset = NeRFShapeNetDataset(root_dir=config['data_dir'], classes=config['classes'], train=False)

    config['batch_size'] = 1

    dataloader = DataLoader(dataset, batch_size=config['batch_size'],
                                    shuffle=False,
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

    try:
        losses_r = np.load(join(results_dir, f'losses_r.npy')).tolist()
        print("Loaded reconstruction losses")
        losses_kld = np.load(join(results_dir, f'losses_kld.npy')).tolist()
        print("Loaded KLD losses")
        losses_depth = np.load(join(results_dir, f'losses_depth.npy')).tolist()
        print("Loaded depth losses")
        losses_total = np.load(join(results_dir, f'losses_total.npy')).tolist()
        print("Loaded total losses")
    except:
        print("Haven't found previous losses. Is this a new experiment?")
        losses_r = []
        losses_kld = []
        loss_depth = []
        losses_total = []

    if losses_total == []:
        print("Loading 'latest' model without loaded losses")
        try:
            hnet.load_state_dict(torch.load(join(results_dir, f"model_hn_latest.pt"))) 
            print("Loaded HNet")
            encoder.load_state_dict(torch.load(join(results_dir, f"model_e_latest.pt")))
            print("Loaded Encoder")
            #scheduler.load_state_dict(torch.load(join(results_dir, f"lr_latest.pt")))
            #print("Loaded Scheduler")
        except:
            print("Haven't loaded all previous models.")
    else:
        starting_epoch = len(losses_total)

        print("starting epoch:", starting_epoch)

        if(starting_epoch>0):
            print("Loading weights since previous losses were found")
            try:
                hnet.load_state_dict(torch.load(join(results_dir, f"model_hn_{starting_epoch-1}.pt"))) 
                print("Loaded HNet")
                encoder.load_state_dict(torch.load(join(results_dir, f"model_e_{starting_epoch-1}.pt")))
                print("Loaded Encoder")
                #scheduler.load_state_dict(torch.load(join(results_dir, f"lr_{starting_epoch-1}.pt")))
                #print("Loaded Scheduler")
            except:
                print("Haven't loaded all previous models.")

    results_dir = join(results_dir, 'eval_download_weights')
    os.makedirs(results_dir, exist_ok=True)

    # --- ADDED: folder to save image pairs ---
    images_dir = join(results_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    encoder.eval()
    hnet.eval()

    mse = torch.nn.MSELoss()
    psnr_metric = lambda x,y: torch.mean(mse2psnr(mse(y,x)))

    eval_results = pd.DataFrame(columns=['class', 'fscore', 'cd', 'psnr'])

    for i, (entry, cat, obj_path) in enumerate(dataloader):
        start_time = datetime.now()

        if config['resnet']:
            nerf_Ws, mu, logvar = get_nerf_resnet(entry, encoder, hnet)
        else:
            nerf_Ws, mu, logvar = get_nerf(entry, encoder, hnet)

        #For batch size == 1 hnet doesn't return batch dimension...
        if config['batch_size'] == 1:
            nerf_Ws = [nerf_Ws]
        

        for j, target_w in enumerate(nerf_Ws):
            print(f"[{i+1}/{len(dataloader)}] evaluating {cat[j]}...", flush=True)
            render_kwargs = get_render_kwargs(config, nerf, target_w, embed_fn, embeddirs_fn)
            render_kwargs['ndc']  = False
            render_kwargs['near'] = config.get('near', 0.1)
            render_kwargs['far']  = config.get('far', 15.0)
            render_kwargs['perturb'] = False
            render_kwargs['raw_noise_std'] = 0.
            
            points = entry["data"][j]
            points = points.to(device, dtype=torch.float)

            # ---- NEW: guard geometry metrics if disabled or if mesh missing ----
            if compute_geometry:
                try:
                    f_score, cd = calculate_best_mesh_metrics(obj_path[j], render_kwargs, save_pc=False, name=str(i), thresholds=[3])
                except Exception as e:
                    print(e)
                    f_score, cd = torch.Tensor([float('nan')]), torch.Tensor([float('nan')])
            else:
                f_score, cd = torch.Tensor([float('nan')]), torch.Tensor([float('nan')])

            psnr = calculate_image_metrics(entry, render_kwargs, psnr_metric, j, count=5)

            # --- ADDED: save ONE rendered + ground-truth pair per object ---
            with torch.no_grad():
                #img_i = np.random.choice(len(entry['images'][j]), 1)
                img_i = [0]

                target = entry['images'][j][img_i][0].to(device)
                target = torch.as_tensor(target, dtype=torch.float32)
                pose = entry['cam_poses'][j][img_i, :3, :4][0].to(device)

                H = entry["images"][j].shape[1]
                W = entry["images"][j].shape[2]
                focal = .5 * W / np.tan(.5 * 0.6911112070083618)
                K = np.array([[focal, 0, 0.5*W],
                            [0,     focal, 0.5*H],
                            [0,     0,     1]])

                img, _, _, _ = render(H, W, K,
                                    chunk=config['model']['TN']['netchunk'],
                                    c2w=pose, verbose=False, retraw=True,
                                    **render_kwargs)

                # to numpy
                pred = img.detach().cpu().numpy()
                gt   = target.detach().cpu().numpy()

                # normalize/clamp for matplotlib
                # pred should already be ~[0,1], but clamp just in case
                pred = np.clip(pred, 0.0, 1.0)

                # gt may be [0,255] or [0,1]; normalize if needed, then clamp
                if gt.max() > 1.0:
                    gt = gt / 255.0
                gt = np.clip(gt, 0.0, 1.0)

                # base = f"{i:04d}_{j:02d}"
                # obj_path is what your dataset returns as the 3rd item (often a name or filename)
                sample_name = sanitize_id(obj_path[j])
                view_id = int(img_i[0])  # here img_i=[0], but keep it general
                base = f"{sample_name}_{view_id:02d}"

                plt.imsave(join(images_dir, base + "_pred.png"), pred)
                plt.imsave(join(images_dir, base + "_gt.png"),   gt)

                # ---- DEBUG: reproject point cloud to check pose/K consistency ----
                pts = entry["data"][j][:, :3].detach().cpu().numpy()
                intens = entry["data"][j][:, 3].detach().cpu().numpy()  # since you stored grayscale replicated in RGB
                pose4 = torch.eye(4, device=pose.device, dtype=pose.dtype)
                pose4[:3, :4] = pose
                c2w_np = pose4.detach().cpu().numpy()

                rgb_u8, depth_u16, hit = project_points_fullscene(
                    pts=pts,
                    intens=intens,
                    c2w=c2w_np,
                    fx=focal, fy=focal, cx=0.5*W, cy=0.5*H,
                    img_w=W, img_h=H,
                    depth_clip=(0.0, 1e9),
                )

                # make it match training GT style: white background
                proj = (rgb_u8.astype(np.float32) / 255.0)
                proj = np.stack([proj, proj, proj], axis=-1)
                proj[~hit] = 1.0
                plt.imsave(join(images_dir, base + "_reproj.png"), proj)

            eval_results = eval_results.append({'class': cat[j], 'fscore': f_score.item(),'cd': cd.item(), 'psnr': psnr.item()}, ignore_index=True)

    print(eval_results.groupby("class").describe())
    print("---------------")
    print(eval_results[['fscore', 'cd', 'psnr']].describe())

    # --- ADDED: save metrics to CSVs ---
    eval_results.to_csv(join(results_dir, "metrics.csv"), index=False)
    grouped = eval_results.groupby("class").describe()
    grouped.to_csv(join(results_dir, "metrics_summary_by_class.csv"))
    overall = eval_results[['fscore', 'cd', 'psnr']].describe()
    overall.to_csv(join(results_dir, "metrics_overall.csv"))
    print(f"Saved metrics to: {join(results_dir, 'metrics.csv')}")
    print(f"Saved images to: {images_dir}")
