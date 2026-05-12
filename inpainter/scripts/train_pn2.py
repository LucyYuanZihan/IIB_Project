
from datetime import datetime
import os
from os.path import join, exists
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from itertools import chain
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import mlflow
from mlflow.tracking import MlflowClient

from src.datasets.seg2tunnel import NeRFShapeNetDataset

from src.models.encoder import Encoder, PointNet2VAEEncoder 
from src.models.nerf import NeRF
from src.models.resnet import resnet18
from hypnettorch.hnets.chunked_mlp_hnet import ChunkedHMLP
from src.render.nerf_helpers import *

from src.utils import *

#Needed for workers for dataloader
from torch.multiprocessing import Pool, Process, set_start_method
set_start_method('spawn', force=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ================= Block 1: Argument Parsing =================
# Description: Parse command-line arguments to get configuration file
# ============================================================
if __name__ == '__main__':
    dirname = os.path.dirname(__file__)

    parser = argparse.ArgumentParser(description='Start training')
    parser.add_argument('config_path', type=str,
                        help='Relative config path')

    args = parser.parse_args()
    # Load configuration from JSON file
    config = None
    with open(args.config_path) as f:
        config = json.load(f)
    assert config is not None

    print(config)

    set_seed(config['seed'])  # ensures reproducibility

    
    print('Device: ', device)
    # Makes all new tensors default to GPU float tensors (be aware of this when debugging).
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

# ---------- Block 2: Dataset and Dataloader ----------
    dataset = NeRFShapeNetDataset(root_dir=config['data_dir'], classes=config['classes'])
    # Create dataloader: arguments control how batches are created and loaded from dataset
    dataloader = DataLoader(dataset, batch_size=config['batch_size'],
                                    shuffle=config['shuffle'],
                                    num_workers=8, drop_last=True,
                                    pin_memory=True, generator=torch.Generator(device='cuda'))

# ---------- Block 3: Positional encoders (for NeRF) ----------
    # embed_fn: the positional encoding for 3D positions.
    embed_fn, config['model']['TN']['input_ch_embed'] = get_embedder(config['model']['TN']['multires'], config['model']['TN']['i_embed'])
    # embeddirs_fn: the positional encoding for viewing directions (if enabled).
    embeddirs_fn = None
    config['model']['TN']['input_ch_views_embed'] = 0
    if config['model']['TN']['use_viewdirs']:
        embeddirs_fn, config['model']['TN']['input_ch_views_embed']= get_embedder(config['model']['TN']['multires_views'], config['model']['TN']['i_embed'])

#
    # Create a NeRF network
    nerf = NeRF(config['model']['TN']['D'],config['model']['TN']['W'], 
                config['model']['TN']['input_ch_embed'], 
                config['model']['TN']['input_ch_views_embed'],
                config['model']['TN']['use_viewdirs']).to(device)

    #Hypernetwork
    hnet = ChunkedHMLP(nerf.param_shapes, uncond_in_size=config['z_size'], cond_in_size=0,
                layers=config['model']['HN']['arch'], chunk_size=config['model']['HN']['chunk_size'], cond_chunk_embs=False, use_bias=config['model']['HN']['use_bias']).to(device)

    print(hnet.param_shapes)
    
    # Create encoder: ResNet or PointNet-style / PointNet++-style
    encoder_type = config.get("encoder_type", "pointnet")  # "pointnet" | "pointnet2"

    if config.get("resnet", False):
        encoder = resnet18(num_classes=config["z_size"]).to(device)
    else:
        if encoder_type.lower() in ["pointnet2", "pointnet++", "pn2"]:
            encoder = PointNet2VAEEncoder(config).to(device)
        else:
            encoder = Encoder(config).to(device)


    #RAdam because it might help with not collapsing to white background
    optimizer = torch.optim.RAdam(chain(encoder.parameters(), hnet.internal_params), **config['optimizer']['E_HN']['hyperparams'])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config['lr_decay'])
    loss_fn = torch.nn.MSELoss()

    depth_loss_fn = torch.nn.MSELoss()
    lambda_depth = config.get('lambda_depth', 0.1)   # add this to config.json

    # MLflow setup
    experiment_name = f'Experiments_seg2tunnel'
    mlflow.set_tracking_uri('/home/zy349/points2nerf/mlflow')
    client = MlflowClient()
    try:
        EXP_ID = client.create_experiment(experiment_name)
    except:
        experiments = client.get_experiment_by_name(experiment_name)
        EXP_ID = experiments.experiment_id

    start_time = datetime.now().strftime("%Y%m%d-%H%M%S")

    with mlflow.start_run(experiment_id=EXP_ID,
                          run_name=f'seg2tunnel_{start_time}'):
        mlflow.log_params(config)
        # model_param_num = sum(p.numel() for p in model.parameters())
        # model_size = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6
        # mlflow.log_metric('model_param_num', model_param_num)
        # mlflow.log_metric('model_size_mb', model_size)

        results_dir = config['results_dir']
        os.makedirs(join(dirname,results_dir), exist_ok=True)

        with open(join(results_dir, "config.json"), "w") as file:
            json.dump(config, file, indent=4)

        try:
            losses_r = np.load(join(results_dir, f'losses_r.npy')).tolist()
            print("Loaded reconstruction losses")
            losses_kld = np.load(join(results_dir, f'losses_kld.npy')).tolist()
            print("Loaded KLD losses")
            losses_total = np.load(join(results_dir, f'losses_total.npy')).tolist()
            print("Loaded total losses")

            # NEW: load depth losses if present, otherwise backfill with zeros
            depth_path = join(results_dir, 'losses_depth.npy')
            if os.path.exists(depth_path):
                losses_depth = np.load(depth_path).tolist()
                print("Loaded depth losses")
            else:
                losses_depth = [0.0] * len(losses_total)
                print("No depth loss history found; initializing losses_depth to zeros")
        except:
            print("Haven't found previous loss data. We are assuming that this is a new experiment.")
            losses_r = []
            losses_kld = []
            losses_total = []
            losses_depth = []

        starting_epoch = len(losses_total)

        print("starting epoch:", starting_epoch)

        if(starting_epoch>0):
            print("Loading weights since previous losses were found")
            try:
                hnet.load_state_dict(torch.load(join(results_dir, f"model_hn_{starting_epoch-1}.pt"))) 
                print("Loaded HNet")
                encoder.load_state_dict(torch.load(join(results_dir, f"model_e_{starting_epoch-1}.pt")))
                print("Loaded Encoder")
                scheduler.load_state_dict(torch.load(join(results_dir, f"lr_{starting_epoch-1}.pt")))
                print("Loaded Scheduler")
            except:
                print("Haven't found all previous models.")


        hnet.train()
        encoder.train()

        os.makedirs(join(results_dir, 'samples'), exist_ok=True)

        loss_log_path = join(results_dir, "loss_log.csv")

        # If new experiment, create file with header
        if not os.path.exists(loss_log_path):
            with open(loss_log_path, "w") as f:
                f.write("epoch,total_loss,loss_r,loss_kld,loss_depth,lambda_depth\n")

        global_batch_idx = 0


        for epoch in range(starting_epoch, starting_epoch+config['max_epochs'] + 1):
            start_epoch_time = datetime.now()
            
            total_loss = 0.0
            total_loss_r = 0.0
            total_loss_kld = 0.0
            total_loss_depth = 0.0

            
            for i, (entry, cat, obj_path) in enumerate(dataloader):
                x = []
                y = []
                depth_x = []
                depth_y = []
                
                if config['resnet']:
                    nerf_Ws, mu, logvar = get_nerf_resnet(entry, encoder, hnet)
                else:
                    nerf_Ws, mu, logvar = get_nerf(entry, encoder, hnet)

                #For batch size == 1 hnet doesn't return batch dimension...
                if config['batch_size'] == 1:
                    nerf_Ws = [nerf_Ws]

                for j, target_w in enumerate(nerf_Ws):
                    render_kwargs_train = get_render_kwargs(config, nerf, target_w, embed_fn, embeddirs_fn)
                    render_kwargs_train = dict(render_kwargs_train)  # make a writable copy

                    # Force metric depth supervision
                    render_kwargs_train['ndc']  = False
                    render_kwargs_train['near'] = config.get('near', 0.1)
                    render_kwargs_train['far']  = config.get('far', 15.0)

                    
                    for p in range(config["poses"]):
                        img_i = np.random.choice(len(entry['images'][j]), 1)
                        target = entry['images'][j][img_i][0].to(device)
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
                        
                        #Calculate rays from camera origin
                        rays_o, rays_d = get_rays(H, W, K, torch.Tensor(pose.float())) 
                        
                        # Create coordinates array (for ray selection)
                        coords = torch.stack(
                            torch.meshgrid(
                                torch.linspace(0, H-1, H, device=device),
                                torch.linspace(0, W-1, W, device=device),
                                indexing='ij'
                            ),
                            -1
                        )  # (H, W, 2)

                        # To 1D
                        coords = coords.reshape(-1, 2)  # (H * W, 2)

                        '''N_rand = config['model']['TN']['N_rand']

                        # We will reuse depth_img later
                        depth_img = None
                        if 'depths' in entry:
                            depth_img = entry['depths'][j][img_i][0].to(device).float()  # (H,W)

                        # Select rays based on a mask if depth is available
                        if 'depths' in entry:
                            # depth_img on GPU
                            depth_img = entry['depths'][j][img_i][0].to(device).float()  # (H, W)
                            valid_mask = (depth_img > 0).reshape(-1)                     # (H*W,)
                            valid_inds = torch.where(valid_mask)[0]                      # (num_valid,) on GPU

                            if valid_inds.numel() == 0:
                                # no valid depth -> sample random pixels
                                select_inds = torch.randperm(coords.shape[0], device=device)[:N_rand]
                            elif valid_inds.numel() >= N_rand:
                                # enough valid -> sample witho ut replacement
                                rand_idx = torch.randperm(valid_inds.numel(), device=device)[:N_rand]
                                select_inds = valid_inds[rand_idx]
                            else:
                                # not enough valid -> sample WITH replacement from valid pixels
                                rand_idx = torch.randint(0, valid_inds.numel(), (N_rand,), device=device)
                                select_inds = valid_inds[rand_idx]
                        else:
                            # no depth in this batch -> random pixels
                            select_inds = torch.randperm(coords.shape[0], device=device)[:N_rand]
                        # Random pixel selection (no depth-based sampling)
                        select_inds = torch.randperm(coords.shape[0], device=device)[:N_rand]'''

                        N_rand = config['model']['TN']['N_rand']

                        # ---- load depth image (independent of ray sampling) ----
                        depth_img = None
                        if 'depths' in entry:
                            depth_img = entry['depths'][j][img_i][0].to(device).float()  # (H, W)

                        # ---- 50/50 sampling: 70% from depth>0 pixels, 50% from depth==0 pixels ----
                        if depth_img is not None:
                            flat = depth_img.reshape(-1)  # (H*W,)
                            valid_inds = torch.where(flat > 0.0)[0]
                            bg_inds    = torch.where(flat <= 0.0)[0]

                            n_valid = int(0.5 * N_rand)
                            n_bg    = N_rand - n_valid

                            # sample valid pixels
                            if valid_inds.numel() >= n_valid:
                                v = valid_inds[torch.randperm(valid_inds.numel(), device=device)[:n_valid]]
                            elif valid_inds.numel() > 0:
                                v = valid_inds[torch.randint(0, valid_inds.numel(), (n_valid,), device=device)]
                            else:
                                v = torch.empty((0,), dtype=torch.long, device=device)

                            # sample background (depth==0) pixels
                            if bg_inds.numel() >= n_bg:
                                b = bg_inds[torch.randperm(bg_inds.numel(), device=device)[:n_bg]]
                            elif bg_inds.numel() > 0:
                                b = bg_inds[torch.randint(0, bg_inds.numel(), (n_bg,), device=device)]
                            else:
                                b = torch.randperm(coords.shape[0], device=device)[:n_bg]

                            select_inds = torch.cat([v, b], dim=0)
                            # shuffle so rays aren't grouped
                            select_inds = select_inds[torch.randperm(select_inds.numel(), device=device)]
                        else:
                            # no depth available -> uniform sampling
                            select_inds = torch.randperm(coords.shape[0], device=device)[:N_rand]





                        # Use the selected coordinates
                        select_coords = coords[select_inds].long()          # (N_rand, 2)
                        r_idx = select_coords[:, 0]
                        c_idx = select_coords[:, 1]

                        rays_o = rays_o[r_idx, c_idx]                       # (N_rand, 3)
                        rays_d = rays_d[r_idx, c_idx]                       # (N_rand, 3)
                        batch_rays = torch.stack([rays_o, rays_d], 0)
                        target_s = target[r_idx, c_idx]                     # (N_rand, 3)

                        # depth supervision (if your npz contains 'depths')
                        if depth_img is not None:
                            depth_gt = depth_img[r_idx, c_idx]              # (N_rand,)
                        else:
                            depth_gt = None


                        if epoch == 0 and i == 0 and j == 0:
                            print("entry keys:", entry.keys())
                            if depth_gt is not None:
                                print("depth_gt min/max:", float(depth_gt.min()), float(depth_gt.max()),
                                    "valid sampled rays:", int((depth_gt > 0).sum()), "/", depth_gt.numel())



                        if epoch == 0 and i == 0 and j == 0:
                            print("render_kwargs_train contains near?", 'near' in render_kwargs_train)
                            print("near/far/ndc:", render_kwargs_train['near'], render_kwargs_train['far'], render_kwargs_train['ndc'])

                        img_r, _, _, extras = render(
                            H, W, K,
                            chunk=config['model']['TN']['netchunk'],
                            rays=batch_rays.to(device),
                            verbose=True, retraw=True,
                            **render_kwargs_train
                        )

                        if epoch == 0 and i == 0 and j == 0:
                            print("render module:", render.__module__)
                            print("extras keys:", extras.keys())

                        depth_pred = extras['depth_map']        # (N_rand,)


                        x.append(target_s)
                        y.append(img_r)

                        if depth_gt is not None:
                            depth_x.append(depth_gt)
                            depth_y.append(depth_pred)

                optimizer.zero_grad()
                x = torch.stack(x)
                y = torch.stack(y)

                loss_r = loss_fn(y, x)

                '''loss_kld = 0.5 * (torch.exp(logvar) + torch.pow(mu, 2) - 1 - logvar).sum()

                loss = loss_r + loss_kld'''
                # depth loss (masked: only where GT depth > 0)
                if len(depth_x) > 0:
                    depth_gt = torch.stack(depth_x)      # (num_sets, N_rand)
                    depth_pr = torch.stack(depth_y)      # (num_sets, N_rand)

                    valid = depth_gt > 0.0               # 0 means "no depth" in your dataset
                    if valid.any():
                        loss_depth = ((depth_pr[valid] - depth_gt[valid]) ** 2).mean()
                    else:
                        loss_depth = torch.tensor(0.0, device=y.device)
                else:
                    loss_depth = torch.tensor(0.0, device=y.device)

                loss_kld = 0.5 * (torch.exp(logvar) + torch.pow(mu, 2) - 1 - logvar).sum()

                loss = loss_r + loss_kld + lambda_depth * loss_depth

                loss.backward()
                optimizer.step()
                
                total_loss_r += loss_r.item()
                total_loss += loss.item()
                total_loss_kld += loss_kld.item()
                total_loss_depth += loss_depth.item()
                global_batch_idx += 1
            
            losses_r.append(total_loss_r)
            losses_kld.append(total_loss_kld)
            losses_depth.append(total_loss_depth)
            losses_total.append(total_loss)


            '''mlflow.log_metrics({
                'batch_loss_r': loss_r.item(),
                'batch_loss_kld': loss_kld.item(),
                'batch_loss_depth': loss_depth.item(),
                'batch_loss': loss.item()
            },
            step=global_batch_idx)'''


                
            if epoch > 0:
                mlflow.log_metrics({
                    'epoch_total_loss_r': total_loss_r,
                    'epoch_total_loss_kld': total_loss_kld,
                    'epoch_total_loss_depth':total_loss_depth,
                    'epoch_total_loss': total_loss,

                    'λ': lambda_depth,

                    'train_epoch_time_seconds': round((datetime.now() - start_epoch_time).total_seconds(), 3),
                },
                step=epoch)

            with open(loss_log_path, "a") as f:
                f.write(f"{epoch},{total_loss},{total_loss_r},{total_loss_kld},{total_loss_depth},{lambda_depth}\n")


            scheduler.step()

            #Log information, save models etc.
            if epoch % config['i_log'] == 0:
                print(f"Epoch {epoch}: took {round((datetime.now() - start_epoch_time).total_seconds(), 3)} seconds")
                print(f"Total loss: {total_loss}     Loss R: {total_loss_r}     Loss KLD: {total_loss_kld} Loss Depth: {total_loss_depth}  (λ={lambda_depth})")
            
            #Compare current reconstruction
            if epoch % config['i_sample'] == 0 or epoch == 0:
                with torch.no_grad():
                    render_kwargs_test = {
                        k: render_kwargs_train[k] for k in render_kwargs_train}
                    render_kwargs_test['perturb'] = False
                    render_kwargs_test['raw_noise_std'] = 0.
                    img, _, _, _ = render(H,W,K, chunk=config['model']['TN']['netchunk'], c2w=pose,
                                                        verbose=True, retraw=True,
                                                        **render_kwargs_test)
                    f, axarr = plt.subplots(1,2)
                    axarr[0].imshow(img.detach().cpu())
                    axarr[1].imshow(target.detach().cpu())
                    f.savefig(join(results_dir, 'samples', f"epoch_{epoch}.png"))
                    plt.close(f)
                    
                    
            if epoch % config['i_save']==0:  
                torch.save(hnet.state_dict(), join(results_dir, f"model_hn_{epoch}.pt"))
                torch.save(encoder.state_dict(), join(results_dir, f"model_e_{epoch}.pt"))
                torch.save(scheduler.state_dict(), join(results_dir, f"lr_{epoch}.pt"))
                torch.save(optimizer.state_dict(), join(results_dir, f"opt_{epoch}.pt"))
                
                np.save(join(results_dir, 'losses_r.npy'), np.array(losses_r))
                np.save(join(results_dir, 'losses_kld.npy'), np.array(losses_kld))
                np.save(join(results_dir, 'losses_total.npy'), np.array(losses_total))
                np.save(join(results_dir, 'losses_depth.npy'), np.array(losses_depth))

                plt.plot(losses_r)
                plt.savefig(os.path.join(results_dir, f'loss_r_plot.png'))
                plt.close()

                plt.loglog(losses_r)
                plt.savefig(os.path.join(results_dir, f'loss_r_plot_log.png'))
                plt.close()

                plt.plot(losses_kld)
                plt.savefig(os.path.join(results_dir, f'loss_kld_plot.png'))
                plt.close()

                plt.plot(losses_total)
                plt.savefig(os.path.join(results_dir, f'loss_total_plot.png'))
                plt.close()                
