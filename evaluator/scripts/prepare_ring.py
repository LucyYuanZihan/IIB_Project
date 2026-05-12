import numpy as np
import os
import pandas as pd
import pickle
from sklearn.neighbors import KDTree

from configs.ring import Config as cfg


"""
File: prepare.py
Project: LiningNet

Description:
This script handles the data preprocessing pipeline for ring-wise testing. 
This script handles the rigorous data preprocessing pipeline for the LiningNet 
architecture. It ingests raw point cloud data (in space-separated text/CSV format) 
of segmental tunnel linings, applies necessary geometric and radiometric transformations, 
and outputs highly optimized binary files ready for 3D deep learning model training 
and evaluation.

Key Operations:
* Data Grouping: Aggregates individual ring point clouds into continuous tunnel 
    "stations" or scenes based on configuration settings.
* Voxel Downsampling: Applies a voxel grid filter to uniformize point density 
    and reduce data redundancy.
* Intensity Normalization: Clips raw laser intensity at the 1st and 99th percentiles 
    to remove outliers, then min-max scales the values to a [0, 1] range.
* Mean Centering: Translates the point cloud to the origin (0,0,0) by subtracting 
    the mean XYZ coordinates to stabilize network training.
* Spatial Partitioning: Constructs and saves a scikit-learn KDTree for rapid 
    nearest-neighbor queries (used for sphere-cropping during data loading).

Inputs:
* Raw point cloud files (.txt or .csv) located in `cfg.data_path`.
* Configuration parameters imported from `config.py` (e.g., `voxel_size`, 
    `flag_prep`, train/test splits).

Outputs (saved to `cfg.data_path_{voxel_size}`):
* `{station}.npy`: Processed, downsampled, centered, and shuffled point cloud arrays.
* `{station}_KDTree.pkl`: Pickled KDTree objects for fast spatial subsetting.
* `mean_xyz.pkl`: A dictionary mapping station names to their original geographic 
    centers, allowing predictions to be mapped back to real-world coordinates.
"""


def grid_sample(points, voxel_size):
    
    features = points[:, 3:]
    points = points[:, 0:3]

    non_empty_voxel_keys, inverse, nb_pts_per_voxel = np.unique(((points - np.min(points, axis=0)) // voxel_size).astype(int), axis=0, return_inverse=True, return_counts=True)
    idx_pts_vox_sorted = np.argsort(inverse)
    voxel_grid={}
    voxel_grid_f={}
    sub_points, sub_features = [], []
    last_seen=0

    for idx, vox in enumerate(non_empty_voxel_keys):
        voxel_grid[tuple(vox)] = points[idx_pts_vox_sorted[last_seen: last_seen+nb_pts_per_voxel[idx]]]
        voxel_grid_f[tuple(vox)] = features[idx_pts_vox_sorted[last_seen: last_seen+nb_pts_per_voxel[idx]]]
        sub_points.append(voxel_grid[tuple(vox)][np.linalg.norm(voxel_grid[tuple(vox)] - np.mean(voxel_grid[tuple(vox)], axis=0), axis=1).argmin()])
        sub_features.append(voxel_grid_f[tuple(vox)][np.linalg.norm(voxel_grid_f[tuple(vox)] - np.mean(voxel_grid_f[tuple(vox)], axis=0), axis=1).argmin()])
        last_seen += nb_pts_per_voxel[idx]
        
    sub_points = np.hstack((np.asarray(sub_points), np.asarray(sub_features)))

    return sub_points


def norm_intensity(intensity):
    
    """This function calculates the 1st and 99th percentiles of the intensity values, 
    clips any outliers outside this range, and then scales all intensity values to sit neatly between 0 and 1."""
    
    bottom, up = np.percentile(intensity, 1), np.percentile(intensity, 99)
    if bottom != up:
        intensity[intensity < bottom] = bottom
        intensity[intensity > up] = up
        intensity -= bottom
        intensity = intensity / (up - bottom)
    
    return intensity


def prepare():
    
    voxel_size = cfg.voxel_size
    
    input_path = cfg.data_path
    files = os.listdir(input_path)
    output_path = cfg.data_path + '_' + str(voxel_size)
        
    stations = {}
    
    for file in files:
        
        if cfg.flag_prep == 'ring-wise':
            station = file.rsplit('-', 1)[0]
        elif cfg.flag_prep == 'scene-wise':
            station = file.rsplit('.', 1)[0]
            
        if (station not in cfg.training_stations) and (station not in cfg.test_stations) and (station not in cfg.test_stations):
            continue
        if station not in stations.keys():
            stations[station] = []
        stations[station].append(file)
    
    mean_xyz = {}
    
    for station in stations.keys():
        pc = []
        for i in range(len(stations[station])):
            ring = pd.read_csv(os.path.join(input_path, stations[station][i]), sep=' ', header=None)
            ring = np.asarray(ring)
            pc.append(ring)
        pc = np.vstack(pc)
        
        if voxel_size != 0:
            pc = grid_sample(pc, voxel_size)
        
        pc[:, 3] = norm_intensity(pc[:, 3])
        
        mean_xyz[station] = np.mean(pc[:, 0:3], axis=0)
        pc[:, 0:3] -= mean_xyz[station]
        np.random.shuffle(pc)
        
        search_tree = KDTree(pc[:, 0:3])
        
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        np.save(output_path + '/' + station + '.npy', pc)
        kd_tree_file = os.path.join(output_path, station + '_KDTree.pkl')
        with open(kd_tree_file, 'wb') as f:
            pickle.dump(search_tree, f)
    
    mean_xyz_file = os.path.join(output_path, 'mean_xyz.pkl')
    with open(mean_xyz_file, 'wb') as f:
        pickle.dump(mean_xyz, f)


if __name__ == '__main__':
    
    prepare()