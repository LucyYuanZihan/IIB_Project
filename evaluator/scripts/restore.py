import numpy as np
import os
import pandas as pd
import pickle

from configs.default import Config as cfg


if __name__ == '__main__':
    
    voxel_size = cfg.voxel_size
    result_path = cfg.result_path
    output_path = cfg.data_path + '_' + str(voxel_size)
    stations = cfg.test_stations
    
    result_restore_path = result_path + '_restore'
    if not os.path.exists(result_restore_path):
        os.mkdir(result_restore_path)
    
    mean_xyz_file = os.path.join(output_path, 'mean_xyz.pkl')
    with open(mean_xyz_file, 'rb') as f:
        mean_xyz = pickle.load(f)
    
    str_fmt = '%.8f ' * cfg.num_raw_features + '%d ' * 2 + '%.8f'
    
    for station in stations:
        result = pd.read_csv(os.path.join(result_path, station + '.txt'), sep=' ', header=None)
        result = np.asarray(result)
        result[:, 0:3] += mean_xyz[station]
        
        np.savetxt(
            os.path.join(result_restore_path, station + '.txt'),
            result,
            fmt=str_fmt
        )