import numpy as np
import open3d as o3d
import pickle
from scipy import optimize
import torch
from torch.utils.data import Dataset

from configs.default import Config as cfg


def compute_dis2axis(p, xy):
    
    a, b, c = p[0], p[1], p[2]
    d2 = (a * xy[:, 0] + b * xy[:, 1] + c) ** 2 / (a ** 2 + b ** 2)
    
    return d2


def project2axis(xy, a, b, c):
    
    x0, y0 = xy[:, 0], xy[:, 1]
    
    x = x0 - a * (a * x0 + b * y0 + c) / (a ** 2 + b ** 2)
    y = y0 - b * (a * x0 + b * y0 + c) / (a ** 2 + b ** 2)
    d = x / abs(x) * np.sqrt(x ** 2 + (y + c / b) ** 2)
    
    return d


def rotate_random_2d(pc):
    
    r = np.random.uniform(0, 360) / 180 * np.pi
    R = np.array([[np.cos(r), -np.sin(r), 0], [np.sin(r), np.cos(r), 0], [0, 0, 1]])
    pc[:, 0:3] = np.dot(R, pc[:, 0:3].T).T
    
    return pc


class Seg2Tunnel(Dataset):
    
    def __init__(self, mode):
        
        self.num_classes = cfg.num_classes
        self.ignored_labels = None
        self.mode = mode
        self.list_dataset = []
        self.path = cfg.data_path + '_' + str(cfg.voxel_size)
        
        if self.mode == 'training':
            stations = cfg.training_stations
            for i in range(cfg.training_num):
                self.list_dataset.append(stations[i%len(stations)])
        if self.mode == 'validation':
            stations = cfg.validation_stations
            for i in range(cfg.validation_num):
                self.list_dataset.append(stations[i%len(stations)])
        if self.mode == 'test':
            stations = cfg.test_stations
            for i in range(cfg.test_num):
                for station in stations:
                    self.list_dataset.append(station)
        if self.mode == 'demo':
            stations = cfg.demo_stations
            for i in range(cfg.demo_num):
                self.list_dataset.append(stations[i%len(stations)])
        
        self.possibility = {}
        
        if cfg.rfa:
            self.axis_ds = {}
            pini = np.asarray([1, 1, 0], dtype=float)
        
        for station in stations:
            raw_pc = np.load(self.path + '/' + station + '.npy')
            self.possibility[station] = np.random.rand(raw_pc.shape[0]) * 1e-8
            if cfg.rfa:
                plsq = optimize.leastsq(compute_dis2axis, pini, args=(raw_pc[:, 0:2]))
                a, b, c = plsq[0][0], plsq[0][1], plsq[0][2]
                axis_d = project2axis(raw_pc[:, 0:2], a, b, c)
                self.axis_ds[station] = axis_d

    def __len__(self):
        
        return len(self.list_dataset)
    
    def __getitem__(self, index):
        
        station = self.list_dataset[index]
        raw_pc = np.load(self.path + '/' + station + '.npy')
        
        if cfg.rfa:
            new_raw_pc = np.zeros((raw_pc.shape[0], raw_pc.shape[1] + 1))
            new_raw_pc[:, 0:cfg.num_raw_features] = raw_pc[:, 0:cfg.num_raw_features]
            new_raw_pc[:, cfg.num_raw_features] = self.axis_ds[station][:]
            new_raw_pc[:, -1] = raw_pc[:, -1]
            raw_pc = new_raw_pc
        
        if cfg.flag_pipe == 'sphere_crop':
            index_min = np.argmin(self.possibility[station])
            centre = raw_pc[index_min, :]
            kd_tree_file = self.path + '/' + station + '_KDTree.pkl'
            with open(kd_tree_file, 'rb') as f:
                search_tree = pickle.load(f)
            dist, neigh_idx = search_tree.query([centre[0:3]], k=cfg.num_points)
            dist = dist[0]
            neigh_idx = neigh_idx[0]
            delta = np.square(1 - dist / np.max(dist))
            self.possibility[station][neigh_idx] += delta
        elif cfg.flag_pipe == 'sample_random':
            neigh_idx = np.argsort(self.possibility[station])[0:cfg.num_points]
            self.possibility[station][neigh_idx] += np.random.rand(cfg.num_points)
        
        raw_pc = raw_pc[neigh_idx, :]
        
        if self.mode == 'training':
            raw_pc = rotate_random_2d(raw_pc)
            np.random.shuffle(raw_pc)
        if self.mode == 'validation':
            np.random.shuffle(raw_pc)
        if self.mode == 'test':
            raw_pc_neigh_idx = np.hstack((raw_pc, np.asarray(neigh_idx)[:, None]))
            np.random.shuffle(raw_pc_neigh_idx)
            raw_pc = raw_pc_neigh_idx[:, 0:-1]
            neigh_idx = raw_pc_neigh_idx[:, -1]
        if self.mode == 'demo':
            np.random.shuffle(raw_pc)
        
        pc = raw_pc
        
        dataset = {}
        
        dataset['features'] = torch.from_numpy(pc[:, 0:cfg.num_raw_features]).transpose(0, 1).float()
        dataset['labels'] = []
        dataset['xyz'] = []
        dataset['neigh_idx'] = []
        dataset['sub_idx'] = []
        dataset['interp_idx'] = []
        
        if self.mode == 'test':
            dataset['test_idx'] = torch.from_numpy(neigh_idx).long()
        
        xyz = pc[:, 0:3]
        labels = pc[:, -1]
        num = xyz.shape[0]
        
        if cfg.rfa:
            dataset['axis_neigh_idx'] = []
            axis_d = pc[:, cfg.num_raw_features]
        
        for i in range(cfg.num_layers):
            
            dataset['xyz'].append(torch.from_numpy(xyz).float())
            dataset['labels'].append(torch.from_numpy(labels).long())
            
            num = int(num / cfg.sub_sampling_ratio[i])
            
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(xyz)
            pcd_tree = o3d.geometry.KDTreeFlann(pcd)
            neigh_idxs = np.zeros((xyz.shape[0], cfg.k_n), dtype=int)
            for j in range(xyz.shape[0]):
                [_, neigh_idx, _] = pcd_tree.search_knn_vector_3d(xyz[j], cfg.k_n)
                neigh_idx = np.asarray(neigh_idx, dtype=int)
                neigh_idxs[j, :] = neigh_idx[:]
            sub_idxs = neigh_idxs[0:num, :]
            dataset['neigh_idx'].append(torch.from_numpy(neigh_idxs).long())
            dataset['sub_idx'].append(torch.from_numpy(sub_idxs).long())
            
            sub_xyz = xyz[0:num, :]
            sub_pcd = o3d.geometry.PointCloud()
            sub_pcd.points = o3d.utility.Vector3dVector(sub_xyz)
            sub_pcd_tree = o3d.geometry.KDTreeFlann(sub_pcd)
            interp_idxs = np.zeros((xyz.shape[0]), dtype=int)
            for j in range(xyz.shape[0]):
                [_, interp_idx, _] = sub_pcd_tree.search_knn_vector_3d(xyz[j], 1)
                interp_idx = np.asarray(interp_idx, dtype=int)
                interp_idxs[j] = interp_idx[:]
            dataset['interp_idx'].append(torch.from_numpy(interp_idxs).long())
            
            labels = labels[0:num]
            xyz = sub_xyz
            
            if cfg.rfa:
                pcd = o3d.geometry.PointCloud()
                d00 = np.zeros((axis_d.shape[0], 3))
                d00[:, 0] = axis_d[:]
                pcd.points = o3d.utility.Vector3dVector(d00)
                pcd_tree = o3d.geometry.KDTreeFlann(pcd)
                axis_neigh_idxs = np.zeros((axis_d.shape[0], cfg.axis_k_n), dtype=int)
                for j in range(d00.shape[0]):
                    [_, neigh_idx, _] = pcd_tree.search_knn_vector_3d(d00[j], cfg.axis_k_n)
                    neigh_idx = np.asarray(neigh_idx, dtype=int)
                    axis_neigh_idxs[j, :] = neigh_idx[:]
                dataset['axis_neigh_idx'].append(torch.from_numpy(axis_neigh_idxs).long())
                axis_d = axis_d[0:num]
            
        return dataset
    
    def collate_fn(self, batches):
        
        inputs = {}
        inputs['features'] = []
        inputs['xyz'] = []
        inputs['neigh_idx'] = []
        inputs['sub_idx'] = []
        inputs['interp_idx'] = []
        inputs['labels'] = []
        
        for i in range(cfg.num_layers):
            inputs['xyz'].append([])
            inputs['neigh_idx'].append([])
            inputs['sub_idx'].append([])
            inputs['interp_idx'].append([])
            inputs['labels'].append([])
        
        for batch in batches:
            inputs['features'].append(batch['features'])
            for i in range(cfg.num_layers):
                inputs['xyz'][i].append(batch['xyz'][i])
                inputs['neigh_idx'][i].append(batch['neigh_idx'][i])
                inputs['sub_idx'][i].append(batch['sub_idx'][i])
                inputs['interp_idx'][i].append(batch['interp_idx'][i])
                inputs['labels'][i].append(batch['labels'][i])
        
        inputs['features'] = torch.stack(inputs['features'], 0)
        
        for i in range(cfg.num_layers):
            inputs['xyz'][i] = torch.stack(inputs['xyz'][i], 0)
            inputs['neigh_idx'][i] = torch.stack(inputs['neigh_idx'][i], 0)
            inputs['sub_idx'][i] = torch.stack(inputs['sub_idx'][i], 0)
            inputs['interp_idx'][i] = torch.stack(inputs['interp_idx'][i], 0)
            inputs['labels'][i] = torch.stack(inputs['labels'][i], 0)
                
        if self.mode == 'test':
            for batch in batches:
                inputs['test_idx'] = batch['test_idx']
        
        if cfg.rfa:
            inputs['axis_neigh_idx'] = []
            for i in range(cfg.num_layers):
                inputs['axis_neigh_idx'].append([])
            for batch in batches:
                for i in range(cfg.num_layers):
                    inputs['axis_neigh_idx'][i].append(batch['axis_neigh_idx'][i])
            for i in range(cfg.num_layers):
                inputs['axis_neigh_idx'][i] = torch.stack(inputs['axis_neigh_idx'][i], 0)

        return inputs

'''
--------VISUALISATION--------
'''
def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)
    
    
if __name__ == '__main__':
    dataset = Seg2Tunnel('test')
    feature = np.asarray(dataset[0]['features'].transpose(0, 1))
    np.savetxt('dataset.txt', feature, delimiter=' ')