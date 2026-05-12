import logging
import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from configs.default import Config as cfg
from src.datasets.seg2tunnel import Seg2Tunnel
from src.models.network import Network
    

torch.backends.cudnn.benchmark = True


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


class Tester:
    
    def __init__(self):
        
        self.data_path = cfg.data_path
        
        self.log_dir = cfg.log_dir
        if not os.path.exists(self.log_dir):
            os.mkdir(self.log_dir)
        log_fname = os.path.join(self.log_dir, 'log_test.txt')
        LOGGING_FORMAT = '%(asctime)s %(levelname)s: %(message)s'
        DATE_FORMAT = '%Y%m%d %H:%M:%S'
        logging.basicConfig(level=logging.DEBUG, format=LOGGING_FORMAT, datefmt=DATE_FORMAT, filename=log_fname)
        self.logger = logging.getLogger("Tester")
        
        if not os.path.exists(cfg.result_path):
            os.mkdir(cfg.result_path)
        
        test_set = Seg2Tunnel('demo')
        self.test_loader = DataLoader(test_set, 
                                      batch_size=cfg.test_batch_size, 
                                      shuffle=False, 
                                      num_workers=cfg.num_workers, 
                                      collate_fn=test_set.collate_fn, 
                                      worker_init_fn=worker_init_fn, 
                                      pin_memory=True)
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net = Network(cfg)
        self.net.to(device)
        
        CHECKPOINT_PATH = cfg.checkpoint_path
        if CHECKPOINT_PATH is not None and os.path.isfile(CHECKPOINT_PATH):
            self.logger.info('Use the existing model!')
            checkpoint = torch.load(CHECKPOINT_PATH, weights_only=False)
            self.net.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.logger.info('There is no existing model!')
            exit()

        if torch.cuda.device_count() > 1:
            self.logger.info('Use %d GPUs!' % (torch.cuda.device_count()))
            self.net = nn.DataParallel(self.net)
        
        self.test_set = test_set

    def test(self):
        
        self.net.eval()

        tqdm_loader = tqdm(self.test_loader, total=len(self.test_loader))
        
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(tqdm_loader):
                
                for key in batch_data:
                    if type(batch_data[key]) is list:
                        for i in range(cfg.num_layers):
                            batch_data[key][i] = batch_data[key][i].cuda(non_blocking=True)
                    else:
                        batch_data[key] = batch_data[key].cuda(non_blocking=True)
                torch.cuda.synchronize()
                
                end_points = self.net(batch_data)
                
                if cfg.flag_vis:
                    str_fmt = '%.8f %.8f %.8f %.8f'
                    for vis_layer in cfg.vis_layers:
                        xyz = end_points['xyz'][vis_layer]
                        xyz = np.asarray(xyz.cpu()).reshape(-1, 3)
                        for vis_channel in cfg.vis_channels:
                            fm = end_points['fm_LFA'][vis_layer][vis_channel]
                            fm = np.asarray(fm.cpu()).reshape(1, -1).T
                            xyz_fm = np.hstack((xyz, fm))
                            np.savetxt(
                                os.path.join(os.path.join(cfg.result_path, 'fm_lfa_' + str(vis_layer) + '_' + str(vis_channel) + '.txt')),
                                xyz_fm,
                                fmt=str_fmt
                            )
                            fm = end_points['fm_RFA'][vis_layer][vis_channel]
                            fm = np.asarray(fm.cpu()).reshape(1, -1).T
                            xyz_fm = np.hstack((xyz, fm))
                            np.savetxt(
                                os.path.join(os.path.join(cfg.result_path, 'fm_rfa_' + str(vis_layer) + '_' + str(vis_channel) + '.txt')),
                                xyz_fm,
                                fmt=str_fmt
                            )
                            

def main():
    
    tester = Tester()
    tester.test()


if __name__ == '__main__':
    main()