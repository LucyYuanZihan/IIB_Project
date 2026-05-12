from concurrent.futures import ProcessPoolExecutor
import logging
import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from configs.ring import Config as cfg
from src.datasets.ring import Seg2Tunnel
from src.losses.losses import compute_loss
from src.metrics.iou import IoUCalculator
from src.models.network import Network
    

torch.backends.cudnn.benchmark = True


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def process_results(result):
    if not result:
        return [0, 0.0]
    pred = max(result, key=result.count)
    count = result.count(pred)
    frequency = float(count) / len(result)
    return [pred, frequency]


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
        
        test_set = Seg2Tunnel('test')
        self.test_loader = DataLoader(
            test_set, 
            batch_size=cfg.test_batch_size, 
            shuffle=False, 
            num_workers=cfg.num_workers, 
            collate_fn=test_set.collate_fn, 
            worker_init_fn=worker_init_fn, 
            pin_memory=True
            )
        
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
        
        iou_calc = IoUCalculator()
        
        results = {}
        for station in cfg.test_stations:
            raw_pc = np.load(self.data_path + '_' + str(cfg.voxel_size) + '/' + station + '.npy')
            results[station] = []
            results[station + 'pc'] = raw_pc
            for _ in range(raw_pc.shape[0]):
                results[station].append([])

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
                
                loss, end_points = compute_loss(end_points)
                end_points = iou_calc.add_data(end_points)
                
                station = cfg.test_stations[batch_idx % len(cfg.test_stations)]
                test_idx = end_points['test_idx']
                test_idx = np.asarray(test_idx.cpu(), dtype=int)
                preds = end_points['preds']
                preds = np.asarray(preds.cpu())

                # only one vote per point in a ring if too few points in a ring
                _, unique_indices = np.unique(test_idx, return_index=True)
                for i in unique_indices:
                    results[station][test_idx[i]].append(preds[i])




                
                #for i in range(len(test_idx)):
                    #results[station][test_idx[i]].append(preds[i])
        
        str_fmt = '%.8f ' * cfg.num_raw_features + '%d ' * 2 + '%.8f'
        
        for station in cfg.test_stations:
            with ProcessPoolExecutor() as executor:
                results[station] = list(executor.map(process_results, results[station]))
            preds_frequencies = np.asarray(results[station])
            result = np.hstack((results[station + 'pc'], preds_frequencies))
            np.savetxt(
                os.path.join(cfg.result_path, station + '.txt'),
                result,
                fmt=str_fmt
            )
            
        oa = iou_calc.compute_oa()
        mean_iou, iou_list = iou_calc.compute_iou()
        self.logger.info('OA: {:.2f}'.format(oa * 100))
        self.logger.info('mIoU: {:.2f}'.format(mean_iou * 100))
        s = 'IoU:'
        for iou_tmp in iou_list:
            s += ' {:.2f}'.format(iou_tmp * 100)
        self.logger.info(s)
        
        return mean_iou


def main():
    
    tester = Tester()
    tester.test()


if __name__ == '__main__':
    main()