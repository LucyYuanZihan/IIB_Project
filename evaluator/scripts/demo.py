import logging
import numpy as np
import os
from thop import profile
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from configs.default import Config as cfg
from src.datasets.seg2tunnel import Seg2Tunnel
from src.models.network import Network


torch.backends.cudnn.benchmark = True


def worker_init_fn(worker_id):
    
    np.random.seed(np.random.get_state()[1][0] + worker_id)


class Demo:
    
    def __init__(self):
        
        self.log_dir = cfg.log_dir
        if not os.path.exists(self.log_dir):
            os.mkdir(self.log_dir)
        log_fname = os.path.join(self.log_dir, 'log_demo.txt')
        LOGGING_FORMAT = '%(asctime)s %(levelname)s: %(message)s'
        DATE_FORMAT = '%Y%m%d %H:%M:%S'
        logging.basicConfig(level=logging.DEBUG, format=LOGGING_FORMAT, datefmt=DATE_FORMAT, filename=log_fname)
        self.logger = logging.getLogger("Demo")
        
        demo_set = Seg2Tunnel('demo')
        self.demo_loader = DataLoader(
            demo_set,
            batch_size=cfg.demo_batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            collate_fn=demo_set.collate_fn,
            worker_init_fn=worker_init_fn,
            pin_memory=True
        )
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.net = Network(cfg)
        self.net.to(device)

        self.optimizer = optim.Adam(self.net.parameters(), lr=cfg.learning_rate)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, 0.95)

        if torch.cuda.device_count() > 1:
            self.logger.info('Use %d GPUs!' % (torch.cuda.device_count()))
            self.net = nn.DataParallel(self.net)
    
    def demo(self):
        
        self.net.eval()
        
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(self.demo_loader):
                for key in batch_data:
                    if type(batch_data[key]) is list:
                        for i in range(cfg.num_layers):
                            batch_data[key][i] = batch_data[key][i].cuda(non_blocking=True)
                    else:
                        batch_data[key] = batch_data[key].cuda(non_blocking=True)

                torch.cuda.synchronize()
                flops, params = profile(self.net, (batch_data,))
                self.logger.info('FLOPs: {:.2f}'.format(flops / 1000**3))
                self.logger.info('Params: {:.2f}'.format(params / 1000**2))


def main():
    
    demo = Demo()
    demo.demo()


if __name__ == '__main__':
    main()