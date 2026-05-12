import logging
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
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


class Trainer:
    
    def __init__(self):
        
        self.log_dir = cfg.log_dir
        if not os.path.exists(self.log_dir):
            os.mkdir(self.log_dir)
        log_fname = os.path.join(self.log_dir, 'log_train.txt')
        LOGGING_FORMAT = '%(asctime)s %(levelname)s: %(message)s'
        DATE_FORMAT = '%Y%m%d %H:%M:%S'
        logging.basicConfig(level=logging.DEBUG, format=LOGGING_FORMAT, datefmt=DATE_FORMAT, filename=log_fname)
        self.logger = logging.getLogger("Trainer")
        
        training_set = Seg2Tunnel('training')
        validation_set = Seg2Tunnel('validation')
        
        self.training_loader = DataLoader(
            training_set,
            batch_size=cfg.training_batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            collate_fn=training_set.collate_fn,
            worker_init_fn=worker_init_fn,
            pin_memory=True
        )
        self.validation_loader = DataLoader(
            validation_set,
            batch_size=cfg.validation_batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            collate_fn=validation_set.collate_fn,
            worker_init_fn=worker_init_fn,
            pin_memory=True
        )
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.net = Network(cfg)
        self.net.to(device)

        self.optimizer = optim.Adam(self.net.parameters(), lr=cfg.learning_rate)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, 0.95)

        self.highest_val_iou = 0
        self.start_epoch = 0
        
        CHECKPOINT_PATH = cfg.checkpoint_path
        if CHECKPOINT_PATH is not None and os.path.isfile(CHECKPOINT_PATH):
            self.logger.info('Use the existing model!')
            checkpoint = torch.load(CHECKPOINT_PATH, weights_only=False)
            self.net.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.start_epoch = checkpoint['epoch']

        if torch.cuda.device_count() > 1:
            self.logger.info('Use %d GPUs!' % (torch.cuda.device_count()))
            self.net = nn.DataParallel(self.net)
        
        self.training_set = training_set
        self.validation_set = validation_set

    def train_one_epoch(self):
        
        self.net.train()
        
        tqdm_loader = tqdm(self.training_loader, total=len(self.training_loader))
        
        loss_one_epoch = 0
        count = 0
        
        for batch_idx, batch_data in enumerate(tqdm_loader):
            for key in batch_data:
                if type(batch_data[key]) is list:
                    for i in range(cfg.num_layers):
                        batch_data[key][i] = batch_data[key][i].cuda(non_blocking=True)
                else:
                    batch_data[key] = batch_data[key].cuda(non_blocking=True)

            self.optimizer.zero_grad()

            torch.cuda.synchronize()
            
            end_points = self.net(batch_data)
            
            loss, end_points = compute_loss(end_points)
            
            loss.backward()
            self.optimizer.step()
            
            loss_one_epoch += loss
            count += 1
        
        loss_one_epoch = loss_one_epoch / count
        self.logger.info('training set loss:{:.8f}'.format(loss_one_epoch))
        
        self.scheduler.step()

    def validate(self):
        
        self.net.eval()
        
        iou_calc = IoUCalculator()

        tqdm_loader = tqdm(self.validation_loader, total=len(self.validation_loader))
        
        loss_one_epoch = 0
        count = 0
        
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
                
                loss_one_epoch += loss
                count += 1
            
        loss_one_epoch = loss_one_epoch / count
        self.logger.info('validation set loss:{:.8f}'.format(loss_one_epoch))
        
        oa = iou_calc.compute_oa()
        mean_iou, iou_list = iou_calc.compute_iou()
        self.logger.info('OA: {:.2f}'.format(oa * 100))
        self.logger.info('mIoU: {:.2f}'.format(mean_iou * 100))
        s = 'IoU:'
        for iou_tmp in iou_list:
            s += ' {:.2f}'.format(iou_tmp * 100)
        self.logger.info(s)
        
        return mean_iou
    
    def save_checkpoint(self, fname):
        
        save_dict = {
            'epoch': self.cur_epoch + 1,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict()
        }
        
        try:
            save_dict['model_state_dict'] = self.net.module.state_dict()
        except AttributeError:
            save_dict['model_state_dict'] = self.net.state_dict()
        torch.save(save_dict, fname)
        
    def train(self):
        
        for epoch in range(self.start_epoch, cfg.max_epoch):
            self.cur_epoch = epoch
            self.logger.info('**** EPOCH %03d ****' % (epoch))
            self.train_one_epoch()
            self.logger.info('**** EVAL EPOCH %03d ****' % (epoch))
            mean_iou = self.validate()
            if mean_iou > self.highest_val_iou:
                self.highest_val_iou = mean_iou
                self.logger.info('Save the best model!')
                checkpoint_file = os.path.join(self.log_dir, 'checkpoint.pt')
                self.save_checkpoint(checkpoint_file)


def main():
    
    trainer = Trainer()
    trainer.train()


if __name__ == '__main__':
    main()