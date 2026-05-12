import numpy as np
from sklearn.metrics import confusion_matrix
import threading
import torch

from configs.default import Config as cfg


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class IoUCalculator:
    
    def __init__(self):
        
        self.gt_classes = [0 for _ in range(cfg.num_classes)]
        self.positive_classes = [0 for _ in range(cfg.num_classes)]
        self.true_positive_classes = [0 for _ in range(cfg.num_classes)]
        self.cfg = cfg
        self.lock = threading.Lock()
        
        self.val_total_correct = 0
        self.val_total_seen = 0

    def add_data(self, end_points):
        
        logits = end_points['valid_logits']
        labels = end_points['valid_labels']
        if cfg.enc == 'ohe':
            preds = logits.max(dim=1)[1]
        elif cfg.enc == 'se':
            logits = logits.unsqueeze(1).repeat(1, cfg.num_classes, 1)
            cus_enc = torch.tensor(cfg.cus_enc).to(device)
            dis = torch.norm(logits - cus_enc, dim=-1)
            preds = dis.min(dim=1)[1]
            
        end_points['preds'] = preds
        
        preds_valid = preds.detach().cpu().numpy()
        labels_valid = labels.detach().cpu().numpy()

        correct = np.sum(preds_valid == labels_valid)
        self.val_total_correct += correct
        self.val_total_seen += len(labels_valid)

        conf_matrix = confusion_matrix(labels_valid, preds_valid, labels=np.arange(0, self.cfg.num_classes, 1))
        self.lock.acquire()
        self.gt_classes += np.sum(conf_matrix, axis=1)
        self.positive_classes += np.sum(conf_matrix, axis=0)
        self.true_positive_classes += np.diagonal(conf_matrix)
        self.lock.release()
        
        return end_points

    def compute_iou(self):
        
        iou_list = []
        for n in range(0, self.cfg.num_classes, 1):
            if float(self.gt_classes[n] + self.positive_classes[n] - self.true_positive_classes[n]) != 0:
                iou = self.true_positive_classes[n] / float(self.gt_classes[n] + self.positive_classes[n] - self.true_positive_classes[n])
                iou_list.append(iou)
            else:
                iou_list.append(0.0)
        mean_iou = sum(iou_list) / float(self.cfg.num_classes)
        
        return mean_iou, iou_list
    
    def compute_oa(self):
        
        oa = self.val_total_correct / self.val_total_seen
        
        return oa


