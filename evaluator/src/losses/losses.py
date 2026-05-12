import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.default import Config as cfg


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def compute_ohe2se_loss(logits, labels, cus_enc):
    
    ohe2se_logits = logits
    F.softmax(ohe2se_logits, dim=1)
    ohe2se_logits = torch.mm(ohe2se_logits, cus_enc)
    F.normalize(ohe2se_logits, dim=1)
    ohe2se_labels = cus_enc[labels[:], :]
    ohe2se_loss = torch.mean(torch.norm(ohe2se_logits - ohe2se_labels, dim=-1), dim=0)
    
    return ohe2se_loss


def compute_loss(end_points):

    if cfg.enc == 'ohe':
        if cfg.loss_func == 'cel':
            criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(cfg.weight_cel).to(device), reduction='mean')
        
        loss = 0
        if cfg.flag_ohe2se:
            cus_enc = torch.tensor(cfg.cus_enc).to(device)
        if cfg.flag_ml:
            for i in range(len(end_points['logits'])):
                logits = end_points['logits'][i]
                logits = logits.transpose(1, 2).reshape(-1, cfg.num_classes)
                valid_logits = logits
                labels = end_points['labels'][-i - 1]  
                labels = labels.reshape(-1)
                valid_labels = labels
                loss += criterion(valid_logits, valid_labels) * cfg.weight_ml[i]
                if cfg.flag_ohe2se:
                    loss += compute_ohe2se_loss(logits, labels, cus_enc) * cfg.weight_ohe2se
        else:
            logits = end_points['logits'][-1]
            logits = logits.transpose(1, 2).reshape(-1, cfg.num_classes)
            valid_logits = logits
            labels = end_points['labels'][0]  
            labels = labels.reshape(-1)
            valid_labels = labels
            loss += criterion(valid_logits, valid_labels)
            if cfg.flag_ohe2se:
                loss += compute_ohe2se_loss(logits, labels, cus_enc) * cfg.weight_ohe2se
        
        end_points['valid_logits'], end_points['valid_labels'] = valid_logits, valid_labels
    
    elif cfg.enc == 'se':
        loss = 0
        cus_enc = torch.tensor(cfg.cus_enc).to(device)
        if cfg.flag_ml:
            for i in range(len(end_points['logits'])):
                logits = end_points['logits'][i]
                logits = logits.transpose(1, 2).reshape(-1, 3)
                F.normalize(logits, dim=1)
                labels = end_points['labels'][-i - 1]  
                labels = labels.reshape(-1)
                enc_labels = cus_enc[labels[:], :]
                loss += torch.mean(torch.norm(logits - enc_labels, dim=-1), dim=0)
        else:
            logits = end_points['logits'][-1]
            logits = logits.transpose(1, 2).reshape(-1, 3)
            F.normalize(logits, dim=1)
            labels = end_points['labels'][0]
            labels = labels.reshape(-1)
            enc_labels = cus_enc[labels[:], :]
            loss += torch.mean(torch.norm(logits - enc_labels, dim=-1), dim=0)
        
        end_points['valid_logits'], end_points['valid_labels'] = logits, labels
    
    end_points['loss'] = loss
    
    return loss, end_points
