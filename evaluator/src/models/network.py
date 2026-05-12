import importlib
import torch
import torch.nn as nn

import src.models.pytorch_utils as pt_utils


class Network(nn.Module):

    def __init__(self, config):        
        super().__init__()
        
        self.config = config

        d_in = 8
        self.fc0 = pt_utils.Conv1d(self.config.num_raw_features, d_in, kernel_size=1, bn=True)
        
        '''
        --------LFA--------
        '''
        lib_lfa = importlib.import_module('src.models.lfa.lfa_{}'.format(self.config.lfa))
        self.lfas = nn.ModuleList()
        if self.config.rfa:
            lib_rfa = importlib.import_module('src.models.rfa.rfa_{}'.format(self.config.rfa))
            self.rfas = nn.ModuleList()
            
        for i in range(self.config.num_layers):
            d_out = self.config.d_out[i]
            self.lfas.append(lib_lfa.LFA(d_in, d_out, self.config))
            d_in = 2 * d_out
            if self.config.rfa:
                self.rfas.append(lib_rfa.RFA(d_in, d_in, self.config))
        '''
        --------LFA--------
        '''

        d_out = d_in
        
        '''
        --------GFA_S--------
        '''
        if self.config.gfa_s == False:
            self.decoder_0 = pt_utils.Conv2d(d_in, d_out, kernel_size=(1, 1), bn=True)
        else:
            lib_gfa_s = importlib.import_module('src.models.gfa_s.gfa_s_{}'.format(self.config.gfa_s))
            self.decoder_0 = lib_gfa_s.GFA(d_in, d_out, self.config)
        '''
        --------GFA_S--------
        '''

        self.decoder_blocks = nn.ModuleList()
        for j in range(self.config.num_layers):
            if j < self.config.num_layers - 1:
                d_in = d_out + 2 * self.config.d_out[-j - 2]
                d_out = 2 * self.config.d_out[-j - 2]
            else:
                d_in = 4 * self.config.d_out[-self.config.num_layers]
                d_out = 2 * self.config.d_out[-self.config.num_layers]
            self.decoder_blocks.append(pt_utils.Conv2d(d_in, d_out, kernel_size=(1, 1), bn=True))
        
        if self.config.flag_ml:
            self.fc1 = nn.ModuleList()
            self.fc2 = nn.ModuleList()
            self.dropout = nn.ModuleList()
            self.fc3 = nn.ModuleList()
            for j in range(self.config.num_layers):
                if j < self.config.num_layers - 1:
                    d_out = 2 * self.config.d_out[-j - 2]
                else:
                    d_out = 2 * self.config.d_out[-self.config.num_layers]
                self.fc1.append(pt_utils.Conv2d(d_out, 64, kernel_size=(1, 1), bn=True))
                self.fc2.append(pt_utils.Conv2d(64, 32, kernel_size=(1, 1), bn=True))
                self.dropout.append(nn.Dropout(0.5))
                if self.config.enc == 'ohe':
                    self.fc3.append(pt_utils.Conv2d(32, self.config.num_classes, kernel_size=(1, 1), activation=None))
                elif self.config.enc == 'se':
                    self.fc3.append(pt_utils.Conv2d(32, 3, kernel_size=(1, 1), activation=None))
        else:
            self.fc1 = pt_utils.Conv2d(d_out, 64, kernel_size=(1, 1), bn=True)
            self.fc2 = pt_utils.Conv2d(64, 32, kernel_size=(1, 1), bn=True)
            self.dropout = nn.Dropout(0.5)
            if self.config.enc == 'ohe':
                self.fc3 = pt_utils.Conv2d(32, self.config.num_classes, kernel_size=(1, 1), activation=None)
            elif self.config.enc == 'se':
                self.fc3 = pt_utils.Conv2d(32, 3, kernel_size=(1, 1), activation=None)
        
        '''
        --------GFA_L--------
        '''
        if self.config.gfa_l:
            lib_gfa_l = importlib.import_module('src.models.gfa_l.gfa_l_{}'.format(self.config.gfa_l))
            self.gfa_l = lib_gfa_l.GFA(d_out, d_out, self.config)
        '''
        --------GFA_L--------
        '''

    def forward(self, end_points):
        
        features = end_points['features']
        features = self.fc0(features)

        features = features.unsqueeze(dim=3)
        
        if self.config.flag_vis:
            end_points['fm_LFA'] = []
            end_points['fm_RFA'] = []
            for i in range(len(self.config.vis_layers)):
                end_points['fm_LFA'].append([])
                end_points['fm_RFA'].append([])
        
        '''
        --------Encoder--------
        '''
        f_encoder_list = []
        for i in range(self.config.num_layers):

            f_encoder_i = self.lfas[i](features, end_points['xyz'][i], end_points['neigh_idx'][i])
            
            if self.config.flag_vis:
                if i in self.config.vis_layers:
                    for vis_channel in self.config.vis_channels:
                        end_points['fm_LFA'][i].append(f_encoder_i[:, vis_channel, :, :])
            
            if self.config.rfa:
                f_encoder_i = self.rfas[i](f_encoder_i, end_points['xyz'][i], end_points['axis_neigh_idx'][i])
            
            if self.config.flag_vis:
                if i in self.config.vis_layers:
                    for vis_channel in self.config.vis_channels:
                        end_points['fm_RFA'][i].append(f_encoder_i[:, vis_channel, :, :])
            
            f_sampled_i = self.random_sample(f_encoder_i, end_points['sub_idx'][i])
            features = f_sampled_i
            if i == 0:
                f_encoder_list.append(f_encoder_i)
            f_encoder_list.append(f_sampled_i)
        '''
        --------Encoder--------
        '''

        features = self.decoder_0(f_encoder_list[-1])
        
        '''
        --------Dncoder--------
        '''
        f_decoder_list = []
        for j in range(self.config.num_layers):
            f_interp_i = self.nearest_interpolation(features, end_points['interp_idx'][-j - 1])
            f_decoder_i = self.decoder_blocks[j](torch.cat([f_encoder_list[-j - 2], f_interp_i], dim=1))
            f_decoder_list.append(f_decoder_i)
            features = f_decoder_i
        '''
        --------Dncoder--------
        '''
        
        end_points['logits'] = []
        
        if self.config.flag_ml:    
            for j in range(self.config.num_layers):
                f_temp = f_decoder_list[j]
                if j == self.config.num_layers - 1 and self.config.gfa_l:
                    f_temp = self.gfa_l(f_temp)
                f_temp = self.fc1[j](f_temp)
                f_temp = self.fc2[j](f_temp)
                f_temp = self.dropout[j](f_temp)
                f_temp = self.fc3[j](f_temp)
                f_temp = f_temp.squeeze(3)
                end_points['logits'].append(f_temp)
        else:
            if self.config.gfa_l:
                features = self.gfa_l(features)
            features = self.fc1(features)
            features = self.fc2(features)
            features = self.dropout(features)
            features = self.fc3(features)
            f_out = features.squeeze(3)
            end_points['logits'].append(f_out)
        
        return end_points

    @staticmethod
    def random_sample(feature, pool_idx):

        feature = feature.squeeze(dim=3)
        num_neigh = pool_idx.shape[-1]
        d = feature.shape[1]
        batch_size = pool_idx.shape[0]
        pool_idx = pool_idx.reshape(batch_size, -1)
        pool_features = torch.gather(feature, 2, pool_idx.unsqueeze(1).repeat(1, feature.shape[1], 1))
        pool_features = pool_features.reshape(batch_size, d, -1, num_neigh)
        pool_features = pool_features.max(dim=3, keepdim=True)[0]
        
        return pool_features

    @staticmethod
    def nearest_interpolation(feature, interp_idx):

        feature = feature.squeeze(dim=3)
        batch_size = interp_idx.shape[0]
        up_num_points = interp_idx.shape[1]
        interp_idx = interp_idx.reshape(batch_size, up_num_points)
        interpolated_features = torch.gather(feature, 2, interp_idx.unsqueeze(1).repeat(1, feature.shape[1], 1))
        interpolated_features = interpolated_features.unsqueeze(3)
        
        return interpolated_features