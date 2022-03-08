import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

class CrowdCounter(nn.Module):
    def __init__(self, gpus, model_name, complete_cfg=None):
        super(CrowdCounter, self).__init__()        
        self.GPU_OK = torch.cuda.is_available()
        if model_name == 'MobileCount':
            from models.MobileCount import MobileCount as net
        elif model_name == 'MobileCountx1_25':
            from models.MobileCountx1_25 import MobileCount as net
        elif model_name == 'MobileCountx2':
            from models.MobileCountx2 import MobileCount as net
        self.complete_cfg = complete_cfg
        self.CCN = net()
        if self.GPU_OK:
            if len(gpus)>1:
                self.CCN = torch.nn.DataParallel(self.CCN, device_ids=gpus).cuda()
            else:
                self.CCN = self.CCN.cuda()
            self.loss_mse_fn = nn.MSELoss().cuda()
        else:
            self.loss_mse_fn = nn.MSELoss()
    
    
    
    def compute_lc_loss(self, output, target, sizes=(1, 2, 4)):
        l1_loss_reduction = 'mean'
        if isinstance(self.complete_cfg, dict) and 'l1_loss_reduction' in self.complete_cfg and self.complete_cfg['l1_loss_reduction'] in ['mean', 'sum']:
            l1_loss_reduction = self.complete_cfg['l1_loss_reduction']
            print('l1_loss_reduction:',l1_loss_reduction)
        criterion_L1 = torch.nn.L1Loss(reduction=l1_loss_reduction)
        if self.GPU_OK:
            criterion_L1 = criterion_L1.cuda()
        lc_loss = None
        for s in sizes:
            pool = torch.nn.AdaptiveAvgPool2d(s)
            if self.GPU_OK:
                pool = pool.cuda()
            est = pool(output.unsqueeze(0))
            gt = pool(target.unsqueeze(0))
            if lc_loss:
                lc_loss += criterion_L1(est, gt) / s**2
            else:
                lc_loss = criterion_L1(est, gt) / s**2
        return lc_loss
    
    @property
    def loss(self):
        return self.loss_mse

    def f_loss(self):
        return self.loss_mse
    
    def forward(self, img, gt_map=None):                               
        density_map = self.CCN(img)    
        if gt_map is not None : 
            self.loss_mse = self.build_loss(density_map.squeeze(), gt_map.squeeze())     

        return density_map
    
    def build_loss(self, density_map, gt_data, custom=True, lbda=10, sizes=(2, 4)):
        loss_mse = self.loss_mse_fn(density_map, gt_data)
        if isinstance(self.complete_cfg, dict) and 'custom_loss' in self.complete_cfg and self.complete_cfg['custom_loss'] in [True, False]:
            custom = self.complete_cfg['custom_loss']
            #print('custom_loss:',custom)
        self.lc_loss = 0
        if custom:
            if isinstance(self.complete_cfg, dict) and 'custom_loss_sizes' in self.complete_cfg and isinstance(self.complete_cfg['custom_loss_sizes'], list):
                sizes = self.complete_cfg['custom_loss_sizes']
                #print('custom_loss_sizes:',sizes) 
            lc_loss = self.compute_lc_loss(density_map, gt_data, sizes=sizes)
            self.lc_loss = lc_loss
            if isinstance(self.complete_cfg, dict) and 'custom_loss_lambda' in self.complete_cfg:
                lbda = self.complete_cfg['custom_loss_lambda']
                #print('custom_loss_lambda:',lbda)
            loss_mse = loss_mse + (lbda * lc_loss)
        return loss_mse

    def test_forward(self, img):                               
        density_map = self.CCN(img)                    
        return density_map

