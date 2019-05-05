import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

class CrowdCounter(nn.Module):
    def __init__(self,gpus,model_name):
        super(CrowdCounter, self).__init__()        
        
        if model_name == 'VGG':
            from VGG import VGG as net
        elif model_name == 'VGG_DECODER':
            from VGG_decoder import VGG_decoder as net
        elif model_name == 'MCNN':
            from MCNN import MCNN as net
        elif model_name == 'LWRN':
            from gcy import ResNetLW as net
        elif model_name == 'DeepLabv3':
            from DeepLab_v3_2 import DeepLabv3 as net
        elif model_name =='MobLWRN':
            from LWRF_MobileNetv2 import MobileNetLWRF as net
        elif model_name =='RFN':
            from RefineNet import RefineNet as net
        elif model_name =='ShufLWRN':
            from LWRF_ShuffleNetv2 import ShuffleNetLWRF as net
        elif model_name =='CSRNet':
            from CSRNet import CSRNet as net
        elif model_name =='Res50':
            from Res50 import Res50 as net
        elif model_name =='Res101':
            from Res50 import Res101 as net
        elif model_name == 'Mobv2':
            from MobileNetv2 import MobileNetV2 as net
        elif model_name == 'FPN':
            from FPN import FPN as net

        self.CCN = net()
        if len(gpus)>1:
            self.CCN = torch.nn.DataParallel(self.CCN, device_ids=gpus).cuda()
        else:
            self.CCN=self.CCN.cuda()
        self.loss_mse_fn = nn.MSELoss().cuda()
        
    @property
    def loss(self):
        return self.loss_mse

    def f_loss(self):
        return self.loss_mse
    
    def forward(self, img, gt_map):                               
        density_map = self.CCN(img)                          
        self.loss_mse= self.build_loss(density_map.squeeze(), gt_map.squeeze())               
        return density_map
    
    def build_loss(self, density_map, gt_data):
        loss_mse = self.loss_mse_fn(density_map, gt_data)  
        return loss_mse

    def test_forward(self, img):                               
        density_map = self.CCN(img)                    
        return density_map

