import os
import sys
import time
import numpy as np
from itertools import product
import torch
from grid_config import cfg_grid

def grid_parameters(parameters):
    for params in product(*parameters.values()):
        yield dict(zip(parameters.keys(), params))

#------------Grid Train------------
grid_train = cfg_grid.GRID_TRAIN
        
#------------Start Training------------
pwd = os.path.split(os.path.realpath(__file__))[0]
if __name__ == '__main__':
    
    if not grid_train:
        grid_train = {"only_one":""}
    print("grid_train:",grid_train)
    
    grid_parameters = [p for p in grid_parameters(grid_train)]
    print("Nombre d'entrainements :",len(grid_parameters))
    
    for parameters in grid_parameters:
        print("Parameters :",parameters)
        
        from config import cfg
        for cfg_key in cfg:
            if cfg_key in parameters:
                cfg[cfg_key] = parameters[cfg_key]

        now = time.strftime("%d%m_%H%M", time.localtime())
        #net_type = 'MC'
        #if cfg.NET=='MobileCountx1_25':
        #    net_type = 'MC125'
        #elif cfg.NET=='MobileCountx2':
        #    net_type = 'MC2'
        str_lr = '%.e' % cfg.LR
        #+ '_' + net_type \        
        EXP_NAME = now \
         + '_' + cfg.DATASET \
         + '_LR' + str_lr
        if 'l1_loss_reduction' in parameters:
            if parameters['l1_loss_reduction']=='mean':
                EXP_NAME += '_Rm'
            else: #sum
                EXP_NAME += '_Rs'
        if 'custom_loss_lambda' in parameters or 'custom_loss_sizes' in parameters:
            parameters['custom_loss'] = True
            if 'custom_loss_lambda' in parameters:
                EXP_NAME += '_L' + str(parameters['custom_loss_lambda'])
            if 'custom_loss_sizes' in parameters:
                str_sizes = ','.join([str(i) for i in parameters['custom_loss_sizes']])
                EXP_NAME += '_S' + str_sizes
        else:
            parameters['custom_loss'] = False
            EXP_NAME += '_CLno'    
        cfg['EXP_NAME'] = EXP_NAME.replace(' ','')
                
        #------------prepare environment------------
        seed = cfg.SEED
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)

        gpus = cfg.GPU_ID
        if len(gpus) == 1:
            torch.cuda.set_device(gpus[0])

        torch.backends.cudnn.benchmark = True

        #------------prepare data loader------------
        data_mode = cfg.DATASET
        if data_mode == 'SHHA':
            from datasets.SHHA.loading_data import loading_data 
            from datasets.SHHA.setting import cfg_data 
        elif data_mode == 'SHHB':
            from datasets.SHHB.loading_data import loading_data 
            from datasets.SHHB.setting import cfg_data 
        elif data_mode == 'QNRF':
            from datasets.QNRF.loading_data import loading_data 
            from datasets.QNRF.setting import cfg_data 
        elif data_mode == 'UCF50':
            from datasets.UCF50.loading_data import loading_data 
            from datasets.UCF50.setting import cfg_data 
        elif data_mode == 'WE':
            from datasets.WE.loading_data import loading_data 
            from datasets.WE.setting import cfg_data 
        elif data_mode == 'GCC':
            from datasets.GCC.loading_data import loading_data
            from datasets.GCC.setting import cfg_data 
            
        complete_cfg = cfg
        complete_cfg.update(cfg_data)
        complete_cfg.update(parameters)
        print("complete_cfg:",complete_cfg)
        
        #------------Prepare Trainer------------
        net = cfg.NET

        from trainer import Trainer

        cc_trainer = Trainer(loading_data, cfg_data, pwd, complete_cfg=complete_cfg)
        cc_trainer.forward()
