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
        grid_train = {"fake_key":"fake_value"}
    print("\ngrid_train:",grid_train)
    
    grid_parameters = [p for p in grid_parameters(grid_train)]
    print("\nNombre d'entrainements :",len(grid_parameters))
    
    for parameters in grid_parameters:
        print("\nParameters :",parameters)
        
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
        #+ '_' + net_type \        
        EXP_NAME = now \
         + '_' + cfg.DATASET \
         + '_LR' + '%.e' % cfg.LR
        if cfg.L1_LOSS_REDUCTION=='mean':
            EXP_NAME += '_Rm'
        else: #sum
            EXP_NAME += '_Rs'
        if cfg.CUSTOM_LOSS:
            EXP_NAME += '_L' + str(cfg.CUSTOM_LOSS_LAMBDA)
            str_sizes = ','.join([str(i) for i in cfg.CUSTOM_LOSS_SIZES])
            EXP_NAME += '_S' + str_sizes
        else:
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
        elif data_mode == 'Multiple':
            from datasets.Multiple.loading_data import loading_data
            from datasets.Multiple.settings import cfg_data
            
        cfg_complete = cfg
        cfg_complete.update(cfg_data)
        cfg_complete.update(parameters)
        print("\ncfg_complete:",cfg_complete,"\n")
        
        #------------Prepare Trainer------------
        net = cfg.NET

        from trainer import Trainer

        cc_trainer = Trainer(loading_data, cfg_complete, pwd)
        cc_trainer.forward()
