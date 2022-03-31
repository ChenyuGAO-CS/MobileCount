import os
import time
from itertools import product

import numpy as np
import torch

from grid_config import cfg_grid


def grid_parameters(parameters):
    for params in product(*parameters.values()):
        yield dict(zip(parameters.keys(), params))


# ------------Grid Train------------
grid_train = cfg_grid.GRID_TRAIN

# ------------Start Training------------
pwd = os.path.split(os.path.realpath(__file__))[0]
if __name__ == '__main__':

    if not grid_train:
        grid_train = {"fake_key": "fake_value"}
    print("\ngrid_train:", grid_train)

    grid_parameters = [p for p in grid_parameters(grid_train)]
    print("\nNombre d'entrainements :", len(grid_parameters))

    for parameters in grid_parameters:
        print("\nParameters :", parameters)

        from config import cfg

        for cfg_key in cfg:
            if cfg_key in parameters:
                print("\nGeneral parameters overload - ", cfg_key, ':', parameters[cfg_key])
                cfg[cfg_key] = parameters[cfg_key]

        now = time.strftime("%d%m_%H%M", time.localtime())
        # net_type = 'MC'
        # if cfg.NET=='MobileCountx1_25':
        #    net_type = 'MC125'
        # elif cfg.NET=='MobileCountx2':
        #    net_type = 'MC2'
        # + '_' + net_type \
        # EXP_NAME = now + '_' + cfg.DATASET
        # EXP_NAME += '_LR' + '%.e' % cfg.LR
        # if cfg.L1_LOSS_REDUCTION=='mean':
        #    EXP_NAME += '_Rm'
        # else: #sum
        #    EXP_NAME += '_Rs'
        # if cfg.CUSTOM_LOSS:
        #    EXP_NAME += '_L' + str(cfg.CUSTOM_LOSS_LAMBDA)
        #    str_sizes = ','.join([str(i) for i in cfg.CUSTOM_LOSS_SIZES])
        #    EXP_NAME += '_S' + str_sizes
        # else:
        #    EXP_NAME += '_CLno'    
        # cfg['EXP_NAME'] = EXP_NAME.replace(' ','')
        # train_size = parameters['TRAIN_SIZE']
        # str_train_size = str(train_size[0]) + 'x' + str(train_size[1])
        cfg['EXP_NAME'] = now + 'WE_' + parameters['PATH_SETTINGS']['SHHB__transform_description']
        # cfg['EXP_NAME'] = 'SHHA+SHHB+WE+BACKGROUND'
        # ------------prepare environment------------
        seed = cfg.SEED
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)

        gpus = cfg.GPU_ID
        if len(gpus) == 1:
            torch.cuda.set_device(gpus[0])

        torch.backends.cudnn.benchmark = True

        # ------------prepare data loader------------
        data_mode = cfg.DATASET
        if data_mode == 'SHHA':
            from datasets.SHHA.setting import cfg_data
        elif data_mode == 'SHHB':
            from datasets.SHHB.setting import cfg_data
        elif data_mode == 'QNRF':
            from datasets.QNRF.setting import cfg_data
        elif data_mode == 'UCF50':
            from datasets.UCF50.setting import cfg_data
        elif data_mode == 'WE':
            from datasets.WE.setting import cfg_data
        elif data_mode == 'GCC':
            from datasets.GCC.setting import cfg_data
        elif data_mode == 'Multiple':
            from datasets.Multiple.settings import cfg_data
            
        for cfg_key in cfg_data:
            if cfg_key in parameters:
                if cfg_key!='PATH_SETTINGS':
                    print("\nDataset parameters overload - ", cfg_key, ':', parameters[cfg_key])
                    cfg_data[cfg_key] = parameters[cfg_key]
                else:
                    for path_settings_key in parameters['PATH_SETTINGS']:
                        print("\nPath settings overload - ", path_settings_key, ':',parameters['PATH_SETTINGS'][path_settings_key])
                        cfg_data['PATH_SETTINGS'][path_settings_key] = parameters['PATH_SETTINGS'][path_settings_key]

        if data_mode == 'SHHA':
            from datasets.SHHA.loading_data import loading_data
        elif data_mode == 'SHHB':
            from datasets.SHHB.loading_data import loading_data
        elif data_mode == 'QNRF':
            from datasets.QNRF.loading_data import loading_data
        elif data_mode == 'UCF50':
            from datasets.UCF50.loading_data import loading_data
        elif data_mode == 'WE':
            from datasets.WE.loading_data import loading_data
        elif data_mode == 'GCC':
            from datasets.GCC.loading_data import loading_data
        elif data_mode == 'Multiple':
            from datasets.Multiple.loading_data import loading_data
            
        cfg_complete = cfg
        cfg_complete.update(cfg_data)
        cfg_complete.update(parameters)
        print("\ncfg_complete:", cfg_complete, "\n")

        # ------------Prepare Trainer------------
        net = cfg.NET

        from trainer import Trainer

        #cc_trainer = Trainer(loading_data, cfg_complete, pwd)
        #cc_trainer.forward()
