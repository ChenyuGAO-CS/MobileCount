import os
from easydict import EasyDict as edict

# init
__C = edict()
cfg_grid = __C

__C.GRID_TRAIN = config_overrides = {
    #To override a config parameter (cfg or cfg_data), use the same key
    #Example : config.py define 'cfg' which contain __C.MAX_EPOCH = 500, you must fix a sery of value like this "MAX_EPOCH":[100, 200, 300]
    "MAX_EPOCH": [100], # default value  = 500 in config.py (cfg)
    #"LR":[1e-4, 1e-5], # default value  = 1e-4 in config.py (cfg)
    #For specific parameters, not present in config files, look at the code Trainer.py or models/CC.py
    #"l1_loss_reduction":["mean", "sum"], # default value = "mean" 
    #if "custom_loss_sizes" or "custom_loss_lambda" exists, custom_loss=True, else otherwise
    "custom_loss_sizes": [(1, 2, 4), (2, 4), (2, 4, 8)],
    "custom_loss_lambda": [1000, 100, 10, 1, 0.1],
}
#__C.GRID_TRAIN = dict()
#================================================================================
#================================================================================
#================================================================================