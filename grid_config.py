import os
from easydict import EasyDict as edict
from datasets.Multiple.loader import CustomGCC, CustomSHH

# init
__C = edict()
cfg_grid = __C

__C.GRID_TRAIN = {
    #To override a config parameter (cfg or cfg_data), use the same key
    #Example : config.py define 'cfg' which contain __C.MAX_EPOCH = 500, you must fix a sery of value like this "MAX_EPOCH":[100, 200, 300]
    "MAX_EPOCH": [500], # default value  = 500 in config.py (cfg)
    #"LR":[1e-4, 1e-5], # default value  = 1e-4 in config.py (cfg)
    "DATASET": ['Multiple'],
    "LIST_C_DATASETS": [[(CustomSHH, '/workspace/data/shanghaiTech/part_B_final/')]],
    "MEAN_STD": [([0.452016860247, 0.447249650955, 0.431981861591],[0.23242045939, 0.224925786257, 0.221840232611])],
    "PATH_SETTINGS": [{'GCC__gt_folder': '/workspace/home/gameiroth/data/GCC/density/maps_adaptive_kernel/',
                         'SHH__gt_name_folder': 'maps_fixed_kernel',
                         'CC__index_filepath' : '/workspace/cclabeler/users/background.json'}],   
    #ATTENTION si CUSTOM_LOSS_LAMBDA ou CUSTOM_LOSS_SIZES sont définis alors CUSTOM_LOSS=True , False sinon
    "CUSTOM_LOSS": [False],
    #"CUSTOM_LOSS_SIZES": [(1, 2, 4), (2, 4), (2, 4, 8)],
    #"CUSTOM_LOSS_LAMBDA": [1000, 100, 10, 1, 0.1],
}

#__C.GRID_TRAIN = dict()

#================================================================================
#================================================================================
#================================================================================