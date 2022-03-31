from easydict import EasyDict as edict

import misc.transforms as own_transforms

TRAIN_SIZE_SHHB = (576, 768)

main_transform_1 = own_transforms.Compose([
    own_transforms.RandomCrop(TRAIN_SIZE_SHHB),
    own_transforms.ColorJitter(0.3),
    own_transforms.RandomHorizontallyFlip()
])

main_transform_2 = own_transforms.Compose([
    own_transforms.RandomCrop(TRAIN_SIZE_SHHB),
    own_transforms.ColorJitter([0.1, 0.2, 0.3, 0.4, 0.5]),
    own_transforms.RandomHorizontallyFlip()
])

# init
__C = edict()
cfg_grid = __C

__C.GRID_TRAIN = {
    ##################################################################################################################
    # 1 - To override a general parameter define in config.py (== variable 'cfg')
    # Use ther same key, example : config.py define 'cfg' which contain __C.MAX_EPOCH = 500
    # You must fix a set of values like  "MAX_EPOCH":[100, 200, 300]
    ##################################################################################################################
    "RESUME": [False],
    #"RESUME_PATH": ['/workspace/share/iafoule/tensorboard/baseline/SHHB_multiple/best_state.pth'],
    # "RESUME_PATH": ['/workspace/home/jourdanfa/mobilecount_GCC_.pth'],
    "MAX_EPOCH": [500],  # default value  = 500 in config.py (cfg)
    "EXP_PATH": ["./exp"],
    # "LR": [1e-4],  # default value  = 1e-4 in config.py (cfg)
    # ATTENTION si CUSTOM_LOSS_LAMBDA ou CUSTOM_LOSS_SIZES sont définis alors CUSTOM_LOSS=True , False sinon
    # "CUSTOM_LOSS": [False],
    # "CUSTOM_LOSS_LAMBDA": [10],
    # "CUSTOM_LOSS_SIZES": [(2, 4)],
    ##################################################################################################################
    # 2 - To override a dataset parameter define in settings.py or loading_data.py (== variable 'cfg_data')
    # Use the same key, Example : settings.py define 'cfg_data' which contain __C.TRAIN_SIZE = (576,768)
    # Other example : loading_data.py define __C.PATH_SETTINGS = {'SHHB__transform': specific_transforms}
    # You must define specific_transforms here and a set of values like
    # "PATH_SETTINGS": [{'SHHB__transform': main_transform_1},{'SHHB__transform': main_transform_2}]
    ##################################################################################################################
    "DATASET": ['Multiple'],
    #"LIST_C_DATASETS": [[(CustomWE, '/workspace/data/worldExpo10_blurred')]],
    #"MEAN_STD": [([0.504379212856, 0.510956227779, 0.505369007587], [0.22513884306, 0.225588873029, 0.22579960525])],
    "PATH_SETTINGS": [
        {'WE__transform_description': 'rcrop+rds2+rdos2+hflip', 'WE__transform': main_transform_1},
        {'WE__transform_description': 'rds2+rdos2+rcrop+hflip', 'WE__transform': main_transform_2},
    ],
    "TRAIN_SIZE": [TRAIN_SIZE_SHHB],  # default value  = (576,768) like SHHB, WE = (512,672)
}

# Activate this line to avoid grid train (in this case, config.py will be used)
# __C.GRID_TRAIN = dict()

# ================================================================================
# ================================================================================
# ================================================================================
