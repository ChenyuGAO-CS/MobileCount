from easydict import EasyDict as edict
from datasets.Multiple.loader import CustomGCC, CustomSHH, CustomCCLabeler


# init
__C_DYN = edict()

cfg_data = __C_DYN

__C_DYN.IMAGE_SIZE = None
__C_DYN.TRAIN_SIZE = (1000, 800)
__C_DYN.LIST_C_DATASETS = [(CustomGCC, '/workspace/data/GCC'), 
                          (CustomSHH, '/workspace/data/shanghaiTech/part_A_final/'), 
                          (CustomSHH, '/workspace/data/shanghaiTech/part_B_final/'),
                          (CustomCCLabeler,  '/workspace/cclabeler/')]
__C_DYN.MEAN_STD = ([1, 1, 1], 
                    [1, 1, 1])
#__C_DYN.PROB = [0.2, 0.4, 0.4] # proba getting images
__C_DYN.COLLATE_FN = True
# better to remove because use in collate but no effect
#__C_DYN.LABEL_FACTOR = 1 
__C_DYN.LOG_PARA = 2550.

# Negative value lead not to take in account those transforms
__C_DYN.RANDOM_DOWNOVER_SAMPLING = -1
__C_DYN.RANDOM_DOWN_SAMPLING = -1

__C_DYN.RESUME_MODEL = '/data/models'
__C_DYN.TRAIN_BATCH_SIZE = 3
__C_DYN.VAL_BATCH_SIZE = 1
__C_DYN.PATH_SETTINGS = {'GCC__gt_folder': '/workspace/home/gameiroth/data/GCC/density/maps_adaptive_kernel/',
                         'SHH__gt_name_folder': 'maps_fixed_kernel',
                         'CC__index_filepath' : '/workspace/cclabeler/users/background.json'}


#- GCC : 
#    - GCC__gt_folder
#    - GCC__index_folder
#    - GCC__gt_format
# CC : 
#     required : CC__index_folder
#     optionnal : 
#        - BG__gt_format
#        - GD__gt_format
#- SHH : 
#    - SHHA__gt_name_folder 
#    - SHHA__gt_format          
#    - SHHB__gt_name_folder
#    - SHHB__gt_format
# NOTE: ds gt folder must be in shh train / test folder (variable is the name of folder)
