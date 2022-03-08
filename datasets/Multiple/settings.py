from easydict import EasyDict as edict
from loader import CustomGCC, CustomSHH


# init
__C_DYN = edict()

cfg_data = __C_DYN

__C_DYN.IMAGE_SIZE = None
__C_DYN.LIST_DATA_PATH = ['/workspace/data/GCC', 
                          '/workspace/data/shanghaiTech/part_A_final/', 
                          '/workspace/data/shanghaiTech/part_B_final/']
__C_DYN.LIST_CLASSES = [CustomGCC, CustomSHH, CustomSHH]
__C_DYN.LIST_PATH = '.'
__C_DYN.MEAN_STD = ([0.302234709263, 0.291243076324, 0.269087553024], 
                    [0.227743327618, 0.211051672697, 0.184846073389])
__C_DYN.LABEL_FACTOR = 1
__C_DYN.LOG_PARA = 2550.

__C_DYN.RESUME_MODEL = '/data/models'
__C_DYN.TRAIN_BATCH_SIZE = 1
__C_DYN.VAL_BATCH_SIZE = 1
__C_DYN.PATH_SETTINGS = {'GCC__gt_folder': '/workspace/home/***/data/GCC/density/maps_adaptive_kernel/',
                         'SHH__gt_name_folder': 'maps_fixed_kernel'} 

# other parameters to classes
# GCC : 
#    - GCC__gt_folder
#    - GCC__index_folder
#    - GCC__gt_format
# SHH :
#    - SHH__gt_name_folder
#    - SHH__gt_format