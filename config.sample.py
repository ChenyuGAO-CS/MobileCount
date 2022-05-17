import time

import torch
from easydict import EasyDict as edict

# init
__C = edict()
cfg = __C

# ------------------------------TRAIN------------------------
__C.SEED = 3035  # random seed,  for reporduction
__C.DATASET = 'Multiple'  # dataset selection: SHHA, SHHB, UCF50, QNRF, WE, JHU, Multiple

if __C.DATASET == 'UCF50':  # only for UCF50
    from datasets.UCF50.setting import cfg_data

    __C.VAL_INDEX = cfg_data.VAL_INDEX

if __C.DATASET == 'GCC':  # only for GCC
    from datasets.GCC.setting import cfg_data

    __C.VAL_MODE = cfg_data.VAL_MODE

__C.NET = 'MobileCount'  # net selection: MobileCount, MobileCountx1_25, MobileCountx2

__C.PRE_GCC = True  # use the pretrained model on GCC dataset
__C.PRE_GCC_MODEL = '/workspace/share/models/mobilecount_gcc_relu.pth'

__C.RESUME = False # contine training
__C.RESUME_PATH = './exp/04-25_09-19_SHHB_VGG_1e-05/latest_state.pth' #

if torch.cuda.is_available():
    __C.GPU_ID = [0] # sigle gpu: [0], [1] ...; multi gpus: [0,1]
else:
    __C.GPU_ID = []
# learning rate settings
__C.LR = 1e-4 # learning rate
__C.LR_DECAY = 0.995 # decay rate
__C.LR_DECAY_START = -1 # when training epoch is more than it, the learning rate will be begin to decay
__C.NUM_EPOCH_LR_DECAY = 1 # decay frequency
__C.MAX_EPOCH = 350

# print 
__C.PRINT_FREQ = 10

now = time.strftime("%m-%d_%H-%M", time.localtime())

__C.EXP_NAME = now \
               + '_' + __C.DATASET \
               + '_' + __C.NET \
               + '_' + str(__C.LR)

if __C.DATASET == 'UCF50':
    __C.EXP_NAME += '_' + str(__C.VAL_INDEX)

if __C.DATASET == 'GCC':
    __C.EXP_NAME += '_' + __C.VAL_MODE

__C.EXP_PATH = './exp'  # the path of logs, checkpoints, and current codes

# ------------------------------VAL------------------------
__C.VAL_DENSE_START = 1
__C.VAL_FREQ = 10  # Before __C.VAL_DENSE_START epoches, the freq is set as __C.VAL_FREQ

# ------------------------------VIS------------------------
__C.VISIBLE_NUM_IMGS = 1  # must be 1 for training images with the different sizes

# L1 loss reduction
__C.L1_LOSS_REDUCTION = "mean"

# Custom LOSS
__C.CUSTOM_LOSS = True
__C.CUSTOM_LOSS_LAMBDA = 10
__C.CUSTOM_LOSS_SIZES = (2, 4, 8)


# Infer on Golden dataset
__C.INFER_GOLDEN_DATASET = True
__C.STORE_MODEL_AFTER_GOLDEN_INFERENCE = True

__C.TEXTS2ADD = ''
# ================================================================================
# ================================================================================
# ================================================================================
