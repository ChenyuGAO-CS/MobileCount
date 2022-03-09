from easydict import EasyDict as edict

# init
__C_SHHB = edict()

cfg_data = __C_SHHB

__C_SHHB.STD_SIZE = (768,1024)
# __C_SHHB.TRAIN_SIZE = (384,512)
__C_SHHB.TRAIN_SIZE = (576,768)

__C_SHHB.DATA_PATH = '/workspace/data/shanghaiTech/part_B_final'
#__C_SHHB.DATA_PATH = r"C:\\workspace\\data\\shanghaiTech\\part_B_final\\"

__C_SHHB.MEAN_STD = ([0.452016860247, 0.447249650955, 0.431981861591],[0.23242045939, 0.224925786257, 0.221840232611])

__C_SHHB.LABEL_FACTOR = 1
__C_SHHB.LOG_PARA = 2550.

# Negative value lead not to take in account those transforms
__C_SHHB.RANDOM_DOWNOVER_SAMPLING = -1 # New parameter to consecutivly down sample and over sample the image in order to simulate bad image encoding.
__C_SHHB.RANDOM_DOWN_SAMPLING = -1 # New parameter todown sample  the image

__C_SHHB.RESUME_MODEL = ''#model path
__C_SHHB.TRAIN_BATCH_SIZE = 6 #imgs

__C_SHHB.VAL_BATCH_SIZE = 1 # must be 1


