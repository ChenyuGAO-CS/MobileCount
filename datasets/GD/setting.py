from easydict import EasyDict as edict

# init
__C_GD = edict()

cfg_data = __C_GD


__C_GD.DATA_PATH = '/workspace/cclabeler'

__C_GD.MEAN_STD = ([0.452016860247, 0.447249650955, 0.431981861591],[0.23242045939, 0.224925786257, 0.221840232611])

__C_GD.VAL_BATCH_SIZE = 1 # must be 1


