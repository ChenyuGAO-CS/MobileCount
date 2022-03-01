import torchvision.transforms as standard_transforms
from torch.utils.data import DataLoader
# from misc.data import DataLoader
import misc.transforms as own_transforms
from .GD import GD
from .setting import cfg_data 
import torch


def loading_data():
    mean_std = cfg_data.MEAN_STD
 
    img_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*mean_std)
    ])

    val_set = GD(cfg_data.DATA_PATH, img_transform=img_transform)
    val_loader = DataLoader(val_set, batch_size=cfg_data.VAL_BATCH_SIZE, num_workers=8, shuffle=False, drop_last=False)

    return val_loader