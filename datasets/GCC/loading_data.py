import torchvision.transforms as standard_transforms
from torch.utils.data import DataLoader
import misc.transforms as own_transforms
from datasets.GCC.GCC import GCC
from datasets.GCC.setting import cfg_data
import torch
import random
import os


def loading_data():
    mean_std = cfg_data.MEAN_STD
    log_para = cfg_data.LOG_PARA
    
    if cfg_data.VAL_MODE=='rd':
        test_list = 'test_list.txt'
        train_list = 'train_list.txt'
        
    elif cfg_data.VAL_MODE=='cc':
        test_list = 'cross_camera_test_list.txt'
        train_list = 'cross_camera_train_list.txt'
        
    elif cfg_data.VAL_MODE=='cl':
        test_list = 'cross_location_test_list.txt'
        train_list = 'cross_location_train_list.txt'   
    
    train_main_transform = own_transforms.Compose([
        own_transforms.RandomHorizontallyFlip()
    ])
    img_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*mean_std)
    ])
    # removed
    gt_transform = standard_transforms.Compose([
        own_transforms.LabelNormalize(log_para)
    ])
    
    gt_transform2 = standard_transforms.Compose([
        standard_transforms.ToTensor()
    ])

    restore_transform = standard_transforms.Compose([
        own_transforms.DeNormalize(*mean_std),
        standard_transforms.ToPILImage()
    ])

 
    train_set = GCC(os.path.join(cfg_data.DATA_PATH, 'GCC' ,'txt_list', train_list), 
                    'train',
                    main_transform=train_main_transform, 
                    img_transform=img_transform, 
                    gt_transform=gt_transform2) # remove gt_transform
    
    train_loader = DataLoader(train_set, 
                              batch_size=cfg_data.TRAIN_BATCH_SIZE, 
                              num_workers=8, 
                              shuffle=True, 
                              drop_last=True)

    val_set = GCC(os.path.join(cfg_data.DATA_PATH, 'GCC', 'txt_list', test_list), 
                  'test',
                  main_transform=None, 
                  img_transform=img_transform, 
                  gt_transform=gt_transform2)
    
    val_loader = DataLoader(val_set, 
                            batch_size=cfg_data.VAL_BATCH_SIZE, 
                            num_workers=8, 
                            shuffle=True, 
                            drop_last=False)

    return train_loader, val_loader, restore_transform
