import numpy as np
import os
import torch
from PIL import Image
from scipy import io as sio
from datasets.GCC.setting import cfg_data
from torch.utils import data
import pdb
import pandas as pd
from scipy.sparse import load_npz

class GCC(data.Dataset):
    def __init__(self, 
                 list_file, 
                 mode, 
                 main_transform=None, 
                 img_transform=None, 
                 gt_transform=None):

        self.crowd_level = []
        self.time = []
        self.weather = []
        self.file_folder = []
        self.file_name = []
        self.gt_cnt = []
        # self.den_folder_name = cfg.GT
        
        with open(list_file) as f:
            lines = f.readlines()
        
        for line in lines:
            splited = line.strip().split()

            self.crowd_level.append(splited[0])
            self.time.append(splited[1])
            self.weather.append(splited[2])
            self.file_folder.append(splited[3])
            self.file_name.append(splited[4])
            self.gt_cnt.append(int(splited[5]))

        self.mode = mode
        self.main_transform = main_transform  
        self.img_transform = img_transform
        self.gt_transform = gt_transform
        self.num_samples = len(lines)
        
    
    def __getitem__(self, index):
        img, den = self.read_image_and_gt(index)
      
        if self.main_transform is not None:
            img, den = self.main_transform(img, den) 
        
        if self.img_transform is not None:
            img = self.img_transform(img)
     
        if self.gt_transform is not None:
            den = self.gt_transform(den)
        
        if self.mode == 'train':    
            return img, den,
        
        elif self.mode == 'test':
            attributes_pt = np.array(
                [int(self.crowd_level[index]), 
                 int(self.time[index]), 
                 int(self.weather[index])]
                                            )
            return img, den, attributes_pt
        else:
            print('invalid data mode')

    def __len__(self):
        return self.num_samples

    
    def read_image_and_gt(self, index):
        img_path = os.path.join(cfg_data.DATA_PATH + self.file_folder[index],
                                'pngs', 
                                self.file_name[index] + '.png')

        den_map_path = os.path.join(cfg_data.GT_PATH,
                                    self.file_name[index] + '.npz')
        img = Image.open(img_path)
        den_array = self.load_sparse(den_map_path)
        den = den_array.astype(np.float32, copy=False)
        den = Image.fromarray(den)
        return img, den


    def get_num_samples(self):
        return self.num_samples       
    
    def load_sparse(self, filename):
        return load_npz(filename).toarray()