import numpy as np
import os, json
import random
from scipy import io as sio
import sys
import torch
from torch.utils import data
from PIL import Image, ImageOps
import h5py
#import pandas as pd

from config import cfg

class GD(data.Dataset):
    def __init__(self, data_path, img_transform=None):
        user_json = os.path.join(data_path,'users' ,'golden.json')

        with open(user_json) as f:

            userdata = json.load(f)
            image_names = userdata['data']

        self.img_path = data_path + '/images'
        self.json_path = data_path + '/jsons'
            
        self.data_files = image_names
        self.num_samples = len(self.data_files) 
        self.img_transform = img_transform

    
    def __getitem__(self, index):
        fname = self.data_files[index]
        img, den, points = self.read_image_and_gt(fname)
        if self.img_transform is not None:
            img = self.img_transform(img)         
          
        return img, den, points

    def __len__(self):
        return self.num_samples

    def read_image_and_gt(self,fname):
        img = Image.open(os.path.join(self.img_path,fname))
        if img.mode in ['L', 'RGBA', 'P']:
            img = img.convert('RGB')


        # den = sio.loadmat(os.path.join(self.gt_path,os.path.splitext(fname)[0] + '.mat'))
        # den = den['map']
        #den = pd.read_csv(os.path.join(self.gt_path,os.path.splitext(fname)[0] + '.csv'), sep=',',header=None).values
        with open(os.path.join(self.json_path,fname + '.json')) as f:
                    js = json.load(f)
                    count = js['human_num']
                    points = js['points']
        
        return img, count, points

    def get_num_samples(self):
        return self.num_samples       
            
        