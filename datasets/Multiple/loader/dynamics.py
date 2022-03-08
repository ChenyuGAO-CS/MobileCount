import torch
import numpy as np
import pandas as pd
import os
import pathlib
from torch.utils.data import Dataset
from PIL import Image
from scipy import io as sio
from torch.utils import data
from scipy.sparse import load_npz
import glob 


class DynamicDataset(Dataset):
    def __init__(self,
                 folder_datasets,
                 class_datasets,
                 mode,
                 main_transform=None, 
                 img_transform=None,
                 gt_transform=None,
                 image_size=None, 
                 **kwargs):
        """
            - folder_datasets : str or list, path list of root data folder
            - class_datasets : ClassDataset or list, Class Dataset to use with data folder
            - img_transform : func, pytorch transform for image
            - gt_transform : func, pytorch transform for ground truth
            - main_transform : func, main pytorch transform
            - image_size : int, tuple or None. Resize the image with the shape
                           by a tupe, a int for a square or 
                           None for no action (default None)
            - mode : str, dataset mode between 'train' and 'test'
            - **kwargs : keywords arguments with : 
                - GCC : 
                    - GCC__gt_folder
                    - GCC__index_folder
                    - GCC__gt_format
                - SHH :
                    - SHH__gt_name_folder
                    - SHH__gt_format
                
        Limitation: - must use the same params if call twice a CustomClass
        """
        self.folder_datasets = folder_datasets if isinstance(
            folder_datasets, list) else [folder_datasets]
        self.class_datasets = class_datasets if isinstance(
            class_datasets, list) else [class_datasets]
        self.img_transform = img_transform
        self.gt_transform = gt_transform
        self.main_transform = main_transform
        self.image_size = image_size
        self.mode = mode
        self.kwargs = kwargs
        self.dataset = pd.DataFrame([])
        self.read_dict = {}
        self.parse_dataset()

    def __len__(self):
        return len(self.dataset)
    
    def resize(self, img):
        return img.resize(self.image_size, Image.BILINEAR)
    
    def __getitem__(self, index):
        row = self.dataset.loc[index]
        dataset_func = self.read_dict[row.folder]
        img, den = dataset_func['img'](row.path_img), dataset_func['gt'](row.path_gt)
        if self.image_size is not None:
            img, den = self.resize(img), self.resize(den)
        img, den = self.transform_img(img, den)
        return img, den
    
    def transform_img(self, img, den):
        if self.main_transform is not None:
            img, den = self.main_transform(img, den) 
        if self.img_transform is not None:
            img = self.img_transform(img)         
        if self.gt_transform is not None:
            den = self.gt_transform(den)
        return img, den
    
    def parse_dataset(self):
        for folder_dataset, LoadClass in zip(self.folder_datasets, self.class_datasets):
            loader = LoadClass(folder_dataset, self.mode, **self.kwargs)
            self.dataset = pd.concat((self.dataset, loader.dataset), axis=0)
            self.read_dict[folder_dataset] = {"gt": loader.read_gt,
                                              "img": loader.read_image}
        self.dataset = self.dataset.reset_index(drop=True)


class CustomDataset:
    """
    Main class for reading Custom Datasets
    """
    def __init__(self):
        self.mode = None
        self.dataset = None
        
    def read_index(self):
        pass
    
    def read_image(self, img_path):
        """
        Read an image and return Pillow Image
        """
        img = Image.open(img_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        return img
    
    def read_gt(self, gt_path):
        """
        Read GT and return Pillow Image
        """
        den_array = self.load_gt(gt_path)
        den = den_array.astype(np.float32, copy=False)
        den = Image.fromarray(den)
        return den
    
    def load_gt(self, gt_path):
        """
        Loader of density map
        """
        pass
    
    def __repr__(self):
        return f'{type(self).__name__} in mode {self.mode}'
    
    def __len__(self):
        return len(self.dataset)