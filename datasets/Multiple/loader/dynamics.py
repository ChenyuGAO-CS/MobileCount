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
import random


class DynamicDataset(Dataset):
    def __init__(self,
                 couple_datasets,
                 mode,
                 main_transform=None, 
                 img_transform=None,
                 gt_transform=None,
                 image_size=None, 
                 **kwargs):
        """
            - couple_datasets : tuple or list of tuple, tuple for dataset class and
                                the dataset path, example : (CustomGCC, '/data/GCC') or 
                                [(CustomGCC, '/data/GCC'), (CustomSHH, '/data/SHHB')]
            - mode : str, dataset mode between 'train' and 'test'
            - **kwargs : keywords arguments, some datasets required arguments with: 
                - GCC : 
                    - GCC__gt_folder
                    - GCC__index_folder
                    - GCC__gt_format
                - SHH :
                    - SHHA__gt_name_folder
                    - SHHA__gt_format          
                    - SHHB__gt_name_folder
                    - SHHB__gt_format
            
            Optionnal:
            - img_transform : func, pytorch transform for image
            - gt_transform : func, pytorch transform for ground truth
            - main_transform : func, main pytorch transform
            - image_size : int, tuple or None. Resize the image with the shape
                           by a tupe, a int for a square or 
                           None for no action (default None) 
        """
        self.couple_datasets = couple_datasets if isinstance(
            couple_datasets, tuple) else [couple_datasets]
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
        for LoadClass, folder_dataset in couple_datasets:
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
    


class CollateFN:
    def __init__(self, train_size):
        self.TRAIN_SIZE = train_size
    
    def get_min_size(self, batch):
        "Find the min shape size in the batch"
        min_ht, min_wd = self.TRAIN_SIZE
        for i_sample in batch:
            _, ht, wd = i_sample.shape
            if ht < min_ht:
                min_ht = ht
            if wd < min_wd:
                min_wd = wd
        return min_ht, min_wd

    def random_crop(self, img, den, dst_size):
        """
        Get the random crop with the desirated size
        """
        _, ts_hd, ts_wd = img.shape
        x1 = random.randint(0, ts_wd - dst_size[1])
        y1 = random.randint(0, ts_hd - dst_size[0])
        x2 = x1 + dst_size[1]
        y2 = y1 + dst_size[0]

        den = den[y1:y2, x1:x2]
        img = img[:, y1:y2, x1:x2]
        return img, den

    def share_memory(self, batch):
        """ 
        If we're in a background process, concatenate directly into a
        shared memory tensor to avoid an extra copy
        """
        out = None
        if False:

            numel = sum([x.numel() for x in batch])
            storage = batch[0].storage()._new_shared(numel)
            out = batch[0].new(storage)
        return out

    def collate(self, batch):
        """
        Puts each data field into a tensor with outer dimension batch size
        """
        transposed = list(zip(*batch))
        imgs, dens = [transposed[0], transposed[1]]

        if isinstance(imgs[0], torch.Tensor) and isinstance(dens[0], torch.Tensor):
            min_ht, min_wd = self.get_min_size(imgs)
            cropped_imgs = []
            cropped_dens = []
            for i_sample in range(len(batch)):
                _img, _den = self.random_crop(img=imgs[i_sample],
                                         den=dens[i_sample],
                                         dst_size=[min_ht, min_wd])
                cropped_imgs.append(_img)
                cropped_dens.append(_den)

            cropped_imgs = torch.stack(cropped_imgs, 0, 
                                       out=self.share_memory(cropped_imgs))
            cropped_dens = torch.stack(cropped_dens, 0, 
                                       out=self.share_memory(cropped_dens))
            return [cropped_imgs, cropped_dens]
        raise TypeError(f"Batch must contain tensors, found: {type(imgs[0])} and {type(dens[1])}")
