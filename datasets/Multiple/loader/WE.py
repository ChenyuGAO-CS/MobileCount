import pandas as pd
import glob
import os
import numpy as np
import logging as lg
import pathlib
from PIL import Image
from scipy.sparse import load_npz
from .dynamics import CustomDataset


class CustomWE(CustomDataset):
    def __init__(self, folder, mode, **kwargs):
        super().__init__()
        self.val_folder = ['104207','200608','200702','202201','500717']
        self.gt_format = kwargs.get('WE__gt_format', '.csv')
        self.transform = kwargs.get('WE__transform', None)
        self.folder = folder
        self.mode = mode
        self.dataset = self.read_index()
        
    def read_index(self):
        """
        Read all images position in WE Dataset
        """
        img_list = []
        if self.mode=='test':
            for folder in self.val_folder:
                img_list += list(glob.glob(os.path.join(self.folder, f'{self.mode}', folder, 'img', '*')))
        else:
            img_list += list(glob.glob(os.path.join(self.folder, f'{self.mode}', 'img', '*')))

        json_data = {}
        for n, im in enumerate(img_list):
            root_dir = pathlib.Path(im).parent.parent
            filename = pathlib.Path(im).stem
            path_gt = os.path.join(root_dir, 'den', filename + self.gt_format)
            gt_count = None
            json_data[n] = {
                            "path_img": im,
                            "path_gt": path_gt,
                            "gt_count": gt_count,
                            "folder": self.folder
                            }
        df = pd.DataFrame.from_dict(json_data, orient='index')
        print(f'CustomWE - mode:{self.mode} - df.shape:{df.shape}')
        return df
    
    def load_gt(self, filename):
        
        density_map = pd.read_csv(filename, sep=',',header=None).values
        density_map = density_map.astype(np.float32, copy=False)    
        
        self.check_density_map(density_map)
        return density_map
    
