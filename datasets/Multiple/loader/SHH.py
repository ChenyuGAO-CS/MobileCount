import pandas as pd
import glob
import numpy as np
import logging as lg
import os
import pathlib
from scipy.sparse import load_npz
from .dynamics import CustomDataset
import h5py

class CustomSHH(CustomDataset):
    def __init__(self, folder, mode, **kwargs):
        """
        Load Custom SHH
        """
        super().__init__()
        self.subset = ''
        if '_A_' in folder:
            self.subset = 'SHHA'
            self.gt_name_folder = kwargs.get('SHHA__gt_name_folder', 'maps_adaptive_kernel') 
            self.gt_format = kwargs.get('SHHA__gt_format', '.h5')
            self.transform = kwargs.get('SHHA__transform', None)
        elif '_B_' in folder:
            self.subset = 'SHHB'
            self.gt_name_folder = kwargs.get('SHHB__gt_name_folder', 'maps_fixed_kernel') 
            self.gt_format = kwargs.get('SHHB__gt_format', '.h5')
            self.transform = kwargs.get('SHHB__transform', None)
        else:
            raise ValueError('Choose a path with SHH part A or SHH part B')
        self.folder = folder
        self.mode = mode
        self.dataset = self.read_index()

    def read_index(self):
        """
        Read all images position in SSHB Dataset
        """
        img_list = list(filter(lambda x: '.txt' not in x and '.zip' not in x, 
                               glob.glob(os.path.join(self.folder, f'{self.mode}_data', 'images', '*'))))
        gt_folder = os.path.join(self.folder, f'{self.mode}_data', self.gt_name_folder)
        json_data = {}
        for n, im in enumerate(img_list):
            filename = pathlib.Path(im).stem
            gt_count = None
            json_data[n] = {
                            "path_img": im,
                            "path_gt": os.path.join(gt_folder, filename + self.gt_format),
                            "gt_count": gt_count,
                            "folder": self.folder
                            }
        df = pd.DataFrame.from_dict(json_data, orient='index')
        print(f'CustomSHH - subset:{self.subset} - mode:{self.mode} - df.shape:{df.shape}')        
        return df
            
    def load_gt(self, filename):
        """
        Load GT in np.array
        """
        density_map = None
        if pathlib.Path(filename).suffix == '.npz':
            density_map = load_npz(filename).toarray()
        elif pathlib.Path(filename).suffix == '.h5':
            gt_file = h5py.File(filename)
            density_map = np.asarray(gt_file['density'])
    
        self.check_density_map(density_map)
        return density_map