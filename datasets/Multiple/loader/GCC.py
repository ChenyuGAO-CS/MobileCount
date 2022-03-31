import pandas as pd
import glob
import os
import numpy as np
import logging as lg
import pathlib
from scipy.sparse import load_npz
from .dynamics import CustomDataset


class CustomGCC(CustomDataset):
    def __init__(self, folder, mode, **kwargs):
        super().__init__()
        self.gt_folder = kwargs.get('GCC__gt_folder')
        self.gt_index_folder = kwargs.get('GCC__index_folder', 'txt_list')
        self.gt_format = kwargs.get('GCC__gt_format', '.npz')
        self.transform = kwargs.get('GCC__transform', None)
        if self.gt_folder is None:
            raise ValueError('Must specify `GCC__gt_folder` parameter')
        self.folder = folder
        self.mode = mode
        self.dataset = self.read_index()
        
    def read_index(self):
        index_file = os.path.join(self.folder, self.gt_index_folder, f'{self.mode}_list.txt')
        with open(index_file) as f:
            lines = f.readlines()
        json_data = {}
        for n, line in enumerate(lines):
            crowd_level, time, weather, file_folder, filename, gt_count = line.strip().split()
            json_data[n] = {
                            "path_img": os.path.join(self.folder, 
                                                     pathlib.Path(file_folder).stem, 
                                                     'pngs', filename + '.png'),
                           "path_gt": os.path.join(self.gt_folder, filename) + self.gt_format,
                           "gt_count": gt_count,
                           "folder": self.folder
                            }
        df = pd.DataFrame.from_dict(json_data, orient='index')
        print(f'CustomGCC - mode:{self.mode} - df.shape:{df.shape}')
        return df
    
    def load_gt(self, filename):
        density_map = load_npz(filename).toarray()
        self.check_density_map(density_map)
        return density_map