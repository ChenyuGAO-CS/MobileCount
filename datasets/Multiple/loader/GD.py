import pandas as pd
import glob
import os
import numpy as np
import pathlib
from scipy.sparse import load_npz
from .dynamics import CustomDataset



class CustomGD(CustomDataset):
    def __init__(self, folder, mode, **kwargs):
        super().__init__()
        self.gt_index_filepath = kwargs.get('GD__index_filepath', None)
        # like: '/workspace/cclabeler/users/golden.json'
        self.gt_format = kwargs.get('GD__gt_format', '.json')
        if self.gt_index_filepath is None:
            raise ValueError('Must specify `GD__index_filepath` parameter')
        self.folder = folder
        self.mode = mode
        self.dataset = self.read_index()
        
    def read_index(self):
        with open(self.gt_index_filepath, 'r', encoding='utf-8') as f:
            list_data = json.load(f)["data"]
        
        json_data = {}
        for n, im in enumerate(list_data):
            json_data[n] = {"path_img": os.path.join(self.folder, 'images',  im),
                           "path_gt":  os.path.join(self.folder, 'jsons',  im + ".json"),
                           "gt_count": None,
                           "folder": self.folder}
        df = pd.DataFrame.from_dict(json_data, orient='index')
        return df
    
    def load_gt(self, filename, is_density_map=False):
        if not density_map:
            with open(filename, 'r') as f:
                js_gt = json.load(f)
                property_img = js_gt['properties']
                shape = (property_img["width"], 
                         property_img['height'])
                points = pd.DataFrame(js_gt['points']).to_numpy().astype(int)

            ds = np.zeros(shape)
            for x, y in points:
                ds[x, y] += 1
            return Image.fromarray(ds.astype('uint8'))
        else:
            raise NotImplementedError