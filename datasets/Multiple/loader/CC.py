import pandas as pd
import os
import json
import pathlib
import numpy as np
import logging as lg
from PIL import Image
from scipy.sparse import load_npz
from .dynamics import CustomDataset


class CustomCCLabeler(CustomDataset):
    def __init__(self, folder, mode, **kwargs):
        super().__init__()
        # like '/workspace/cclabeler/users/golden.json' for Golden
        # or '/workspace/cclabeler/users/Background.json' or Background
        self.gt_index_filepath = kwargs.get('CC__index_filepath', None)
        
        if self.gt_index_filepath is None:
            raise ValueError('Must specify `CC__index_filepath` parameter')
        
        self.subset = ''
        if 'background' in self.gt_index_filepath:
            self.subset = 'background'
            self.gt_path = kwargs.get('BG__gt_path', None)
            self.gt_format = kwargs.get('BG__gt_format', '.json')
            self.transform = kwargs.get('BG__transform', None)
        elif 'golden' in self.gt_index_filepath:
            self.subset = 'golden'
            self.gt_path = kwargs.get('GD__gt_path', None)            
            self.transform = kwargs.get('GD__transform', None)
            self.gt_format = kwargs.get('GD__gt_format', '.json')
        else:
            raise NotImplementedError
        
        self.folder = folder
        self.mode = mode
        
        if self.mode == 'test':
            self.dataset = pd.DataFrame([])
            return
        self.dataset = self.read_index()

        
    def read_index(self):
        with open(self.gt_index_filepath, 'r', encoding='utf-8') as f:
            list_data = json.load(f)["data"]
        
        json_data = {}
        for n, im in enumerate(list_data):
            image_path = os.path.join(self.folder, 'images',  im)
            
            if self.gt_path is None:
                path_gt = os.path.join(self.folder, 'jsons', im + ".json")
            else:
                path_gt = os.path.join(self.gt_path, pathlib.Path(im).stem + '.npz')
            
            try:
                img = Image.open(image_path)
                json_data[n] = {"path_img": image_path,
                               "path_gt":  path_gt,
                               "gt_count": None,
                               "folder": self.folder}
            except Exception as e:
                lg.warning(f'Cannot read image: {image_path}, error: {str(e)}')
        df = pd.DataFrame.from_dict(json_data, orient='index')
        print(f'CustomCCLabeler - subset:{self.subset} - mode:{self.mode} - df.shape:{df.shape}')
        return df
    
    def load_gt(self, filename):
        density_map = None
        if self.gt_path is None:
            with open(filename, 'r') as f:
                js_gt = json.load(f)
                property_img = js_gt['properties']
                shape = (property_img["width"], 
                         property_img['height'])
                points = pd.DataFrame(js_gt['points']).to_numpy().astype(int)

            ds = np.zeros(shape)
            for x, y in points:
                try:
                    ds[x, y] += 1
                except Exception as e:
                    lg.warning('Point outside of the image point')
            density_map = ds.astype('uint8').T
            # if we want return PIL: Image.fromarray(den)
        else:
            density_map = load_npz(filename).toarray()
            
        self.check_density_map(density_map)
        return density_map