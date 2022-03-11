import pandas as pd
import glob
import os
import json
import numpy as np
import pathlib
from PIL import Image
from .dynamics import CustomDataset



class CustomCCLabeler(CustomDataset):
    def __init__(self, folder, mode, **kwargs):
        super().__init__()
        # like: '/workspace/cclabeler/users/golden.json' for Golden
        # or '/workspace/cclabeler/users/user4.json' or Background
        if mode == 'train':
            self.gt_format = kwargs.get('BG__gt_format', '.json')
            self.gt_index_filepath = kwargs.get('BG__index_filepath', None)
            self.transform = kwargs.get('BG__transform', None)
        else:
            self.gt_index_filepath = kwargs.get('GD__index_filepath', None)
            self.transform = kwargs.get('GD__transform', None)
            self.gt_format = kwargs.get('GD__gt_format', '.json')
            
        if self.gt_index_filepath is None:
            raise ValueError('Must specify `<BG or GD>__index_filepath` parameter')
        self.folder = folder
        self.mode = mode
        self.dataset = self.read_index()
        
    def read_index(self):
        with open(self.gt_index_filepath, 'r', encoding='utf-8') as f:
            list_data = json.load(f)["data"]
        
        json_data = {}
        for n, im in enumerate(list_data):
            image_path = os.path.join(self.folder, 'images',  im)
            try:
                img = Image.open(image_path)
                json_data[n] = {"path_img": image_path,
                               "path_gt":  os.path.join(self.folder, 'jsons' + "/" + im + ".json"),
                               "gt_count": None,
                               "folder": self.folder}
            except Exception as e:
                print('Cannot read image :',image_path, '- error :',str(e))

        df = pd.DataFrame.from_dict(json_data, orient='index')
        return df
    
    def load_gt(self, filename, is_density_map=False):
        if not is_density_map:
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
                    print('Point outside of the image point ({},{}) - shape {} filename: {}'.format(x,y,shape,filename))
            den = ds.astype('uint8').T
            return den
            # if we want return PIL: Image.fromarray(den)
        else:
            raise NotImplementedError
