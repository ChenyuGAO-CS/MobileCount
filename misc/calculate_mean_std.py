import sys

sys.path.append("../")
sys.path.append("../../ia-foule-lab")

import os
import time
import numpy as np
import pandas as pd
import torchvision.transforms as standard_transforms
import misc.transforms as own_transforms
from torch.utils.data import DataLoader

from datasets.Multiple.loader import DynamicDataset
from datasets.Multiple.loader import CustomCCLabeler, CustomGCC, CustomSHH
from datasets.Multiple.settings import cfg_data

from iafoule.utils import get_mean_and_std_by_channel

beginning_time = time.time()

# Test de recalcul des mean std des images de ShangaiTechA et B, change the list below and run the python script
output_directory = "/workspace/home/jourdanfa/data/"
tests_dictionary = {
    "SHHA": {"LIST_C_DATASETS": [(CustomSHH, '/workspace/data/shanghaiTech/part_A_final/')],
             "VAL_BATCH_SIZE": 1,
             "MEAN_STD_REFERENCE": (
                 [0.410824894905, 0.370634973049, 0.359682112932], [0.278580576181, 0.26925137639, 0.27156367898]),
             "RECALCULATE": True
             },
    "SHHB": {"LIST_C_DATASETS": [(CustomSHH, '/workspace/data/shanghaiTech/part_B_final/')],
             "VAL_BATCH_SIZE": 1,
             "MEAN_STD_REFERENCE": (
                 [0.452016860247, 0.447249650955, 0.431981861591], [0.23242045939, 0.224925786257, 0.221840232611]),
             "RECALCULATE": True
             },
    "GCC": {"LIST_C_DATASETS": [(CustomGCC, '/workspace/data/GCC')],
            "VAL_BATCH_SIZE": 1,
            "MEAN_STD_REFERENCE": (
                [0.302234709263, 0.291243076324, 0.269087553024], [0.227743327618, 0.211051672697, 0.184846073389]),
            "RECALCULATE": True
            },
    "BACKGROUND": {"LIST_C_DATASETS": [(CustomCCLabeler, '/workspace/cclabeler/')],
                   "PATH_SETTINGS": {'BG__index_filepath': '/workspace/cclabeler/users/background.json'},
                   "VAL_BATCH_SIZE": 1,
                   "MEAN_STD_REFERENCE": ([0.45974886, 0.46210667, 0.46128407], [0.2600742, 0.2610275, 0.28212664]),
                   "RECALCULATE": True
                   },
    "SHHA+SHHB+GCC+BACKGROUND": {"LIST_C_DATASETS": [(CustomSHH, '/workspace/data/shanghaiTech/part_A_final/'),
                                                     (CustomSHH, '/workspace/data/shanghaiTech/part_B_final/'),
                                                     (CustomGCC, '/workspace/data/GCC'),
                                                     (CustomCCLabeler, '/workspace/cclabeler/')],
                                 "PATH_SETTINGS": {'BG__index_filepath': '/workspace/cclabeler/users/background.json'},
                                 "VAL_BATCH_SIZE": 1,
                                 "MEAN_STD_REFERENCE": (
                                     [0.32202065, 0.311729, 0.29429936], [0.23350027, 0.21823275, 0.19834095]),
                                 "RECALCULATE": True
                                 },
}

# The image should be preprocessed by torch.transform.ToTensor(), so the value is in [0,1] (all pixels values are divided by 255)
img_transform = standard_transforms.Compose([
    standard_transforms.ToTensor()
])
gt_transform = standard_transforms.Compose([
    own_transforms.LabelNormalize(cfg_data.LOG_PARA)
])

if __name__ == '__main__':

    nb_images_total = 0
    records = []
    for dataset_name, record in tests_dictionary.items():

        if record["RECALCULATE"]:

            start_time = time.time()

            print("\nDataset:", dataset_name)

            path_settings = cfg_data.PATH_SETTINGS
            if "PATH_SETTINGS" in record:
                for key, value in record['PATH_SETTINGS'].items():
                    path_settings[key] = value

            val_set = DynamicDataset(couple_datasets=record['LIST_C_DATASETS'],
                                     mode='train',
                                     main_transform=None,
                                     img_transform=img_transform,
                                     gt_transform=gt_transform,
                                     image_size=(1024, 768),
                                     **path_settings)

            val_loader = DataLoader(val_set,
                                    batch_size=record['VAL_BATCH_SIZE'],
                                    num_workers=8,
                                    shuffle=True,
                                    drop_last=False)

            nb_images = len(val_set)
            print("Nombre d'images : {}".format(nb_images))

            mean_std = get_mean_and_std_by_channel(val_loader)

            nb_images_total += len(val_set)

            print("mean_std (recalcule) : {}".format(mean_std))
            print("mean_std (reference) : {}".format(record["MEAN_STD_REFERENCE"]))

            ratio_mean_std = (
                list(np.array(record["MEAN_STD_REFERENCE"][0]) / np.array(mean_std[0])),
                list(np.array(record["MEAN_STD_REFERENCE"][1]) / np.array(mean_std[1])))
            print('ratio_mean_std:', ratio_mean_std)

            end_time = time.time()
            calculation_time = round((end_time - start_time), 3)
            print("Temps de calcul : {} seconde(s)".format(calculation_time))
            nb_images_per_second = round(nb_images / (end_time - start_time), 3)
            print("Nombre d'images par seconde : ", nb_images_per_second)
            # info3 = "Nombre d'images : {}".format(len(dataset))

            record = {
                "dataset": dataset_name,
                "nb_images": nb_images,
                "calculation_time (s)": calculation_time,
                "nb_images/seconde": nb_images_per_second,
                "mean/std - recalculate": str((list(mean_std[0]), list(mean_std[1]))),
                "mean/std - reference": str(
                    (list(record["MEAN_STD_REFERENCE"][0]), list(record["MEAN_STD_REFERENCE"][1]))),
                "mean/std - ratio": str((list(ratio_mean_std[0]), list(ratio_mean_std[1])))
            }
            records.append(record)

    print("\nNombre d'images total : ", nb_images_total)
    end_time = time.time()
    calculation_time = round((end_time - beginning_time), 3)
    print("Temps de calcul total : ", calculation_time, "secondes")
    nb_images_per_second = round(nb_images_total / (end_time - beginning_time), 3)
    print("Nombre d'images par seconde : ", nb_images_per_second)

    record = {
        "dataset": "TOTAL",
        "nb_images": nb_images_total,
        "calculation_time (s)": calculation_time,
        "nb_images/seconde": nb_images_per_second,
        "mean/std - recalculate": "",
        "mean/std - reference": "",
        "mean/std - ratio": ""
    }
    records.append(record)

    if len(records):
        results_df = pd.DataFrame(records)

        results_df = results_df[['dataset', 'nb_images', 'calculation_time (s)', 'nb_images/seconde',
                                 'mean/std - recalculate', 'mean/std - reference', 'mean/std - ratio']]

        xlsx_file = os.path.join(output_directory, 'mean_std.xlsx')
        writer = pd.ExcelWriter(xlsx_file)
        results_df.to_excel(writer, 'results', index=False)
        writer.save()
