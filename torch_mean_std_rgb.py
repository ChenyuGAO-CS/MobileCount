import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as standard_transforms
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm

beginning_time = time.time()

# Test de recalcul des mean std des images de ShangaiTechA et B, change the list below and run the python script
tests_dictionary = {
    "SHHB": {"image_path_list": ["/workspace/data/shanghaiTech/part_B_final/train_data/images"],
             "mean_std": (
                 [0.452016860247, 0.447249650955, 0.431981861591], [0.23242045939, 0.224925786257, 0.221840232611]),
             # "image_size": (576,768),
             "image_size": (1024, 768),
             "recalculate": True
             },
    "SHHA": {"image_path_list": ["/workspace/data/shanghaiTech/part_A_final/train_data/images"],
             "mean_std": (
                 [0.410824894905, 0.370634973049, 0.359682112932], [0.278580576181, 0.26925137639, 0.27156367898]),
             # "image_size": (576,768),
             "image_size": (1024, 768),
             "recalculate": False
             },
    "GOLDEN": {"image_path_list": ["/workspace/cclabeler/images"],
               "cclabeler_filter": ["/workspace/cclabeler/users/golden.json"],  # Spécifique pour data de CCLabeler
               "mean_std": ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
               # "image_size": (576,768),
               "image_size": (1024, 768),
               "recalculate": False
               },
    "BACKGROUND": {"image_path_list": ["/workspace/cclabeler/images"],
                   "cclabeler_filter": ["/workspace/cclabeler/users/user4.json"],  # Spécifique pour data de CCLabeler
                   "mean_std": ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
                   # "image_size": (576,768),
                   "image_size": (1024, 768),
                   "recalculate": True
                   },
    "WE": {"image_path_list": ["/workspace/data/worldExpo10_blurred/train/img/"],
           "mean_std": (
               [0.504379212856, 0.510956227779, 0.505369007587], [0.22513884306, 0.225588873029, 0.22579960525]),
           # image_size": (512,672),
           "image_size": (1024, 768),
           "recalculate": False
           },
    "UHK": {"image_path_list": ["/workspace/data/CityUHK-X-BEV/images"],
            "mean_std": ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
            # "image_size": (576,768),
            "image_size": (1024, 768),
            "recalculate": False
            },
    "UCF50": {"image_path_list": ["/workspace/data/UCF-QNRF_ECCV18/Train"],
              "mean_std": (
                  [0.403584420681, 0.403584420681, 0.403584420681], [0.268462955952, 0.268462955952, 0.268462955952]),
              # "image_size": (576,768),
              "image_size": (1024, 768),
              "recalculate": False
              },
    "QNRF": {"image_path_list": ["/workspace/data/UCF-QNRF_ECCV18/Train"],
             "mean_std": (
                 [0.413525998592, 0.378520160913, 0.371616870165], [0.284849464893, 0.277046442032, 0.281509846449]),
             # "image_size": (576,768),
             "image_size": (1024, 768),
             "recalculate": False
             },
    "NWPU": {"image_path_list": ["/workspace/data/nwpu/images"],
             "mean_std": ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
             # "image_size": (576,768),
             "image_size": (1024, 768),
             "recalculate": False
             },
    "JHU": {"image_path_list": ["/workspace/data/jhu_crowd_v2.0/train/images"],
            "mean_std": ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
            # "image_size": (576,768),
            "image_size": (1024, 768),
            "recalculate": False
            },
    "GCC": {"image_path_list": ["/workspace/data"],
            "gcc_filter": ["/workspace/data/GCC/txt_list/train_list.txt"],
            "mean_std": (
                [0.302234709263, 0.291243076324, 0.269087553024], [0.227743327618, 0.211051672697, 0.184846073389]),
            # "image_size": (480,848),
            "image_size": (1024, 768),
            "recalculate": False
            },
    "SHHB+BACKGROUND": {"image_path_list": ["/workspace/data/shanghaiTech/part_B_final/train_data/images",
                                            "/workspace/cclabeler/images"],
                        "cclabeler_filter": [None,
                                             "/workspace/cclabeler/users/user4.json"],
                        # Spécifique pour data de CCLabeler
                        "mean_std": ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
                        # "image_size": (576,768),
                        "image_size": (1024, 768),
                        "recalculate": True
                        },
}

# The image should be preprocessed by torch.transform.ToTensor(), so the value is in [0,1] (all pixels values are divided by 255)
image_transform = standard_transforms.Compose([
    standard_transforms.ToTensor()
])


class CustomDataset(Dataset):

    def __init__(self, image_files, image_transform=None):
        self.image_files = image_files
        self.nb_images = len(self.image_files)
        self.image_transform = image_transform
        self.image_size = (1024, 768)

    def __len__(self):
        return self.nb_images

    def setImageSize(self, image_size):
        self.image_size = image_size

    def __getitem__(self, idx):
        img_file = self.image_files[idx]
        img = Image.open(os.path.join(img_file))
        img = img.convert('RGB')
        img = img.resize(self.image_size, Image.BILINEAR)  # identical to /misc/cal_mean.py
        if self.image_transform is not None:
            img = self.image_transform(img)
        return img


def get_mean_and_std_by_channel(loader):
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    for data in tqdm(loader):
        # Mean over batch, height and width, but not over the channels
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(data ** 2, dim=[0, 2, 3])
        num_batches += 1

    mean = channels_sum / num_batches

    # std = sqrt(E[X^2] - (E[X])^2)
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5
    return list(mean.numpy()), list(std.numpy())


def get_mean_and_std_by_channel2(loader):
    # Compute the mean and sd in an online fashion
    # Var[x] = E[X^2] - E^2[X]
    cnt = 0
    fst_moment = torch.empty(3)
    snd_moment = torch.empty(3)

    for images in tqdm(loader):
        b, c, h, w = images.shape
        nb_pixels = b * h * w
        sum_ = torch.sum(images, dim=[0, 2, 3])
        sum_of_square = torch.sum(images ** 2,
                                  dim=[0, 2, 3])
        fst_moment = (cnt * fst_moment + sum_) / (cnt + nb_pixels)
        snd_moment = (cnt * snd_moment + sum_of_square) / (cnt + nb_pixels)
        cnt += nb_pixels

    mean, std = fst_moment, torch.sqrt(snd_moment - fst_moment ** 2)
    return list(mean.numpy()), list(std.numpy())


if __name__ == '__main__':

    nb_images_total = 0
    records = []
    for dataset_name, record in tests_dictionary.items():

        if record["recalculate"]:

            start_time = time.time()

            print("\nDataset:", dataset_name)

            image_files = []

            if "cclabeler_filter" in record:
                cclabeler_filter = record["cclabeler_filter"]
            else:
                cclabeler_filter = []

            if "gcc_filter" in record:
                gcc_filter = record["gcc_filter"]
            else:
                gcc_filter = []

            for ix, ip in enumerate(record["image_path_list"]):

                if len(gcc_filter) > 0 and gcc_filter[ix] is not None:

                    with open(gcc_filter[ix]) as f:
                        lines = f.readlines()

                    for line in lines:

                        splited = line.strip().split()
                        file_folder = splited[3]
                        file_name = splited[4]

                        image_file_path = os.path.join(ip + file_folder + "/pngs/" + file_name + ".png")
                        if os.path.isfile(image_file_path):
                            image_files.append(image_file_path)
                else:
                    
                    images_filter = None
                    if len(cclabeler_filter) > 0 and cclabeler_filter[ix] is not None:
                        user_json_file = cclabeler_filter[ix]
                        with open(os.path.join(user_json_file)) as f:
                            userdata = json.load(f)
                            image_names = userdata['data']
                            images_filter = []
                            for image_name in image_names:
                                images_filter.append(image_name)

                    for filename in os.listdir(ip):
                        if os.path.isfile(os.path.join(ip, filename)) \
                                and os.path.splitext(filename)[1] in ['.jpg', '.jpeg', '.png'] \
                                and (images_filter is None or filename in images_filter):
                            try:
                                img = Image.open(os.path.join(ip, filename))
                                image_files += [os.path.join(ip, filename)]
                            except Exception as e:
                                print(os.path.join(ip, filename), str(e))

            print("Nombre d'images pour le dataset : {}".format(len(image_files)))

            dataset = CustomDataset(
                image_files=image_files,
                image_transform=image_transform)

            image_size = record["image_size"]
            dataset.setImageSize(image_size)

            loader = DataLoader(dataset, batch_size=6, num_workers=8)

            mean_std = get_mean_and_std_by_channel2(loader)

            print("Nombre d'images : {}".format(len(dataset)))
            nb_images_total += len(dataset)

            print("mean_std (recalcule) : {}".format(mean_std))
            print("mean_std (reference) : {}".format(record["mean_std"]))

            ratio_mean_std = (
                list(np.array(record["mean_std"][0]) / np.array(mean_std[0])),
                list(np.array(record["mean_std"][1]) / np.array(mean_std[1])))
            print('ratio_mean_std:', ratio_mean_std)

            end_time = time.time()
            calculation_time = round((end_time - start_time), 3)
            print("Temps de calcul : {} seconde(s)".format(calculation_time))
            nb_images_per_second = round(len(dataset) / (end_time - start_time), 3)
            print("Nombre d'images par seconde : ", nb_images_per_second)
            info3 = "Nombre d'images : {}".format(len(dataset))

            record = {
                "dataset": dataset_name,
                "nb_images": len(dataset),
                "calculation_time (s)": calculation_time,
                "nb_images/seconde": nb_images_per_second,
                "mean/std - recalculate": str((list(mean_std[0]), list(mean_std[1]))),
                "mean/std - reference": str((list(record["mean_std"][0]), list(record["mean_std"][1]))),
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

        xlsx_file = os.path.join("mean_std.xlsx")
        writer = pd.ExcelWriter(xlsx_file)
        results_df.to_excel(writer, 'results', index=False)
        writer.save()
