import sys
sys.path.append("../")

import os
import time
from tqdm import tqdm
import torchvision.transforms as standard_transforms
from torch.utils.data import DataLoader

from datasets.Multiple.loader import DynamicDataset
from datasets.Multiple.loader import CustomGCC, CustomSHH, CustomCC

beginning_time = time.time()

PATH_SETTINGS = {'GCC__gt_folder': '/workspace/home/gameiroth/data/GCC/density/maps_adaptive_kernel/',
                 'SHH__gt_name_folder': 'maps_fixed_kernel',
                 'BG__index_folder' : '/workspace/cclabeler/users/background.json'}

# Test de recalcul des mean std des images de ShangaiTechA et B, change the list below and run the python script
tests_dictionary = {
    "GCC+SHHA+SHHB": {"LIST_C_DATASETS": [(CustomGCC, '/workspace/data/GCC'), 
                          (CustomSHH, '/workspace/data/shanghaiTech/part_A_final/'), 
                          (CustomSHH, '/workspace/data/shanghaiTech/part_B_final/')
                         ],
             "VAL_BATCH_SIZE": 1,                      
             "MEAN_STD_REFERENCE": (
                 [0.452016860247, 0.447249650955, 0.431981861591], [0.23242045939, 0.224925786257, 0.221840232611]),
             "RECALCULATE": True
             },
    "SHHA+SHHB": {"LIST_C_DATASETS": [(CustomSHH, '/workspace/data/shanghaiTech/part_A_final/'), 
                                 (CustomSHH, '/workspace/data/shanghaiTech/part_B_final/')
                         ],
             "VAL_BATCH_SIZE": 1,  
             "MEAN_STD_REFERENCE": (
                 [0.410824894905, 0.370634973049, 0.359682112932], [0.278580576181, 0.26925137639, 0.27156367898]),
             "RECALCULATE": True
             },
}
    
# The image should be preprocessed by torch.transform.ToTensor(), so the value is in [0,1] (all pixels values are divided by 255)
img_transform = standard_transforms.Compose([
    standard_transforms.ToTensor()
])


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

        if record["RECALCULATE"]:

            start_time = time.time()

            print("\nDataset:", dataset_name)

            val_set = DynamicDataset(couple_datasets=record['LIST_C_DATASETS'],
                                     mode='test',
                                     main_transform=None,
                                     img_transform=img_transform,
                                     gt_transform=None,
                                     image_size=None,
                                     **PATH_SETTINGS)

            val_loader = DataLoader(val_set, 
                                    batch_size=record['VAL_BATCH_SIZE'], 
                                    num_workers=8, 
                                    shuffle=False, 
                                    drop_last=False)

            print("Nombre d'images pour le dataset : {}".format(len(val_set)))

            mean_std = get_mean_and_std_by_channel(val_loader)

            print("Nombre d'images : {}".format(len(dataset)))
            nb_images_total += len(dataset)

            print("mean_std (recalcule) : {}".format(mean_std))
            print("mean_std (reference) : {}".format(record["MEAN_STD_REFERENCE"]))

            ratio_mean_std = (
                list(np.array(record["MEAN_STD_REFERENCE"][0]) / np.array(mean_std[0])),
                list(np.array(record["MEAN_STD_REFERENCE"][1]) / np.array(mean_std[1])))
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


