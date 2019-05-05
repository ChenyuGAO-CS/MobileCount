from matplotlib import pyplot as plt
import os
import random
import torch
from torch.autograd import Variable
import torchvision.transforms as standard_transforms
import misc.transforms as own_transforms
import pandas as pd

from models.ptflops import get_model_complexity_info
from models.CC import CrowdCounter
from config import cfg
from misc.utils import *
import scipy.io as sio
from PIL import Image, ImageOps

import time
import pdb

torch.cuda.set_device(0)
torch.backends.cudnn.benchmark = True

exp_name = './DULR-display-save-mat'
if not os.path.exists(exp_name):
    os.mkdir(exp_name)

mean_std = ([0.452016860247, 0.447249650955, 0.431981861591],[0.23242045939, 0.224925786257, 0.221840232611])

img_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*mean_std)
    ])
restore = standard_transforms.Compose([
        own_transforms.DeNormalize(*mean_std),
        standard_transforms.ToPILImage()
    ])
pil_to_tensor = standard_transforms.ToTensor()

# dataRoot = './exp/data/shanghaitech_part_B/test'
# model_path = './exp/6.23G_3M_SHHB_MobLWRN_0.0001/all_ep_477_mae_9.0_mse_15.4.pth'

# dataRoot = './exp/data/shanghaitech_part_A/test'
# model_path = './exp/04-23_00-06_SHHA_MobLWRN_0.0001/all_ep_438_mae_89.4_mse_146.0.pth'

dataRoot = './exp/data/UCF-QNRF-1024x1024-mod16/test'
model_path = './exp/04-23_00-08_QNRF_MobLWRN_0.0001/all_ep_448_mae_131.1_mse_222.6.pth'

def main():
    # file_list = [filename for filename in os.listdir(dataRoot+'/img/') if os.path.isfile(os.path.join(dataRoot+'/img/',filename))]
    file_list = [filename for root,dirs,filename in os.walk(dataRoot+'/img/')]

    test(file_list[0], model_path)
   

def test(file_list, model_path):
    net = CrowdCounter(cfg.GPU_ID, cfg.NET)
    net.load_state_dict(torch.load(model_path))
    net.cuda()
    net.eval()

    step = 0
    for filename in file_list:
        step = step + 1
    	print filename
        imgname = dataRoot + '/img/' + filename
        filename_no_ext = filename.split('.')[0]

        denname = dataRoot + '/den/' + filename_no_ext + '.csv'

        den = pd.read_csv(denname, sep=',',header=None).values
        den = den.astype(np.float32, copy=False)

        img = Image.open(imgname)

        if img.mode == 'L':
            img = img.convert('RGB')

        # prepare
        wd_1, ht_1 = img.size
        # pdb.set_trace()

        # if wd_1 < 1024:
        #     dif = 1024 - wd_1
        #     img = ImageOps.expand(img, border=(0,0,dif,0), fill=0)
        #     pad = np.zeros([ht_1,dif])
        #     den = np.array(den)
        #     den = np.hstack((den,pad))
        #
        # if ht_1 < 768:
        #     dif = 768 - ht_1
        #     img = ImageOps.expand(img, border=(0,0,0,dif), fill=0)
        #     pad = np.zeros([dif,wd_1])
        #     den = np.array(den)
        #     den = np.vstack((den,pad))

        # plt.figure("org-img")
        # plt.imshow(img)
        # plt.show()
        # print img.size



        img = img_transform(img)

        img = Variable(img[None,:,:,:],volatile=True).cuda()

        pred_map = net.test_forward(img)
        pred_map = pred_map.cpu().data.numpy()[0, 0, :, :]

        gt_count = np.sum(den)
        pred_cnt = np.sum(pred_map) / 2550.0
        print("gt_%f,et_%f",gt_count,pred_cnt)

        den = den / np.max(den + 1e-20)
        den = den[0:ht_1, 0:wd_1]
        plt.figure("gt-den" + filename)
        plt.imshow(den)
        plt.show()


        pred_map = pred_map / np.max(pred_map + 1e-20)
        pred_map = pred_map[0:ht_1, 0:wd_1]
        plt.figure("pre-den"+filename)
        plt.imshow(pred_map)
        plt.show()


if __name__ == '__main__':
    main()




