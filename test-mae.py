from matplotlib import pyplot as plt
import os
import random
import torch
from torch.autograd import Variable
import torchvision.transforms as standard_transforms
import misc.transforms as own_transforms
import pandas as pd

from models.CC import CrowdCounter
from config import cfg
from misc.utils import *
import scipy.io as sio
from PIL import Image, ImageOps

import time

torch.cuda.set_device(0)
torch.backends.cudnn.benchmark = True

exp_name = './DULR-display-save-mat'
if not os.path.exists(exp_name):
    os.mkdir(exp_name)

mean_std = ([0.452016860247, 0.447249650955, 0.431981861591], [0.23242045939, 0.224925786257, 0.221840232611])

img_transform = standard_transforms.Compose([
    standard_transforms.ToTensor(),
    standard_transforms.Normalize(*mean_std)
])
restore = standard_transforms.Compose([
    own_transforms.DeNormalize(*mean_std),
    standard_transforms.ToPILImage()
])
pil_to_tensor = standard_transforms.ToTensor()

dataRoot = './exp/data/shanghaitech_part_B/test'
# dataRoot = './exp/data/shanghaitech_part_A/test'

# LWRF+Mobv2
# model_path = './exp/01-27_19-30_SHHB_MobLWRN_0.0001/all_ep_529_mae_8.3_mse_13.3.pth'
# LWRF+res50
# model_path = './exp/12-30_15-58_SHHB_LWRN_0.0001/all_ep_440_mae_6.8_mse_11.4.pth'
# LWRF+res101
# model_path = './exp/01-10_14-49_SHHA_LWRN_0.0001/all_ep_180_mae_69.4_mse_122.4.pth'
# RFN+res50
# model_path = './exp/02-05_11-58_SHHB_RFN_0.0001/all_ep_350_mae_7.5_mse_12.8.pth'
# DeepLabv3+res50
# model_path = './exp/02-23_17-59_SHHB_DeepLabv3_0.0001/all_ep_1_mae_45.9_mse_54.2.pth'
# DeepLabv3+res101
# model_path = './exp/03-23_10-46_SHHB_ShufLWRN_0.0001/all_ep_285_mae_8.2_mse_12.8.pth'
model_path = './exp/03-30_13-38_GCC_MobLWRN_0.0001_rd/all_ep_287_mae_30.2_mse_64.1.pth'

def main():
    # file_list = [filename for filename in os.listdir(dataRoot+'/img/') if os.path.isfile(os.path.join(dataRoot+'/img/',filename))]
    file_list = [filename for root, dirs, filename in os.walk(dataRoot + '/img/')]
    # pdb.set_trace()

    # ht_img = cfg.TRAIN.INPUT_SIZE[0]
    # wd_img = cfg.TRAIN.INPUT_SIZE[1]

    test(file_list[0], model_path)


def test(file_list, model_path):
    net = CrowdCounter(cfg.GPU_ID, cfg.NET)
    net.load_state_dict(torch.load(model_path))
    net.cuda()
    net.eval()

    maes = AverageMeter()
    mses = AverageMeter()

    step = 0
    time_sampe = 0
    for filename in file_list:
        step = step + 1
        print filename
        imgname = dataRoot + '/img/' + filename
        filename_no_ext = filename.split('.')[0]

        denname = dataRoot + '/den/' + filename_no_ext + '.csv'
        den = pd.read_csv(denname, sep=',', header=None).values

        # den = sio.loadmat(dataRoot + '/den/' + filename_no_ext + '.mat')
        # den = den['map']

        den = den.astype(np.float32, copy=False)

        img = Image.open(imgname)

        if img.mode == 'L':
            img = img.convert('RGB')

        # prepare
        wd_1, ht_1 = img.size
        # pdb.set_trace()

        # if wd_1 < 1024:
        #     dif = 1024 - wd_1
        #     img = ImageOps.expand(img, border=(0, 0, dif, 0), fill=0)
        #     pad = np.zeros([ht_1, dif])
        #     den = np.array(den)
        #     den = np.hstack((den, pad))
        #
        # if ht_1 < 768:
        #     dif = 768 - ht_1
        #     img = ImageOps.expand(img, border=(0, 0, 0, dif), fill=0)
        #     pad = np.zeros([dif, wd_1])
        #     den = np.array(den)
        #     den = np.vstack((den, pad))

        img = img_transform(img)

        gt_count = np.sum(den)

        img = Variable(img[None, :, :, :], volatile=True).cuda()

        # forward
        pred_map = net.test_forward(img)



        pred_map = pred_map.cpu().data.numpy()[0, 0, :, :]
        pred_cnt = np.sum(pred_map) / 2550.0
        pred_map = pred_map / np.max(pred_map + 1e-20)
        pred_map = pred_map[0:ht_1, 0:wd_1]

        den = den / np.max(den + 1e-20)
        den = den[0:ht_1, 0:wd_1]

        maes.update(abs(gt_count - pred_cnt))
        mses.update((gt_count - pred_cnt) * (gt_count - pred_cnt))

    mae = maes.avg
    mse = np.sqrt(mses.avg)

    print '\n[MAE: %fms][MSE: %fms]' % (mae, mse)


def get_pts(data):
    pts = []
    cols, rows = data.shape
    data = data * 100

    for i in range(0, rows):
        for j in range(0, cols):
            loc = [i, j]
            for i_pt in range(0, int(data[i][j])):
                pts.append(loc)
    return pts


if __name__ == '__main__':
    main()




