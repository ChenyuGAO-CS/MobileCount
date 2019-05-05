import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torchvision.models as models
from PIL import Image, ImageOps
import torchvision

import time
import argparse
import gc

import datetime
import os
from models.VGG16 import VGG16
from models.MCNN import MCNN
from models.VGG import VGG
from models.VGG_decoder import VGG_decoder
from models.gcy import ResNetLW
from models.DeepLab_v3_2 import DeepLabv3
from models.LWRF_MobileNetv2 import MobileNetLWRF
from models.LWRF_ShuffleNetv2 import ShuffleNetLWRF
from models.CSRNet import CSRNet
from models.Res50 import Res50
from models.MobileNetv2 import MobileNetV2
from models.CMTL import CMTL
from models.SANet import SANet
from models.FPN import FPN
from models.MobileNetv2_org_4 import MobileNetV2 as mob_org
from models.Setting1_LWRN import MobileNetLWRF as setting1
from models.Setting2_LWRN import MobileNetLWRF as setting2
from models.Setting3_LWRN import MobileNetLWRF as setting3

pt_models = { 'resnet18': models.resnet18, 'resnet50': models.resnet50,
              'alexnet': models.alexnet, 'CMTL': CMTL, 'SANet': SANet,
              'vgg16': models.vgg16, 'squeezenet': models.squeezenet1_0,
              'densenet': models.densenet161, 'MobileNetV2': MobileNetV2,
              'MCNN': MCNN, 'VGG': VGG, 'VGG_decoder': VGG_decoder, 'FPN': FPN,
              'ResNetLW': ResNetLW, 'DeepLabv3': DeepLabv3, 'MobileNetLWRF': MobileNetLWRF,
              'ShuffleNetLWRF': ShuffleNetLWRF, 'CSRNet': CSRNet, 'Res50': Res50,
              'mob_org': mob_org, 'setting1': setting1, 'setting2': setting2, 'setting3': setting3}


def measure(model, x):
    # synchronize gpu time and measure fp
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        y_pred = model(x)
    torch.cuda.synchronize()
    elapsed_fp = time.time() - t0

    return elapsed_fp


def benchmark(model, x):
    # transfer the model on GPU
    model = model.cuda().eval()

    # DRY RUNS
    for i in range(10):
        _ = measure(model, x)

    print('DONE WITH DRY RUNS, NOW BENCHMARKING')

    # START BENCHMARKING
    t_forward = []
    t_backward = []
    for i in range(10):
        t_fp = measure(model, x)
        t_forward.append(t_fp)

    # free memory
    del model

    return t_forward


def main():

    # fix random
    torch.manual_seed(1234)

    # create model
    # print("=> creating model '{}'".format(m))
    model = pt_models['CMTL']()

    cudnn.benchmark = True

    scale = 0.875

    # print('Images transformed from size {} to {}'.format(
    #     int(round(max(model.input_size) / scale)),
    #     model.input_size))

    # mean_tfp = []
    # std_tfp = []
    # x = torch.randn(1, 3, 224, 224).cuda()
    # tmp = benchmark(model, x)
    # # NOTE: we are estimating inference time per image
    # mean_tfp.append(np.asarray(tmp).mean() / 1 * 1e3)
    # std_tfp.append(np.asarray(tmp).std() / 1 * 1e3)
    #
    # print(mean_tfp, std_tfp)

    batch_sizes = [1, 2, 4]
    mean_tfp = []
    std_tfp = []
    for i, bs in enumerate(batch_sizes):
        # x = torch.randn(bs, 3, 224, 224).cuda()
        x = torch.randn(bs, 3, 1920, 1080).cuda()
        tmp = benchmark(model, x)
        # NOTE: we are estimating inference time per image
        mean_tfp.append(np.asarray(tmp).mean() / bs * 1e3)
        std_tfp.append(np.asarray(tmp).std() / bs * 1e3)
        print(np.asarray(tmp).mean() / bs * 1e3)

    print(mean_tfp, std_tfp)


    # force garbage collection
    gc.collect()

if __name__ == '__main__':
	main()






