import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import torchvision.models as models
import numpy as np
import math
from torch.autograd import Variable

model_url = 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth'


class PyramidFeatures(nn.Module):
    def __init__(self, C3_size, C4_size, C5_size, C6_size, feature_size=256):
        super(PyramidFeatures, self).__init__()

        # upsample C5 to get P5 from the FPN paper
        self.P6_1 = nn.Sequential(nn.Conv2d(C6_size, feature_size, kernel_size=1, stride=1, padding=0),
                                  nn.BatchNorm2d(feature_size),nn.ReLU(inplace=True))
        self.P6_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P6_2 = nn.Sequential(nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(feature_size),nn.ReLU(inplace=True))

        # upsample C5 to get P5 from the FPN paper
        self.P5_1 = nn.Sequential(nn.Conv2d(C5_size, feature_size, kernel_size=1, stride=1, padding=0),
                                  nn.BatchNorm2d(feature_size),nn.ReLU(inplace=True))
        self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P5_2 = nn.Sequential(nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(feature_size),nn.ReLU(inplace=True))

        # add P5 elementwise to C4
        self.P4_1 = nn.Sequential(nn.Conv2d(C4_size, feature_size, kernel_size=1, stride=1, padding=0),
                                  nn.BatchNorm2d(feature_size),nn.ReLU(inplace=True))
        self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P4_2 = nn.Sequential(nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(feature_size),nn.ReLU(inplace=True))

        # add P4 elementwise to C3
        self.P3_1 = nn.Sequential(nn.Conv2d(C3_size, feature_size, kernel_size=1, stride=1, padding=0),
                                  nn.BatchNorm2d(feature_size),nn.ReLU(inplace=True))
        self.P3_2 = nn.Sequential(nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(feature_size),nn.ReLU(inplace=True))

        

    def forward(self, inputs):

        C3, C4, C5,C6 = inputs
        
        P6_x = self.P6_1(C6)
        P6_upsampled_x = nn.Upsample(size=C5.size()[2:], mode='bilinear')(P6_x)
        P6_x = self.P6_2(P6_x)

        #print C5.shape
        P5_x = self.P5_1(C5)
        P5_x = P6_upsampled_x + P5_x
        P5_upsampled_x = nn.Upsample(size=C4.size()[2:], mode='bilinear')(P5_x)
        P5_x = self.P5_2(P5_x)

        #print P5_upsampled_x.shape
        #print P5_x.shape
        P4_x = self.P4_1(C4)
        P4_x = P5_upsampled_x + P4_x
        P4_upsampled_x = nn.Upsample(size=C3.size()[2:], mode='bilinear')(P4_x)
        P4_x = self.P4_2(P4_x)

        P3_x = self.P3_1(C3)
        P3_x = P3_x + P4_upsampled_x
        P3_x = self.P3_2(P3_x)

        return P3_x


def conv3x3(in_planes, out_planes, stride=1, bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=bias)


def conv1x1(in_planes, out_planes, stride=1, bias=False):
    "1x1 convolution"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     padding=0, bias=bias)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=0.05)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=0.05)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, expansion=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, inplanes*expansion, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(inplanes*expansion)
        self.conv2 = nn.Conv2d(inplanes*expansion, inplanes*expansion, kernel_size=3, stride=stride,
                               padding=1, bias=False, groups=inplanes*expansion)
        self.bn2 = nn.BatchNorm2d(inplanes*expansion)
        self.conv3 = nn.Conv2d(inplanes*expansion, planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class FPN(nn.Module):
    def __init__(self):
        super(FPN, self).__init__()
        self.inplanes = 32
        block = Bottleneck
        layers = [1, 2, 3, 4]
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 32, layers[0], stride=1, expansion=1)
        self.layer2 = self._make_layer(block, 64, layers[1], stride=2, expansion=6)
        self.layer3 = self._make_layer(block, 128, layers[2], stride=2, expansion=6)
        self.layer4 = self._make_layer(block, 256, layers[3], stride=2, expansion=6)

        fpn_sizes = [32, 64, 128, 256]
        self.fpn = PyramidFeatures(fpn_sizes[0], fpn_sizes[1], fpn_sizes[2], fpn_sizes[3])

        self.fc2 = nn.Conv2d(256, 1, kernel_size=1)
        
    def forward(self, x):
        size1 = x.shape[2:]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.maxpool(x)

        l1 = self.layer1(x)
        l2 = self.layer2(l1)
        l3 = self.layer3(l2)
        l4 = self.layer4(l3)
        x = self.fpn([l1,l2,l3,l4])

        x = self.fc2(x)
        x = F.upsample(x, size=size1, mode='bilinear')
        return x 

    def _make_layer(self, block, planes, blocks, stride, expansion):

        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride=stride, downsample=downsample, expansion=expansion))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, expansion=expansion))

        return nn.Sequential(*layers)

