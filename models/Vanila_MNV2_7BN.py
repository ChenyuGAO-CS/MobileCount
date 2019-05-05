"""RefineNet-LightWeight. No RCU, only LightWeight-CRP block."""

import math

import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import torch
from torch.autograd import Variable
import torchvision.models as models


model_urls = {
    'mobilev2': './per_model/mobilenet_v2.pth.tar',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
}


# Helpers / wrappers
def conv3x3(in_planes, out_planes, stride=1, bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=bias)


def conv1x1(in_planes, out_planes, stride=1, bias=False):
    "1x1 convolution"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     padding=0, bias=bias)


class CRPBlock(nn.Module):

    def __init__(self, in_planes, out_planes, n_stages):
        super(CRPBlock, self).__init__()
        for i in xrange(n_stages):
            setattr(self, '{}_{}'.format(i + 1, 'outvar_dimred'),
                    conv1x1(in_planes if (i == 0) else out_planes,
                            out_planes, stride=1,
                            bias=False))
        self.stride = 1
        self.n_stages = n_stages
        self.maxpool = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)

    def forward(self, x):
        top = x
        for i in xrange(self.n_stages):
            top = self.maxpool(top)
            top = getattr(self, '{}_{}'.format(i + 1, 'outvar_dimred'))(top)
            x = top + x
        return x


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


class MobileNetV2(nn.Module):

    def __init__(self, num_classes=1, pretrained=True):
        self.inplanes = 32
        block = Bottleneck
        layers = [1, 2, 3, 4, 3, 3, 1]
        super(MobileNetV2, self).__init__()

        # implement of mobileNetv2
        # self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
        #                        bias=False)

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 16, layers[0], stride=1, expansion = 1)
        self.layer2 = self._make_layer(block, 24, layers[1], stride=2, expansion = 6)
        self.layer3 = self._make_layer(block, 32, layers[2], stride=2, expansion = 6)
        self.layer4 = self._make_layer(block, 64, layers[3], stride=2, expansion = 6)
        self.layer5 = self._make_layer(block, 96, layers[4], stride=1, expansion=6)
        self.layer6 = self._make_layer(block, 160, layers[5], stride=2, expansion=6)
        self.layer7 = self._make_layer(block, 320, layers[6], stride=1, expansion=6)

        self.dropout4 = nn.Dropout(p=0.5)
        self.clf_conv = nn.Conv2d(320, 1, kernel_size=3, stride=1,
                                  padding=1, bias=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        if pretrained:
            print('load the pre-trained model.')

            state_dict = torch.load('./per_model/model_best.pth.tar', map_location='cpu')  # add map_location='cpu' if no gpu

            # load dict of layer_Bottlenecks_0
            layer1_dict = self.layer1.state_dict()

            for k, v in state_dict.items():
                if k == 'state_dict':
                    pretrained_dict = {i.replace('module.bottlenecks.Bottlenecks_0.LinearBottleneck0_', ''): v[i] for i in v if
                                       i.replace('module.bottlenecks.Bottlenecks_0.LinearBottleneck0_', '') in layer1_dict}
                    # for i in v:
                    #     print(i)

            print('*****************layer1******************')

            # for k, v in layer1_dict.items():
            #     print(k)

            print("Restoring {}/{} parameters".format(len(pretrained_dict), len(layer1_dict)))

            layer1_dict.update(pretrained_dict)
            pretrained_dict.update(layer1_dict)
            self.layer1.load_state_dict(pretrained_dict)


            # load dict of layer_Bottlenecks_1
            layer2_dict = self.layer2.state_dict()
            for k, v in state_dict.items():
                if k == 'state_dict':
                    pretrained_dict2 = {i.replace('module.bottlenecks.Bottlenecks_1.LinearBottleneck1_', ''): v[i] for i in v if
                                       i.replace('module.bottlenecks.Bottlenecks_1.LinearBottleneck1_', '') in layer2_dict}
                    # for i in v:
                    #     print(i)
            print('*****************layer2******************')
            print("Restoring {}/{} parameters".format(len(pretrained_dict2), len(layer2_dict)))
            layer2_dict.update(pretrained_dict2)
            pretrained_dict2.update(layer2_dict)
            self.layer2.load_state_dict(pretrained_dict2)


            # load dict of layer_Bottlenecks_2
            layer3_dict = self.layer3.state_dict()
            for k, v in state_dict.items():
                if k == 'state_dict':
                    pretrained_dict3 = {i.replace('module.bottlenecks.Bottlenecks_2.LinearBottleneck2_', ''): v[i] for i in v if
                                       i.replace('module.bottlenecks.Bottlenecks_2.LinearBottleneck2_', '') in layer3_dict}
                    # for i in v:
                    #     print(i)
            print('*****************layer3******************')
            print("Restoring {}/{} parameters".format(len(pretrained_dict3), len(layer3_dict)))
            layer3_dict.update(pretrained_dict3)
            pretrained_dict3.update(layer3_dict)
            self.layer3.load_state_dict(pretrained_dict3)


            # load dict of layer_Bottlenecks_3
            layer4_dict = self.layer4.state_dict()
            for k, v in state_dict.items():
                if k == 'state_dict':
                    pretrained_dict4 = {i.replace('module.bottlenecks.Bottlenecks_3.LinearBottleneck3_', ''): v[i] for i in v if
                                       i.replace('module.bottlenecks.Bottlenecks_3.LinearBottleneck3_', '') in layer4_dict}
                    # for i in v:
                    #     print(i)
            print('*****************layer4******************')
            print("Restoring {}/{} parameters".format(len(pretrained_dict4), len(layer4_dict)))
            layer4_dict.update(pretrained_dict4)
            pretrained_dict4.update(layer4_dict)
            self.layer4.load_state_dict(pretrained_dict4)

            # load dict of layer_Bottlenecks_4
            layer5_dict = self.layer5.state_dict()
            for k, v in state_dict.items():
                if k == 'state_dict':
                    pretrained_dict5 = {i.replace('module.bottlenecks.Bottlenecks_4.LinearBottleneck4_', ''): v[i] for i
                                        in v if
                                        i.replace('module.bottlenecks.Bottlenecks_4.LinearBottleneck4_',
                                                  '') in layer5_dict}
                    # for i in v:
                    #     print(i)
            print('*****************layer5******************')
            print("Restoring {}/{} parameters".format(len(pretrained_dict5), len(layer5_dict)))
            layer5_dict.update(pretrained_dict5)
            pretrained_dict5.update(layer5_dict)
            self.layer5.load_state_dict(pretrained_dict5)

            # load dict of layer_Bottlenecks_5
            layer6_dict = self.layer6.state_dict()
            for k, v in state_dict.items():
                if k == 'state_dict':
                    pretrained_dict6 = {i.replace('module.bottlenecks.Bottlenecks_5.LinearBottleneck5_', ''): v[i] for i
                                        in v if
                                        i.replace('module.bottlenecks.Bottlenecks_5.LinearBottleneck5_',
                                                  '') in layer6_dict}
                    # for i in v:
                    #     print(i)
            print('****************layer6*******************')
            print("Restoring {}/{} parameters".format(len(pretrained_dict6), len(layer6_dict)))
            layer6_dict.update(pretrained_dict6)
            pretrained_dict6.update(layer6_dict)
            self.layer6.load_state_dict(pretrained_dict6)

            # load dict of layer_Bottlenecks_6
            layer7_dict = self.layer7.state_dict()
            for k, v in state_dict.items():
                if k == 'state_dict':
                    pretrained_dict7 = {i.replace('module.bottlenecks.Bottlenecks_6.LinearBottleneck6_', ''): v[i] for i
                                        in v if
                                        i.replace('module.bottlenecks.Bottlenecks_6.LinearBottleneck6_',
                                                  '') in layer7_dict}
                    # for i in v:
                    #     print(i)
            print('*****************layer7******************')
            print("Restoring {}/{} parameters".format(len(pretrained_dict7), len(layer7_dict)))
            layer7_dict.update(pretrained_dict7)
            pretrained_dict7.update(layer7_dict)
            self.layer7.load_state_dict(pretrained_dict7)



    def _make_crp(self, in_planes, out_planes, stages):
        layers = [CRPBlock(in_planes, out_planes, stages)]
        return nn.Sequential(*layers)

    def _make_layer(self, block, planes, blocks, stride, expansion):

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

    def forward(self, x):
        size1 = x.shape[2:]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        """add a max pool before BN"""
        x = self.maxpool(x)

        l1 = self.layer1(x)
        l1 = self.layer2(l1)

        l2 = self.layer3(l1)
        l2 = self.layer4(l2)

        l3 = self.layer5(l2)

        l4 = self.layer6(l3)
        l4 = self.layer7(l4)

        l4 = self.dropout4(l4)

        out = self.clf_conv(l4)

        out = F.upsample(out, size=size1, mode='bilinear')

        return out




