"""RefineNet-LightWeight. No RCU, only LightWeight-CRP block."""

import math

import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import torch
from torch.autograd import Variable
import torchvision.models as models

model_urls = {
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
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

def channel_split(x, split):
    """split a tensor into two pieces along channel dimension
    Args:
        x: input tensor
        split:(int) channel size for each pieces
    """
    assert x.size(1) == split * 2
    return torch.split(x, split, dim=1)


def channel_shuffle(x, groups):
    """channel shuffle operation
    Args:
        x: input tensor
        groups: input branch number
    """

    batch_size, channels, height, width = x.size()
    channels_per_group = int(channels / groups)

    x = x.view(batch_size, groups, channels_per_group, height, width)
    x = x.transpose(1, 2).contiguous()
    x = x.view(batch_size, -1, height, width)

    return x


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

class ShuffleUnit(nn.Module):

    def __init__(self, in_channels, out_channels, stride):
        super(ShuffleUnit, self).__init__()

        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels

        if stride != 1 or in_channels != out_channels:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 1),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, in_channels, 3, stride=stride, padding=1, groups=in_channels),
                nn.BatchNorm2d(in_channels),
                nn.Conv2d(in_channels, int(out_channels / 2), 1),
                nn.BatchNorm2d(int(out_channels / 2)),
                nn.ReLU(inplace=True)
            )

            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 3, stride=stride, padding=1, groups=in_channels),
                nn.BatchNorm2d(in_channels),
                nn.Conv2d(in_channels, int(out_channels / 2), 1),
                nn.BatchNorm2d(int(out_channels / 2)),
                nn.ReLU(inplace=True)
            )
        else:
            self.shortcut = nn.Sequential()

            in_channels = int(in_channels / 2)
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 1),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, in_channels, 3, stride=stride, padding=1, groups=in_channels),
                nn.BatchNorm2d(in_channels),
                nn.Conv2d(in_channels, in_channels, 1),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):

        if self.stride == 1 and self.out_channels == self.in_channels:
            shortcut, residual = channel_split(x, int(self.in_channels / 2))
        else:
            shortcut = x
            residual = x

        shortcut = self.shortcut(shortcut)
        residual = self.residual(residual)
        x = torch.cat([shortcut, residual], dim=1)
        x = channel_shuffle(x, 2)

        return x


class ShuffleNetLWRF(nn.Module):

    def __init__(self, num_classes=1, pretrained=True):

        super(ShuffleNetLWRF, self).__init__()

        out_channels = [48, 96, 192, 1024]

        self.pre = nn.Sequential(
            nn.Conv2d(3, 24, 3, padding=1),
            nn.BatchNorm2d(24)
        )
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_stage(24, out_channels[0], 3)
        self.layer2 = self._make_stage(out_channels[0], out_channels[1], 7)
        self.layer3 = self._make_stage(out_channels[1], out_channels[2], 3)
        self.layer4 = nn.Sequential(nn.Conv2d(out_channels[2], out_channels[3], 1), nn.BatchNorm2d(out_channels[3]), nn.ReLU(inplace=True))

        self.dropout4 = nn.Dropout(p=0.5)
        self.p_ims1d2_outl1_dimred = conv1x1(1024, 96, bias=False)
        self.mflow_conv_g1_pool = self._make_crp(96, 96, 4)
        self.mflow_conv_g1_b3_joint_varout_dimred = conv1x1(96, 48, bias=False)

        self.dropout3 = nn.Dropout(p=0.5)
        self.p_ims1d2_outl2_dimred = conv1x1(192, 48, bias=False)
        self.adapt_stage2_b2_joint_varout_dimred = conv1x1(48, 48, bias=False)
        self.mflow_conv_g2_pool = self._make_crp(48, 48, 4)
        self.mflow_conv_g2_b3_joint_varout_dimred = conv1x1(48, 48, bias=False)

        self.p_ims1d2_outl3_dimred = conv1x1(96, 48, bias=False)
        self.adapt_stage3_b2_joint_varout_dimred = conv1x1(48, 48, bias=False)
        self.mflow_conv_g3_pool = self._make_crp(48, 48, 4)
        self.mflow_conv_g3_b3_joint_varout_dimred = conv1x1(48, 48, bias=False)

        self.p_ims1d2_outl4_dimred = conv1x1(48, 48, bias=False)
        self.adapt_stage4_b2_joint_varout_dimred = conv1x1(48, 48, bias=False)
        self.mflow_conv_g4_pool = self._make_crp(48, 48, 4)

        self.dropout_clf = nn.Dropout(p=0.5)

        self.clf_conv = nn.Conv2d(48, 1, kernel_size=3, stride=1,
                                  padding=1, bias=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_crp(self, in_planes, out_planes, stages):
        layers = [CRPBlock(in_planes, out_planes, stages)]
        return nn.Sequential(*layers)

    def _make_stage(self, in_channels, out_channels, repeat):
        layers = []
        layers.append(ShuffleUnit(in_channels, out_channels, 2))

        while repeat:
            layers.append(ShuffleUnit(out_channels, out_channels, 1))
            repeat -= 1

        return nn.Sequential(*layers)

    def forward(self, x):
        size1 = x.shape[2:]
        x = self.pre(x)

        l1 = self.layer1(x)
        l2 = self.layer2(l1)
        l3 = self.layer3(l2)
        l4 = self.layer4(l3)

        l4 = self.dropout4(l4)
        x4 = self.p_ims1d2_outl1_dimred(l4)
        x4 = self.relu(x4)
        x4 = self.mflow_conv_g1_pool(x4)
        x4 = self.mflow_conv_g1_b3_joint_varout_dimred(x4)
        x4 = nn.Upsample(size=l3.size()[2:], mode='bilinear')(x4)

        l3 = self.dropout3(l3)
        x3 = self.p_ims1d2_outl2_dimred(l3)
        x3 = self.adapt_stage2_b2_joint_varout_dimred(x3)
        x3 = x3 + x4
        x3 = F.relu(x3)
        x3 = self.mflow_conv_g2_pool(x3)
        x3 = self.mflow_conv_g2_b3_joint_varout_dimred(x3)
        x3 = nn.Upsample(size=l2.size()[2:], mode='bilinear')(x3)

        x2 = self.p_ims1d2_outl3_dimred(l2)
        x2 = self.adapt_stage3_b2_joint_varout_dimred(x2)
        x2 = x2 + x3
        x2 = F.relu(x2)
        x2 = self.mflow_conv_g3_pool(x2)
        x2 = self.mflow_conv_g3_b3_joint_varout_dimred(x2)
        x2 = nn.Upsample(size=l1.size()[2:], mode='bilinear')(x2)

        x1 = self.p_ims1d2_outl4_dimred(l1)
        x1 = self.adapt_stage4_b2_joint_varout_dimred(x1)
        x1 = x1 + x2
        x1 = F.relu(x1)
        x1 = self.mflow_conv_g4_pool(x1)

        x1 = self.dropout_clf(x1)
        out = self.clf_conv(x1)

        out = F.upsample(out, size=size1, mode='bilinear')

        return out





