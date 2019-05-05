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
        P6_upsampled_x = self.P6_upsampled(P6_x)
        P6_x = self.P6_2(P6_x)
        #print C5.shape
        P5_x = self.P5_1(C5)
        P5_x = P6_upsampled_x + P5_x
        P5_upsampled_x = self.P5_upsampled(P5_x)
        P5_x = self.P5_2(P5_x)
        #print P5_upsampled_x.shape
        #print P5_x.shape
        P4_x = self.P4_1(C4)
        P4_x = P5_upsampled_x + P4_x
        P4_upsampled_x = self.P4_upsampled(P4_x)
        P4_x = self.P4_2(P4_x)

        P3_x = self.P3_1(C3)
        P3_x = P3_x + P4_upsampled_x
        P3_x = self.P3_2(P3_x)

        return P3_x

class Atrous_Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, rate=1, downsample=None):
        super(Atrous_Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               dilation=rate, padding=rate, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
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

class Atrous_ResNet_features(nn.Module):

    def __init__(self, block, layers, pretrained=False):
        super(Atrous_ResNet_features, self).__init__()
        self.inplanes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1, rate=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, rate=1)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, rate=1)
        self.layer4 = self._make_MG_unit(block, 512, stride=1, rate=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        if pretrained:
            print('load the pre-trained model resnet50.')
            resnet = models.resnet50(pretrained)
            self.conv1 = resnet.conv1
            self.bn1 = resnet.bn1
            self.layer1 = resnet.layer1
            self.layer2 = resnet.layer2
            self.layer3 = resnet.layer3
            self.layer4 = resnet.layer4

        '''for m in self.layer1.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
        for m in self.layer2.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
        for m in self.layer3.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()'''

    def _make_layer(self, block, planes, blocks, stride=1, rate=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, rate, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_MG_unit(self, block, planes, blocks=[1,2,1], stride=1, rate=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, rate=blocks[0]*rate, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, len(blocks)):
            layers.append(block(self.inplanes, planes, stride=1, rate=blocks[i]*rate))

        return nn.Sequential(*layers)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

		#output:1/16
        return x1,x2,x3,x4 

class Atrous_module(nn.Module):
    def __init__(self, inplanes, planes, rate):
        super(Atrous_module, self).__init__()
        self.atrous_convolution = nn.Conv2d(inplanes, planes, kernel_size=3,
                                            stride=1, padding=rate, dilation=rate)
        self.batch_norm = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.atrous_convolution(x)
        #x = self.batch_norm(x)
        x = self.relu(x)

        return x

class FPN(nn.Module):
    def __init__(self, num_classes, small=True, pretrained=False):
        super(FPN, self).__init__()
        self.bn = nn.BatchNorm2d(3)
        block = Atrous_Bottleneck
        self.resnet_features = Atrous_ResNet_features(block, [3, 4, 6], pretrained)
        layers = [3, 4, 6]
        fpn_sizes = [256, 512, 1024, 2048]
        self.fpn = PyramidFeatures(fpn_sizes[0], fpn_sizes[1], fpn_sizes[2], fpn_sizes[3])
        #rates = [1, 2, 4, 6]
        #self.aspp1 = Atrous_module(2048 , 256, rate=rates[0])
        #self.aspp2 = Atrous_module(2048 , 256, rate=rates[1])
        #self.aspp3 = Atrous_module(2048 , 256, rate=rates[2])
        #self.aspp4 = Atrous_module(2048 , 256, rate=rates[3])
        #self.image_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1,1)),nn.Conv2d(2048, 256, kernel_size=1))

        #self.fc1 = nn.Sequential(nn.Conv2d(1280, 256, kernel_size=1),nn.ReLU(inplace=True))
                                 #nn.BatchNorm2d(256))
        self.fc2 = nn.Conv2d(256, num_classes, kernel_size=1)
        
    def forward(self, x):
        size1 = x.shape[2:]
        x = self.bn(x)
        x1,x2,x3,x4 = self.resnet_features(x)
        x = self.fpn([x1,x2,x3,x4])
        #x1 = self.aspp1(x)
        #x2 = self.aspp2(x)
        #x3 = self.aspp3(x)
        #x4 = self.aspp4(x)
        #x5 = self.image_pool(x)
        #x5 = F.upsample(x5, size=x4.size()[2:], mode='bilinear')

        #x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        #x = torch.cat((x1, x5), dim=1)
        #print x.shape
        #x = self.fc1(x)
        x = self.fc2(x)
        # x = F.upsample(x, scale_factor=(16,16), mode='bilinear')
        x = F.upsample(x, size=size1, mode='bilinear')
        return x 

'''model = DeepLabv3(num_classes=1,pretrained=True)
a = Variable(torch.rand(3,512,512))
# print model
out = model(a.unsqueeze(0))
print out.shape'''

