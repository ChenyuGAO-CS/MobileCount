import torch
import torch.nn as tnn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable


class VGG16(tnn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        self.layer1 = tnn.Sequential(

            # 1-1 conv layer
            tnn.Conv2d(3, 64, kernel_size=3, padding=1),
            tnn.BatchNorm2d(64),
            tnn.ReLU(),

            # 1-2 conv layer
            tnn.Conv2d(64, 64, kernel_size=3, padding=1),
            tnn.BatchNorm2d(64),
            tnn.ReLU(),

            # 1 Pooling layer
            tnn.MaxPool2d(kernel_size=2, stride=2))

        self.layer2 = tnn.Sequential(

            # 2-1 conv layer
            tnn.Conv2d(64, 128, kernel_size=3, padding=1),
            tnn.BatchNorm2d(128),
            tnn.ReLU(),

            # 2-2 conv layer
            tnn.Conv2d(128, 128, kernel_size=3, padding=1),
            tnn.BatchNorm2d(128),
            tnn.ReLU(),

            # 2 Pooling lyaer
            tnn.MaxPool2d(kernel_size=2, stride=2))

        self.layer3 = tnn.Sequential(

            # 3-1 conv layer
            tnn.Conv2d(128, 256, kernel_size=3, padding=1),
            tnn.BatchNorm2d(256),
            tnn.ReLU(),

            # 3-2 conv layer
            tnn.Conv2d(256, 256, kernel_size=3, padding=1),
            tnn.BatchNorm2d(256),
            tnn.ReLU(),

            # 3 Pooling layer
            tnn.MaxPool2d(kernel_size=2, stride=2))

        self.layer4 = tnn.Sequential(

            # 4-1 conv layer
            tnn.Conv2d(256, 512, kernel_size=3, padding=1),
            tnn.BatchNorm2d(512),
            tnn.ReLU(),

            # 4-2 conv layer
            tnn.Conv2d(512, 512, kernel_size=3, padding=1),
            tnn.BatchNorm2d(512),
            tnn.ReLU(),

            # 4 Pooling layer
            tnn.MaxPool2d(kernel_size=2, stride=2))

        self.layer5 = tnn.Sequential(

            # 5-1 conv layer
            tnn.Conv2d(512, 512, kernel_size=3, padding=1),
            tnn.BatchNorm2d(512),
            tnn.ReLU(),

            # 5-2 conv layer
            tnn.Conv2d(512, 512, kernel_size=3, padding=1),
            tnn.BatchNorm2d(512),
            tnn.ReLU(),

            # 5 Pooling layer
            tnn.MaxPool2d(kernel_size=2, stride=2),
            tnn.AdaptiveAvgPool2d((7, 7)))

        self.layer6 = tnn.Sequential(

            # 6 Fully connected layer
            # Dropout layer omitted since batch normalization is used.
            tnn.Linear(512*7*7, 4096),
            tnn.BatchNorm1d(4096),
            tnn.ReLU())

        self.layer7 = tnn.Sequential(

            # 7 Fully connected layer
            # Dropout layer omitted since batch normalization is used.
            tnn.Linear(4096, 4096),
            tnn.BatchNorm1d(4096),
            tnn.ReLU())

        self.layer8 = tnn.Sequential(

            # 8 output layer
            tnn.Linear(4096, 1000),
            tnn.BatchNorm1d(1000),
            tnn.Softmax())

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        vgg16_features = out.view(out.size(0), -1)
        out = self.layer6(vgg16_features)
        out = self.layer7(out)
        out = self.layer8(out)
        return out