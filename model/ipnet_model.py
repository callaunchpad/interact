import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np


'''
Flatten module
Source: https://stackoverflow.com/questions/45584907/flatten-layer-of-pytorch-build-by-sequential-container
'''
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size()[0], -1)

'''
HO-CNN - Baseline model
Source: https://github.com/ywchao/ho-rcnn/blob/master/models/rcnn_caffenet_ho_pconv/train.prototxt
'''
class HOCNN(nn.Module):
    def __init__(self):
        super(HOCNN, self).__init__()
        # Formula for new image size calculation W′=(W−F+2P)/S+1
        
        # Layers for human stream
        # Input: [batch_size, input_channels, input_height, input_width] => [32, 3, 64, 64]
        self.h_conv1 = nn.Sequential(
                nn.Conv2d(3, 96, kernel_size=11, stride=4),
                nn.ReLU()
                )
        self.h_pool1 = nn.MaxPool2d(3, stride=2, padding=1)
        self.h_norm1 = nn.BatchNorm2d(96)
        self.h_conv2 = nn.Sequential(
                nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
                nn.ReLU()
                )
        self.h_pool2 = nn.MaxPool2d(3, stride=2)
        self.h_norm2 = nn.BatchNorm2d(256)
        self.h_conv3 = nn.Sequential(
                nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
                nn.ReLU()
                )
        self.h_conv4 = nn.Sequential(
                nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
                nn.ReLU()
                )
        self.h_conv5 = nn.Sequential(
                nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
                nn.ReLU()
                )
        self.h_pool3 = nn.MaxPool2d(3, stride=2)
        self.h_fcn1 = nn.Sequential(
                Flatten(),
                nn.Linear(256, 4096),
                nn.ReLU()
                )
        self.h_drop1 = nn.Dropout(p=0.5)
        self.h_fcn2 = nn.Sequential(
                nn.Linear(4096, 4096),
                nn.ReLU()
                )
        self.h_drop2 = nn.Dropout(p=0.5)
        self.h_fcn3 = nn.Linear(4096, 600)

        # Layers for object stream
        self.o_conv1 = nn.Sequential(
                nn.Conv2d(3, 96, kernel_size=11, stride=4),
                nn.ReLU()
                )
        self.o_pool1 = nn.MaxPool2d(3, stride=2, padding=1)
        self.o_norm1 = nn.BatchNorm2d(96)
        self.o_conv2 = nn.Sequential(
                nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
                nn.ReLU()
                )
        self.o_pool2 = nn.MaxPool2d(3, stride=2)
        self.o_norm2 = nn.BatchNorm2d(256)
        self.o_conv3 = nn.Sequential(
                nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
                nn.ReLU()
                )
        self.o_conv4 = nn.Sequential(
                nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
                nn.ReLU()
                )
        self.o_conv5 = nn.Sequential(
                nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
                nn.ReLU()
                )
        self.o_pool3 = nn.MaxPool2d(3, stride=2)
        self.o_fcn1 = nn.Sequential(
                Flatten(),
                nn.Linear(256, 4096),
                nn.ReLU()
                )
        self.o_drop1 = nn.Dropout(p=0.5)
        self.o_fcn2 = nn.Sequential(
                nn.Linear(4096, 4096),
                nn.ReLU()
                )
        self.o_drop2 = nn.Dropout(p=0.5)
        self.o_fcn3 = nn.Linear(4096, 600)
        
        # Layers for pairwise stream
        self.p_conv1 = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=5, stride=1),
                nn.ReLU()
                )
        self.p_pool1 = nn.MaxPool2d(3, stride=2)
        self.p_conv2 = nn.Sequential(
                nn.Conv2d(64, 32, kernel_size=5, stride=1),
                nn.ReLU()
                )
        self.p_pool2 = nn.MaxPool2d(3, stride=2)
        self.p_fcn1 = nn.Sequential(
                Flatten(),
                nn.Linear(4608, 256), # Input size is probably wrong cuz I'm bad at math
                nn.ReLU()
                )
        self.p_fcn2 = nn.Linear(256, 600)

    def forward(self, h, o, p):
        # h: human, o: object, p: pairwise
        
        # Forward pass human stream
        
        h = self.h_conv1(h)
        h = self.h_pool1(h)
        h = self.h_norm1(h)
        h = self.h_conv2(h)
        h = self.h_pool2(h)
        h = self.h_norm2(h)
        h = self.h_conv3(h)
        h = self.h_conv4(h)
        h = self.h_conv5(h)
        h = self.h_pool3(h)
        h = self.h_fcn1(h)
        h = self.h_drop1(h)
        h = self.h_fcn2(h)
        h = self.h_drop2(h)
        h = self.h_fcn3(h)
        
        # Forward pass object stream
        
        o = self.o_conv1(o)
        o = self.o_pool1(o)
        o = self.o_norm1(o)
        o = self.o_conv2(o)
        o = self.o_pool2(o)
        o = self.o_norm2(o)
        o = self.o_conv3(o)
        o = self.o_conv4(o)
        o = self.o_conv5(o)
        o = self.o_pool3(o)
        o = self.o_fcn1(o)
        o = self.o_drop1(o)
        o = self.o_fcn2(o)
        o = self.o_drop2(o)
        o = self.o_fcn3(o)
        
        # Forward pass pairwise stream
        
        p = self.p_conv1(p)
        p = self.p_pool1(p)
        p = self.p_conv2(p)
        p = self.p_pool2(p)
        p = self.p_fcn1(p)
        p = self.p_fcn2(p)

        summed_results = torch.add(h, torch.add(o, p)) # This might not be adding along the right axis but I can't check, omega sketch
        
        return F.log_softmax(summed_results)

class HORCNN(nn.Module):
    def __init__(self):
        super(HORCNN, self).__init__()
        pass

    def forward(self, add_args_here):
        pass

    # Potentially more functions here, names should start with an underscore

class DNN(nn.Module):
    def __init__(self):
        super(DNN, self).__init__()
        pass

    def forward(self, add_args_here):
        pass

    # Potentially more functions here, names should start with an underscore
