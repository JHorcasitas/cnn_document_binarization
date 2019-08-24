"""
Network inspired on:
Highly Efficient Forward and Backward Propagation of Convolutional Neural
Networks for Pixelwise Classification
https://arxiv.org/pdf/1412.4526.pdf
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class StridedNet(nn.Module):
    
    def __init__(self):
        super(StridedNet, self).__init__()
    
        self.conv1 = nn.Conv2d(in_channels=1,
                               out_channels=10,
                               kernel_size=6,
                               stride=1,
                               dilation=1) 
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=1, dilation=1)

        self.conv2 = nn.Conv2d(in_channels=10,
                               out_channels=20,
                               kernel_size=4,
                               stride=1,
                               dilation=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=1, dilation=2)
        
        self.fc1 = nn.Conv2d(in_channels=20,
                             out_channels=32,
                             kernel_size=2,
                             stride=1,
                             dilation=4)

        self.fc2 = nn.Conv2d(in_channels=32,
                             out_channels=16,
                             kernel_size=1,
                             stride=1,
                             dilation=1)
        
        self.fc3 = nn.Conv2d(in_channels=16,
                             out_channels=1,
                             kernel_size=1,
                             stride=1,
                             dilation=1)

    def forward(self, x):
        out = F.gelu(self.pool1(self.conv1(x)))
        out = F.gelu(self.pool2(self.conv2(out)))
        out = F.gelu(self.fc1(out))
        out = F.gelu(self.fc2(out))
        out = self.fc3(out)
        return out