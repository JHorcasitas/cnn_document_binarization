import torch
import torch.nn as nn
import torch.nn.functional as F


class BinNet(nn.Module):
    def __init__(self):
        super(BinNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, 
                               out_channels=10, 
                               kernel_size=6,
                               stride=1)
        self.conv2 = nn.Conv2d(in_channels=10,
                               out_channels=20,
                               kernel_size=4,
                               stride=1)
        self.fc1 = nn.Linear(in_features=80, out_features=32)
        self.fc2 = nn.Linear(in_features=32, out_features=16)
        self.fc3 = nn.Linear(in_features=16, out_features=1)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, (2, 2))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, (2, 2))
        x = x.view(-1, 80)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)