import torch.nn as nn
import torch.nn.functional as F


class SlideNet(nn.Module):
    """
    Slided window network
    """
    def __init__(self):
        super(SlideNet, self).__init__()
    
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=6)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=4)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        flat_features = 80
        self.fc1   = nn.Linear(in_features=flat_features, out_features=32)
        self.fc2   = nn.Linear(in_features=32, out_features=16)
        self.fc3   = nn.Linear(in_features=16, out_features=1)

    def forward(self, x):
        out = F.relu(self.pool1(self.conv1(x)))
        out = F.relu(self.pool2(self.conv2(out)))
        out = out.view(-1, 80)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        return self.fc3(out)