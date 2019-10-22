import torch.nn as nn
import torch.nn.functional as F


class SlideNet(nn.Module):
    """
    Slided window network
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=6)
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=4)
        self.fc1 = nn.Linear(in_features=80, out_features=32)
        self.fc2 = nn.Linear(in_features=32, out_features=16)
        self.fc3 = nn.Linear(in_features=16, out_features=1)

    def forward(self, x):
        out = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2))
        out = F.relu(F.max_pool2d(self.conv2(out), kernel_size=2))
        out = out.view(-1, 80)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        return self.fc3(out)
