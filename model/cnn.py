import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils import generate_equiangular_tight


class CNNCifar(nn.Module):
    def __init__(self, num_classes):
        super(CNNCifar, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc0 = nn.Linear(64 * 5 * 5, 384)
        self.fc1 = nn.Linear(384, 192)
        self.fc = nn.Linear(192, num_classes)

        self.fdim = 192
        self.projection = {self.fc0, self.fc1}
        self.temp = nn.Parameter(torch.tensor(1.0))
        self.etf = nn.Parameter(F.normalize(generate_equiangular_tight(num_classes, 192), dim=1))

    def forward(self, x, proto=None):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 5 * 5)
        x = F.relu(self.fc0(x))
        x1 = F.relu(self.fc1(x))
        if proto is None:
            x = self.fc(x1)
        else:
            x1 = F.normalize(x1, dim=1)
            x = x1.mm(proto.T)
        return x1, x


