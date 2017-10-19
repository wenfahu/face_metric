import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F

from inceptionresnetv2 import inceptionresnetv2

class LOCNet(nn.Module):
    def __init__(self):
        super(LOCNet, self).__init__()
        self.features =  nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.Dropout()
            # Flatten()
        )
        self.linear = nn.Linear(5184, 6)
        nn.init.constant(self.linear.weight, 0)
        self.linear.bias = Parameter(torch.Tensor([1, 0, 0, 0, 1, 0]))
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x.view(-1, 2, 3)

class TransClf(nn.Module):
    def __init__(self, num_classes=100):
        super(TransClf, self).__init__()
        self.loc_net = LOCNet()
        self.features = inceptionresnetv2(num_classes=num_classes)
    def forward(self, x):
        out = self.loc_net(x)
        grid = F.affine_grid(out, x.size())
        out = F.grid_sample(x, grid)
        out = self.features(out)
        return out
