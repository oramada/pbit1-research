"""Compact conventional RGB CNN for the next native-operator benchmark."""
import torch
from torch import nn
from torch.nn import functional as F


class CifarCNN16(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.head = nn.Linear(16*4*4, 10)

    def activations(self, x):
        first = F.max_pool2d(F.relu(self.conv1(x)), 2)
        second = F.max_pool2d(F.relu(self.conv2(first)), 2)
        return first, second

    def forward(self, x):
        return self.head(self.activations(x)[1].flatten(1))


def features(raw):
    x = torch.as_tensor(raw).permute(0, 3, 1, 2).contiguous().float()/255
    return F.avg_pool2d(x, 2).contiguous()
