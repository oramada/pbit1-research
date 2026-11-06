"""Conventional FP32 CNN; native sampler code consumes the exported arrays only."""
import torch
from torch import nn
from torch.nn import functional as F

class FashionCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=nn.Conv2d(1,8,3,padding=1)
        self.conv2=nn.Conv2d(8,16,3,padding=1)
        self.head=nn.Linear(16*3*3,10)

    def activations(self,x):
        first=F.max_pool2d(F.relu(self.conv1(x)),2)
        second=F.max_pool2d(F.relu(self.conv2(first)),2)
        return first,second

    def forward(self,x):
        return self.head(self.activations(x)[1].flatten(1))

def features(raw):
    return F.avg_pool2d(raw[:,None].float()/255,2).contiguous()
