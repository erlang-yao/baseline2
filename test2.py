# -*- coding: utf-8 -*-
import torch
from torch.ao.nn.quantized import Softmax
from torch.nn import functional as F
from torch.utils.data import DataLoader
import numpy as np
from torch.nn import Module,MaxPool2d,Conv2d,Flatten,Linear,Sequential,Softmax
from torch.utils.tensorboard import SummaryWriter


class YY(Module):
    def __init__(self):
        super(YY,self).__init__()
        self.model1=Sequential(
            Conv2d(3,32,5,padding=2),
            MaxPool2d(2),
            Conv2d(32,32,5,padding=2),
            MaxPool2d(2),
            Conv2d(32,64,5,padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(64*4*4,64),
            Linear(64,10),
        )
    def forward(self,x):
        x=self.model1(x)
        return x
yy=YY()
print(yy)
x=torch.randn(64,3,32,32)
print(yy.forward(x).shape)
writer=SummaryWriter("yy")
writer.add_graph(yy,x)
writer.close()
