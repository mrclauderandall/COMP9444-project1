# kuzu.py


from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F

class NetLin(nn.Module):
    # linear function followed by log_softmax
    def __init__(self):
        super(NetLin, self).__init__()
        self.linfunc = nn.Linear(28*28, 10)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        
        x = self.linfunc(x)
        x = F.log_softmax(x, dim = -1)

        return x

class NetFull(nn.Module):
    # two fully connected tanh layers followed by log softmax
    def __init__(self):
        super(NetFull, self).__init__()
        
        self.hidden = nn.Linear(28*28, 100)
        self.output = nn.Linear(100, 10)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        
        x = self.hidden(x)
        x = torch.tanh(x)
        x = self.output(x)
        x = F.log_softmax(x, dim = 1)

        return x

class NetConv(nn.Module):
    # two convolutional layers and one fully connected layer,
    # all using relu, followed by log_softmax
    def __init__(self):
        super(NetConv, self).__init__()
        # INSERT CODE HERE

    def forward(self, x):
        return 0 # CHANGE CODE HERE
