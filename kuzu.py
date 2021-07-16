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
        
        self.hidden = nn.Linear(28*28, 250)
        self.output = nn.Linear(250, 10)

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
        
        self.conv1 = nn.Conv2d(1, 6, 5)     # image 24 x 24
        self.pool = nn.MaxPool2d(2)         # image 12 x 12 
        self.conv2 = nn.Conv2d(6,50,5)      # image 8 x 8
        self.fc = nn.Linear(4*4*50, 300)    # image 4 x 4
        self.output = nn.Linear(300, 10)


    def forward(self, x):

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc(x))
        x = self.output(x)
        x = F.log_softmax(x, dim=1)
        return x
