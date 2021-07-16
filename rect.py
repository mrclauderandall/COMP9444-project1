# rect.py

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class Network(torch.nn.Module):
    def __init__(self, layer, hid):
        super(Network, self).__init__()
        self.layer = layer

        self.hidden1 = nn.Linear(2, hid)

        if (layer == 2):
            self.hidden2 = nn.Linear(hid, hid)

        self.output = nn.Linear(hid, 1)

    def forward(self, input):

        x = input.view(input.shape[0], -1)
        self.x1 = self.hidden1(x)
        x = torch.tanh(self.x1)

        if (self.layer == 2):
            x = self.hidden2(x)
            x = torch.tanh(x)
            self.x2 = x

        x = self.output(x)
        x = torch.sigmoid(x)
        return x

def graph_hidden(net, layer, node):
    plt.clf()
    xrange = torch.arange(start=-8,end=8.1,step=0.01,dtype=torch.float32)
    yrange = torch.arange(start=-8,end=8.1,step=0.01,dtype=torch.float32)
    xcoord = xrange.repeat(yrange.size()[0])
    ycoord = torch.repeat_interleave(yrange, xrange.size()[0], dim=0)
    grid = torch.cat((xcoord.unsqueeze(1),ycoord.unsqueeze(1)),1)


    with torch.no_grad():       # suppress updating of gradients
        net.eval()              # toggle batch norm, dropout
        output = net(grid)
        net.train()             # toggle batch norm, dropout back again
        
        if (layer == 2):
            pred = (net.x2[:,node] >= 0).float()
        else:
            pred = (net.x1[:,node] >= 0).float()

        # plot function computed by model
        plt.clf()
        plt.pcolormesh(xrange,yrange,pred.cpu().view(yrange.size()[0],xrange.size()[0]),
                       cmap='Wistia', shading='auto')
