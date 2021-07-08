# rect.py

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class Network(torch.nn.Module):
    def __init__(self, layer, hid):
        super(Network, self).__init__()
        self.layer = layer
        # INSERT CODE HERE

    def forward(self, input):
        output = 0*input[:,0] # CHANGE CODE HERE
        return output

def graph_hidden(net, layer, node):
    plt.clf()
    # INSERT CODE HERE
