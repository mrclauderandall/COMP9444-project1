'''

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F

# Hyperparameters for our network
input_size = 784
hidden_sizes = [128, 64]
output_size = 10# Build a feed-forward network
model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[1], output_size),
                      nn.Softmax(dim=1))
print(model)
print(model[0])
#print(model.fc1)
'''

mylist = ['a', 'b', 'c', 'd', 'e', 'f', 'g']

print(mylist[:,0])