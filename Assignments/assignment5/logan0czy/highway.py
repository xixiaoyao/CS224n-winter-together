#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5

Usage:
  highway.py (options)

Options:
  -h --help  show this document
  --view    view model parameters
  --value   input, output and intermediate value check
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.special import expit
from docopt import docopt

class Highway(nn.Module):
    # Remember to delete the above 'pass' after your implementation
    ### YOUR CODE HERE for part 1f
    def __init__(self, in_dim, out_dim, dropout_rate=0.5):
        """Generate highway network layers.
        """
        super(Highway, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dropout_rate = dropout_rate
        
        self.proj = nn.Linear(self.in_dim, self.out_dim)
        self.gate = nn.Linear(self.in_dim, self.out_dim)
        self.dropout = nn.Dropout(self.dropout_rate)
        
    def forward(self, x):
        """
        param:
         x tensor (*, in_dim)
        return:
         x_embed tensor (*, out_dim)
        """
        x_proj = F.relu(self.proj(x))
        x_gate = torch.sigmoid(self.gate(x))
        x_highway = x_gate*x_proj + (1-x_gate)*x
        x_embed = self.dropout(x_highway)
        return x_embed
    ### END YOUR CODE


if __name__ == '__main__':
    args = docopt(__doc__)
    print("highway net sanity check......")
    seed = 2020
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    x = torch.tensor([[1, 2, 3, 4], [-2, -4, -6, -8]], dtype=torch.float)
    model = Highway(x.size()[-1], x.size()[-1])
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.ones_(p)
        else:
            nn.init.zeros_(p) 
    if args['--view']:
        print("model parameters dic:\n")
        for p in model.parameters():
            print(p)
    elif args['--value']:
        x_out = model(x)
        print("input:\n", x)
        print("output:\n", x_out)
        print("input size:", x.size(), "type:", x.dtype)
        print("output size:", x_out.size(), "type:", x_out.dtype)
        
        print("expected highway out:")
        x = x.numpy()
        weight = np.ones((x.shape[-1], x.shape[-1]))
        gate = expit(x.dot(weight))
        highway_expect = gate*np.maximum(0, x.dot(weight)) + (1-gate)*x
        print(highway_expect)
    