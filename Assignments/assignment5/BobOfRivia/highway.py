#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
"""

import torch
import torch.nn as nn

class Highway(nn.Module):
    # Remember to delete the above 'pass' after your implementation
    ### YOUR CODE HERE for part 1f

    def __init__(self,D_emb):
        super(Highway,self).__init__()

        self.D_emb = D_emb

        self.linear = nn.Linear(D_emb,D_emb)

        self.gate_linear = nn.Linear(D_emb,D_emb)

    def forward(self, x):
        x_proj = self.linear(x).relu()

        x_gate = self.linear(x).sigmoid()

        return x_gate * x_proj + (1-x_gate) * x


    ### END YOUR CODE

