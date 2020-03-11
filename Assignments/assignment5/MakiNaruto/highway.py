#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Highway(nn.Module):

    # Remember to delete the above 'pass' after your implementation
    ### YOUR CODE HERE for part 1f

    def __init__(self, e_word):
        super(Highway, self).__init__()

        self.W_proj = nn.Linear(e_word, e_word)
        self.W_gate = nn.Linear(e_word, e_word)
        # pass

    def forward(self, x_conv_out: torch.Tensor) -> torch.Tensor:

        x_proj = F.relu(self.W_proj(x_conv_out))
        x_gate = torch.sigmoid(self.W_gate(x_conv_out))

        x_highway = x_gate.mul(x_proj) + (1 - x_gate).mul(x_conv_out)

        return x_highway

    ### END YOUR CODE

