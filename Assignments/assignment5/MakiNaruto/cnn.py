#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):

    # Remember to delete the above 'pass' after your implementation
    ### YOUR CODE HERE for part 1g

    def __init__(self, e_char, e_word, kernel_size, max_word_len):
        super(CNN, self).__init__()

        self.Conv1d = nn.Conv1d(in_channels=e_char, out_channels=e_word, kernel_size=kernel_size, padding=1)
        self.MaxPool = nn.MaxPool1d(max_word_len-kernel_size+1+2)

    def forward(self, X_reshape: torch.Tensor) -> torch.Tensor:

        X_conv = self.Conv1d(X_reshape)
        X_conv_out = self.MaxPool(F.relu(X_conv))

        return X_conv_out




    ### END YOUR CODE

