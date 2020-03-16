#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
"""

import torch
import torch.nn as nn
import math
import numpy as np

class CNN(nn.Module):
    # Remember to delete the above 'pass' after your implementation
    ### YOUR CODE HERE for part 1g

    def __init__(self,char_emb_size,word_emb_size,kernel_size):

        # equals to channel of kernal
        self.char_emb_size = char_emb_size
        self.word_emb_size = word_emb_size

        self.kernel_size = kernel_size

        super(CNN,self).__init__()

        self.conv = nn.Conv1d(in_channels=1,out_channels=self.word_emb_size,kernel_size=(kernel_size,char_emb_size),stride=1)

        self.maxpool = nn.MaxPool1d(kernel_size=1,stride=1)

    # shape of  (batch_size_of_words,char_size,char_emb_size)
    # return (batch_size_of_words,word_emb_size)
    def forward(self, x):
        # TODO
        zeros = torch.zeros((x.shape[0],int(math.ceil((self.kernel_size-1)/2)),x.shape[2]))

        # x = np.concatenate((zeros,x,zeros),axis=1)
        x = torch.cat((zeros,x,zeros),dim=1)

        x = torch.tensor(x)

        x = x.unsqueeze(dim=1)

        # x = x.float()
        x_conv = self.conv(x)
        x_conv = x_conv.relu()
        poold = self.maxpool(x_conv[:,:,:,0])
        return poold[:,:,0]

    ### END YOUR CODE

