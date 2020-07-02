#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    """
    Class of Convolution Neural Network
    that applys kernel over x_reshaped to compute x_conv_out
    """
    #pass
    # Remember to delete the above 'pass' after your implementation
    ### YOUR CODE HERE for part 1g
    def __init__(self, word_embed_size, char_embed_size=50, k=5, padding=1):
        """
        Init CNN Layers.
        @param word_embed_size (int): size of word embedding
        @param char_embed_size (int) = 50: size of word embedding
        @param k (int) = 5: kernel size for convolution
        @padding (int) = 1: size of padding applied to x_reshaped bilaterally

        """
        super(CNN, self).__init__() # Initialize self._modules as OrderedDict

        self.word_embed_size = word_embed_size
        self.char_embed_size = char_embed_size
        self.k = k
        self.padding = padding

        # Default Values
        self.apply_conv = None
        self.apply_maxpool = None
        
        # Initialize Variables
        """
        torch.nn.Conv1d(in_channels,  # 
                        out_channels, # f = number of output channels = word_embed_size
                        kernel_size,  # k=5
                        stride=1, 
                        padding=0,    # padding=1
                        dilation=1, 
                        groups=1, 
                        bias=True, 
                        padding_mode='zeros'
                        )
        """
        self.apply_conv = nn.Conv1d(in_channels = self.char_embed_size,
                              out_channels = self.word_embed_size, 
                              kernel_size=self.k, 
                              padding=self.padding
                              )

    def forward(self, x):
        """
        @param x (tensor): x_reshaped in shape (b, char_embed_size, m_word),
                      where b = batch size,
                      char_embed_size = size of the character embedding, and
                      m_word = length of longest word in the batch.
        @return x_conv_out (tensor): tensor of shape (b, word_embed_size)
        """
        # x_conv shape (b, word_embed_size, m_word-k+1)
        x_conv = self.apply_conv(x) #  (b, word_embed_size, m_word+2*padding-k+1)

        m_word = x.shape[2]
        """
        torch.nn.MaxPool1d(kernel_size, 
                           stride=None, 
                           padding=0, 
                           dilation=1, 
                           return_indices=False, 
                           ceil_mode=False
                           )
        """
        apply_maxpool = nn.MaxPool1d(kernel_size = m_word + 2*self.padding - self.k + 1)
        x_conv_out = apply_maxpool(F.relu(x_conv)).squeeze(2)


        return x_conv_out

    ### END YOUR CODE

