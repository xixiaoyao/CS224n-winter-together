#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Highway(nn.Module):
    """
    Class that computes X_highway from X_conv_out
    """
    # Remember to delete the above 'pass' after your implementation
    ### YOUR CODE HERE for part 1f

    def __init__(self, word_embed_size):
      """
      Init Highway Layer.

      @param word_embed_size (int): Embedding size (dimensionality) of word
      
      """
      super(Highway, self).__init__() # Initialize self._modules as OrderedDict
      self.word_embed_size = word_embed_size

      # default values
      self.w_proj = None
      self.w_gate = None

      # initialize variables
      # torch.nn.Linear(in_features, out_features, bias=True)
      self.w_proj = nn.Linear(word_embed_size, word_embed_size, bias=True) # W_project
      self.w_gate = nn.Linear(word_embed_size, word_embed_size, bias=True) # W_gate

    
    def forward(self, x):
      """Maps x_conv_out to x_highway
       # nn.Linear
      @param x (tensor): x_conv_out tensor of shape (b, word_embed_size), 
                         where b = batch size
      @returns x_highway (tensor): tenosor of shape (b, word_embed_size)
      """

      x_proj = F.relu(self.w_proj(x)) # (b, word_embed_size)
      x_gate = torch.sigmoid(self.w_gate(x)) # (b, word_embed_size)

      # element wise multiplication: * or mul()
      x_highway = x_gate * x_proj + (1 - x_gate) * x # (b, word_embed_size)

      return x_highway

    ### END YOUR CODE