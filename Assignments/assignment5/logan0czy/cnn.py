#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5

Usage:
  cnn.py view
  cnn.py value
  cnn.py -h
"""

from docopt import docopt
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CNN(nn.Module):
    # Remember to delete the above 'pass' after your implementation
    ### YOUR CODE HERE for part 1g
    def __init__(self, char_embedding, n_features, kernel_size=5, padding=1):
        """define conv1d network
        params:
          char_embedding (int): characters' embedding dimension
          n_features (int): number of conv1d filters
          kernel_size (int): convolution window size
        """
        super(CNN, self).__init__()
        self.char_embedding = char_embedding
        self.conv = nn.Conv1d(char_embedding, n_features, kernel_size, padding=padding)

    def forward(self, x):
        """
        params:
          x (n_words, char_embed, n_chars): words in a sentence with embedded characters
        
        return:
          x_conv (n_words, word_embedding): embedded words matrix
        """
        assert x.size()[-2] == self.char_embedding, "input tensor shape invalid, should be (n_words, char_embed, n_chars)"
        x = self.conv(x)
        x = F.relu(x)
        x_conv, _ = torch.max(x, dim=-1)
        return x_conv
    
    ### END YOUR CODE


if __name__ == '__main__':
    args = docopt(__doc__)
    seed = 2020
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed // 2)
    
    x = torch.tensor([[[1., 1., 1., 1.],
                  [-2, -2, -2., -2.]],
                 [[2, 2, 1, 1],
                  [0.5, 0.5, 0, 0]]], dtype=torch.float32)
    print("input tensor shape:  ", x.size())
    x = x.permute(0, 2, 1).contiguous()
    model = CNN(x.size()[-2], 3, kernel_size=2)
    if args['view']:
        print("model's parameter print...")
        for p in model.parameters():
            print(p)
    elif args['value']:
        print("value confirmation...")
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.ones_(p)
            else:
                nn.init.zeros_(p)
        x_conv = model(x)
        print("input:\n{}\nsize: {}".format(x, x.size()))
        print("output:\n{}\nsize: {}".format(x_conv, x_conv.size()))