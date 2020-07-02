#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
model_embeddings.py: Embeddings for the NMT model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
Anand Dhoot <anandd@stanford.edu>
Michael Hahn <mhahn2@stanford.edu>
"""

import torch.nn as nn

# Do not change these imports; your module names should be
#   `CNN` in the file `cnn.py`
#   `Highway` in the file `highway.py`
# Uncomment the following two imports once you're ready to run part 1(j)

from cnn import CNN
from highway import Highway
"""
Init CNN Layers.
@param word_embed_size (int): size of word embedding
@param char_embed_size (int) = 50: size of word embedding
@param k (int) = 5: kernel size for convolution
@padding (int) = 1: size of padding applied to x_reshaped bilaterally

"""
# End "do not change"

class ModelEmbeddings(nn.Module):
    """
    Class that converts input words to their CNN-based embeddings.
    """

    def __init__(self, word_embed_size, vocab, char_embed_size=50, k=5, padding=1, dropout_rate=0.3):
        """
        Init the Embedding layer for one language
        @param word_embed_size (int): Embedding size (dimensionality) for the output word
        @param vocab (VocabEntry): VocabEntry object. See vocab.py for documentation.

        Hints: - You may find len(self.vocab.char2id) useful when create the embedding
        """
        super(ModelEmbeddings, self).__init__()

        ### YOUR CODE HERE for part 1h
        self.vocab = vocab # VocabEntry object
        self.word_embed_size = word_embed_size
        self.char_embed_size = char_embed_size
        self.k = k
        self.padding = padding
        self.dropout_rate = dropout_rate
        self.num_char_embeddings = len(vocab.char2id)
        self.char_pad = vocab.char_pad


        ### DEFAULT
        self.char_mbedding = None
        self.cnn = None
        self.highway = None
        self.dropout = None

        ### INITIALIZE VARIABLES
        """
        torch.nn.Embedding(num_embeddings, 
                           embedding_dim, 
                           padding_idx=None, 
                           max_norm=None, 
                           norm_type=2.0, 
                           scale_grad_by_freq=False, 
                           sparse=False, 
                           _weight=None)
        
        Input: (*) , LongTensor of arbitrary shape containing the indices to extract
        Output: (*, H), where * is the input shape and H=embedding_dim
        """
        self.char_embedding = nn.Embedding(num_embeddings = self.num_char_embeddings,
                                           embedding_dim = self.char_embed_size,
                                           padding_idx = self.char_pad
                                          )
        self.cnn = CNN(self.word_embed_size, self.char_embed_size, self.k, self.padding)
        self.highway = Highway(self.word_embed_size)
        self.dropout = nn.Dropout(p=self.dropout_rate)

        ### END YOUR CODE

    def forward(self, input):
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param input: Tensor of integers of shape (sentence_length, batch_size, max_word_length) where
            each integer is an index into the character vocabulary

        @param output: Tensor of shape (sentence_length, batch_size, word_embed_size), containing the
            CNN-based embeddings for each word of the sentences in the batch
        """
        ### YOUR CODE HERE for part 1h

        # [max_sentence_length, b, m_word, char_embed_size]
        X_char_embedding = self.char_embedding(input)

        # X_char_embedding => X_reshaped (b*max_sentence_length, char_embed_size, m_word)
        max_len_sents, b_sents, m_word, char_embed_size = X_char_embedding.shape
        X_reshaped = X_char_embedding.reshape(max_len_sents*b_sents, m_word, char_embed_size)
        X_reshaped = X_reshaped.permute(0, 2, 1).contiguous()

        # X_reshaped => cnn => X_conv_out
        # X_conv_out: [b, word_embed_size]
        X_conv_out = self.cnn(X_reshaped)

        # X_conv_out  => highway => X_highway
        # X_highway: [b, word_embed_size]
        X_highway = self.highway(X_conv_out)

        # X_highway => dropout => X_word_embed
        # X_word_embed: [b, word_embed_size]
        X_word_embed = self.dropout(X_highway)

        # Reshape X_word_embed to sents level, [sentence_length, batch_size, word_embed_size]       
        # b = max_len_sents*b_sents
        X_word_embed = X_word_embed.reshape(max_len_sents, b_sents, -1).contiguous()

        return X_word_embed
        ### END YOUR CODE

