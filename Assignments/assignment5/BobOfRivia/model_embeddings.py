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


import torch
import torch.nn as nn

# Do not change these imports; your module names should be
#   `CNN` in the file `cnn.py`
#   `Highway` in the file `highway.py`
# Uncomment the following two imports once you're ready to run part 1(j)

from cnn import CNN
from highway import Highway


# End "do not change"

class ModelEmbeddings(nn.Module):
    """
    Class that converts input words to their CNN-based embeddings.
    """

    def __init__(self, word_embed_size, vocab):
        """
        Init the Embedding layer for one language
        @param word_embed_size (int): Embedding size (dimensionality) for the output word
        @param vocab (VocabEntry): VocabEntry object. See vocab.py for documentation.

        Hints: - You may find len(self.vocab.char2id) useful when create the embedding
        """
        super(ModelEmbeddings, self).__init__()

        ### YOUR CODE HERE for part 1h

        self.word_embed_size = word_embed_size
        self.vocab = vocab

        self.char_emb_size = 50

        # kernal size : 5
        self.cnn = CNN(self.char_emb_size,word_embed_size,5)


        # hiway
        self.hiway = Highway(word_embed_size)


        self.dropout = nn.Dropout(p=0.3)

        # embedding TODO
        self.charEmbedding = torch.randn(len(self.vocab.char2id),self.char_emb_size)


        ### END YOUR CODE

    def forward(self, input:torch.Tensor):
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param input: Tensor of integers of shape (sentence_length, batch_size, max_word_length) where
            each integer is an index into the character vocabulary

        @param output: Tensor of shape (sentence_length, batch_size, word_embed_size), containing the
            CNN-based embeddings for each word of the sentences in the batch
        """
        ### YOUR CODE HERE for part 1h

        (sentence_length, batch_size, max_word_length) = input.shape

        # shape of (batch_size_of_words, max_word_length,char_emb_size)
        inputs = self.charEmbedding[input.contiguous().view(sentence_length * batch_size,max_word_length)]

        # shape of  (batch_size_of_words,word_emb_size)
        cnnd = self.cnn(inputs)

        # print("="*80)
        # print(cnnd.shape)

        # shape of  (batch_size,word_embed_size)
        hiwayd = self.dropout(self.hiway(cnnd))

        return hiwayd.contiguous().view(sentence_length, batch_size, -1)




        ### END YOUR CODE

