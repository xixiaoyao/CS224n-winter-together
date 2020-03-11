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
import torch.nn.functional as F
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
        self.char_embed_size = 50
        self.max_word_len = 21
        self.dropout_rate = 0.3
        self.vocab = vocab
        self.vocab_char2id = len(self.vocab.char2id)
        self.pad_token_idx = vocab.char2id['<pad>']

        self.X_char_emb = nn.Embedding(
                                        self.vocab_char2id,
                                        self.char_embed_size,
                                        self.pad_token_idx
                                       )


        self.CNN_model = CNN(
                            e_char=self.char_embed_size,
                            e_word=self.word_embed_size,
                            kernel_size=5,
                            max_word_len=self.max_word_len
                            )

        self.Highway_model = Highway(self.word_embed_size)



        ### END YOUR CODE

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param input: Tensor of integers of shape (sentence_length, batch_size, max_word_length) where
            each integer is an index into the character vocabulary

        @param output: Tensor of shape (sentence_length, batch_size, word_embed_size), containing the
            CNN-based embeddings for each word of the sentences in the batch
        """
        ### YOUR CODE HERE for part 1h


        X = self.X_char_emb(input)

        sent_len, batch_size, max_word, char_embed_size = X.shape
        view_shape = (sent_len*batch_size, max_word, char_embed_size)

        X_reshaped = X.view(view_shape).transpose(1, 2)
        # 卷积
        X_conv_out = self.CNN_model(X_reshaped).squeeze(2)

        X_highway = self.Highway_model(X_conv_out)
        X_word_emb = F.dropout(X_highway, self.dropout_rate)
        X_word_emb = X_word_emb.view(sent_len, batch_size, self.word_embed_size)

        return X_word_emb

        ### END YOUR CODE

