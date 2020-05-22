#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from vocab import VocabEntry
import numpy as np
import re


class CharDecoder(nn.Module):
    def __init__(self, hidden_size, char_embedding_size=50, target_vocab=None):
        """ Init Character Decoder.

        @param hidden_size (int): Hidden size of the decoder LSTM
        @param char_embedding_size (int): dimensionality of character embeddings
        @param target_vocab (VocabEntry): vocabulary for the target language. See vocab.py for documentation.
        """
        super(CharDecoder, self).__init__()
        self.target_vocab = target_vocab
        self.charDecoder = nn.LSTM(char_embedding_size, hidden_size)
        self.char_output_projection = nn.Linear(hidden_size, len(self.target_vocab.char2id))
        self.decoderCharEmb = nn.Embedding(len(self.target_vocab.char2id), char_embedding_size,
                                           padding_idx=self.target_vocab.char_pad)

    def forward(self, input, dec_hidden=None):
        """ Forward pass of character decoder.

        @param input (Tensor): tensor of integers, shape (length, batch_size)
        @param dec_hidden (tuple(Tensor, Tensor)): internal state of the LSTM before reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)

        @returns scores (Tensor): called s_t in the PDF, shape (length, batch_size, self.vocab_size)
        @returns dec_hidden (tuple(Tensor, Tensor)): internal state of the LSTM after reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)
        """
        ### YOUR CODE HERE for part 2a
        ### TODO - Implement the forward pass of the character decoder.

        # dec_hidden (tuple(Tensor, Tensor)), each Tensor (1, batch, h)
        
        # input: [length, b] ==> decoderCharEmb => X: [length, b, char_embed_size]
        X = self.decoderCharEmb(input)

        # X: [length, b, char_embed_size], dec_hidden = (h_n, c_n): ([1, b, h], [1, b, h])
        #   ==> charDecoder ==> 
        # h_t: [length, b char_embed_size], dec_hidden = (h_n, c_n): ([1, b, h], [1, b, h])
        h_t, dec_hidden = self.charDecoder(X, dec_hidden)

        # h_t: [length, b char_embed_size] ==> char_output_projection ==> scores = s_t : [length, b, self.vocab_size]
        scores = self.char_output_projection(h_t)
   
        return scores, dec_hidden
        ### END YOUR CODE

    def train_forward(self, char_sequence, dec_hidden=None):
        """ Forward computation during training.

        @param char_sequence (Tensor): tensor of integers, shape (length, batch_size). Note that "length" here and in forward() need not be the same.
        @param dec_hidden (tuple(Tensor, Tensor)): initial internal state of the LSTM, obtained from the output of the word-level decoder. A tuple of two tensors of shape (1, batch_size, hidden_size)

        @returns The cross-entropy loss (Tensor), computed as the *sum* of cross-entropy losses of all the words in the batch.
        """
        ### YOUR CODE HERE for part 2b
        ### TODO - Implement training forward pass.
        ###
        ### Hint: - Make sure padding characters do not contribute to the cross-entropy loss. Check vocab.py to find the padding token's index.
        ###       - char_sequence corresponds to the sequence x_1 ... x_{n+1} (e.g., <START>,m,u,s,i,c,<END>). Read the handout about how to construct input and target sequence of CharDecoderLSTM.
        ###       - Carefully read the documentation for nn.CrossEntropyLoss and our handout to see what this criterion have already included:
        ###             https://pytorch.org/docs/stable/nn.html#crossentropyloss


        # char_sequence: [length, b] => delete end_token => input_sequence: [length, b]
        X_input = char_sequence[:-1]

        # char_sequence: [length, b] => delete start_token => input_sequence: [length, b]
        X_target = char_sequence[1:]

        # X_input: [length, b], dec_hidden = (h_n, c_n): ([1, b, h], [1, b, h])
        #    ==> softmax   ==>
        # s_t: [length, b, self.vocab_size], dec_hidden = (h_n, c_n): ([1, b, h], [1, b, h])
        s_t, dec_hidden = self.forward(X_input, dec_hidden)

        # For lookup char_pad index value, shall be 0
        vocab_entry = VocabEntry()
        idx_char_pad = vocab_entry.char_pad

        # Initialiate CrossEntropyLoss Instances, combines logsoftmax and nllloss
        compute_loss = nn.CrossEntropyLoss(ignore_index = idx_char_pad,
                                           reduction ='sum'
                                          )
        
        # Reshape s_t for compute_loss, length*b => b_char
        # length = length of a word, b = batch size, length*b = # of characters in the batch
        # s_t: [length, b, self.vocab_size] ==> s_t: [length*b, self.vocab_size] = [N, C]
        s_t = s_t.reshape(s_t.shape[0]*s_t.shape[1], -1)

        # Reshape X_target for compute_loss
        # X_target: [length, b] ==> X_target: [length*b] = [N]
        X_target = X_target.reshape(-1)

        # s_t: [length*b, self.vocab_size] = [N, C, d1...dk], X_target: [length*b] = [N]
        #   ==> compute_loss ==> loss_char_dec: 
        loss_char_dec = compute_loss(s_t, X_target) 

        return loss_char_dec
        ### END YOUR CODE

    def decode_greedy(self, initialStates, device, max_length=21):
        """ Greedy decoding
        @param initialStates (tuple(Tensor, Tensor)): initial internal state of the LSTM, a tuple of two tensors of size (1, batch_size, hidden_size)
        @param device: torch.device (indicates whether the model is on CPU or GPU)
        @param max_length (int): maximum length of words to decode

        @returns decodedWords (List[str]): a list (of length batch_size) of strings, each of which has length <= max_length.
                              The decoded strings should NOT contain the start-of-word and end-of-word characters.
        """

        ### YOUR CODE HERE for part 2c
        ### TODO - Implement greedy decoding.
        ### Hints:
        ###      - Use initialStates to get batch_size = b.
        ###      - Use target_vocab.char2id and target_vocab.id2char to convert between integers and characters
        ###      - Use torch.tensor(..., device=device) to turn a list of character indices into a tensor.
        ###      - You may find torch.argmax or torch.argmax useful
        ###      - We use curly brackets as start-of-word and end-of-word characters. That is, use the character '{' for <START> and '}' for <END>.
        ###        Their indices are self.target_vocab.start_of_word and self.target_vocab.end_of_word, respectively.

        ### END YOUR CODE

        # initialStates (tuple(Tensor, Tensor)): ([1, batch_size, hidden_size], [1, batch_size, hidden_size])
        #    ==> read ==> batch_size [int]
        batch_size = initialStates[0].shape[1]

        # iniitalStates ==> dec_hidden = (h0, c0) 
        #    (tuple(Tensor, Tensor)): [1, batch_size, hidden_size], [1, batch_size, hidden_size]
        dec_hidden = initialStates

        # Initialize output_word as an empty, output_word (Tensor): [length <= max_length = 0, batch_size]
        output_word = torch.empty(0, batch_size, dtype=torch.long , device=device)

        # Initiated VocabEntry Instance for character-index lookups
        vocab_entry = VocabEntry() # vocab_entry.start_of_word = index of (<START>='{')

        # Initialize current_char (Tensor): [1, batch_size]
        current_char = torch.tensor([vocab_entry.start_of_word]*batch_size, dtype=torch.long, device=device).reshape(1, -1).contiguous()

        # Keep finding next character, until reaching max-length of word.
        for i in range(0, max_length-1):

            # current_char (Tensor): [1, b], dec_hidden = (h_n, c_n) (tuple(Tensor, Tensor)): ([1, b, h], [1, b, h])
            #     ==> self.forward   ==>
            # s_t (Tensor): [1, b, self.vocab_size], dec_hidden (tuple(Tensor, Tensor)): ([1, b, h], [1, b, h])
            s_t, dec_hidden = self.forward(current_char, dec_hidden)          

            # s_t (Tensor): [1, b, self.vocab_size] ==> softmax ==> p_t (Tensor): [1, b, self.vocab.size]
            p_t = F.softmax(s_t, dim=2)

            # p_t (Tensor): [1, b, self.vocab_size]
            #     ==> argmax ==> current_char (Tensor): [1, b]
            current_char = torch.argmax(p_t, dim=2)

            # current_char (Tensor): [1, b]  ==>  output_word (Tensor): [length <= max_length, b]
            output_word = torch.cat((output_word, current_char), dim=0)

        # output_word (Tensor): [max_length, b] ==> output_word (List(List[int]): [b, max_length]
        output_word = output_word.permute(1,0).tolist()

        # Trucate each word in batch starting from the first end_of_word token <END>='}'
        # output_word (List(List[int]): [b, max_length] ==> output_word (List(List(int))): [b, length <= max_length]
        output_word = [cids[0:cids.index(vocab_entry.end_of_word)] if vocab_entry.end_of_word in cids else cids for cids in output_word]

        # Convert character indices to characters
        # output_word (List(List[int])): [b, length <= max_length] 
        #    ==> (List(List[str])): [b, length <= max_length, str_len=1]
        decodedWords = [[vocab_entry.id2char[cid] for cid in word] for word in output_word]

        # decodedWords (List(List[str])): [b, length <= max_length, str_len = 1]
        #     ==> decodedWords (List[str]): [b, length <= max_length]
        decodedWords = [''.join(char) for char in decodedWords]

        return decodedWords

