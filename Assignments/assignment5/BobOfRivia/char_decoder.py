#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
"""

import torch
import torch.nn as nn
import numpy as np


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

        input = self.decoderCharEmb(input)
        # input must be shape of (seq_len, batch, input_size)

        print("input")
        print(input.shape)


        # char_dec_hidden must be (1, length, batch , hidden_size)
        # print(dec_hidden[0].shape)
        # (seq_len, batch, num_directions * hidden_size)
        char_dec_hidden,dec_hidden = self.charDecoder(input,dec_hidden)

        print("char_dec_hidden")
        print(char_dec_hidden.shape)


        # char_outputs must be (1,length, batch , vocab_size)
        char_outputs = self.char_output_projection(char_dec_hidden)

        print("char_outputs")
        print(char_outputs.shape)

        char_outputs = char_outputs.squeeze()

        return char_outputs,dec_hidden

        ### END YOUR CODE

    def train_forward(self, char_sequence:torch.Tensor, dec_hidden=None):
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
        input = self.decoderCharEmb(char_sequence)

        char_dec_hidden,dec_hidden = self.charDecoder(input,dec_hidden)


        char_outputs = self.char_output_projection(char_dec_hidden)
        # char_outputs must be (1,length, batch , vocab_size)
        char_outputs = char_outputs.squeeze()

        loss = nn.CrossEntropyLoss(reduction='sum')

        emb_size = char_outputs.shape[-1]
        '''
        Input: (N, C)(N,C)
        Target: (N)(N) 
        '''
        Input = (char_outputs[:-1]).contiguous().view(-1,emb_size)
        Target = (char_sequence[1:]).contiguous().view(-1)

        return loss(Input,Target)


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
        ###      - Use initialStates to get batch_size.
        ###      - Use target_vocab.char2id and target_vocab.id2char to convert between integers and characters
        ###      - Use torch.tensor(..., device=device) to turn a list of character indices into a tensor.
        ###      - You may find torch.argmax or torch.argmax useful
        ###      - We use curly brackets as start-of-word and end-of-word characters. That is, use the character '{' for <START> and '}' for <END>.
        ###        Their indices are self.target_vocab.start_of_word and self.target_vocab.end_of_word, respectively.

        '''
        procedure decode greedy
            output_word ← []
            current_char ← <START>
            for t = 0, 1, ..., max length − 1 do
                ht+1, ct+1 ← CharDecoder(current char, ht, ct)
                st+1 ← W_dec * h_t+1 + bdec
                pt+1 ← softmax(s_t+1)
                current char ← argmaxc
                if current char=<END> then
                    break
            output word ← output word + [current char] return output word
        '''

        (h,c) = initialStates
        h=torch.tensor(h,device=device)
        c=torch.tensor(c,device=device)

        batch_size = h.shape[1]

        tensor = torch.ones(())

        # current_char = np.ones((1,batch_size)) * self.target_vocab.char2id['{']


        current_char_idx = tensor.new_full(( 1,batch_size),self.target_vocab.char2id['{'],dtype=torch.long,device=device)

        current_char = ['']*batch_size

        end_char_id =  self.target_vocab.char2id['}']

        for i in range(max_length):

            #  return scores (Tensor):(1, batch_size, self.vocab_size)
            #  return dec_hidden : (tuple(Tensor, Tensor))
            # print(current_char.shape)
            scores,(h,c) = self.forward(current_char_idx,(h,c))

            # (1, batch_size, self.vocab_size)

            print("score")
            print(scores.shape)
            probs = scores.softmax(dim=1)

            # (1, batch_size, 1)
            greedy_char_idx = probs.argmax(dim=1)

            print('greedy_char_idx')

            greedy_char = np.array([self.target_vocab.id2char[int(i)] for i in greedy_char_idx])

            print(greedy_char)

            continue_flags = greedy_char_idx  != end_char_id

            # current_char[continue_flags] += torch.tensor(greedy_char)[continue_flags]

            current_char_idx = (greedy_char_idx[continue_flags]).unsqueeze(dim=0)

            padding_idx = np.argwhere(continue_flags == True)

            for p in padding_idx[0]:
                current_char[p] = greedy_char[p]

        # (1, batch_size,1)
        print(current_char)
        # print(output_str)
        return current_char

        ### END YOUR CODE

