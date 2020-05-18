#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
"""

import torch
import torch.nn as nn


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
        x = self.decoderCharEmb(input)
        x, dec_hidden = self.charDecoder(x, dec_hidden) if dec_hidden else self.charDecoder(x)
        scores = self.char_output_projection(x)
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
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        tgt_char_seq = char_sequence[1:, :]
        input_char_seq = []  # input char_sequence without <end> token
        for item in char_sequence:
            vec = [self.target_vocab.char_pad 
                   if char_idx==self.target_vocab.end_of_word else char_idx for char_idx in item]
            input_char_seq.append(torch.tensor(vec, device=device).unsqueeze(0))
        input_char_seq = torch.cat(input_char_seq)[:-1, :]

        target_scores, _ = self.forward(input_char_seq, dec_hidden)
        metric = nn.CrossEntropyLoss(ignore_index=self.target_vocab.char_pad, reduction='sum')
        loss = metric(target_scores.view(-1, len(self.target_vocab.char2id)).contiguous(), tgt_char_seq.view(-1).contiguous())
        
        return loss
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
        batch_size = initialStates[0].size()[1]
        decodedChars = []
        for step in range(max_length):
            if step == 0:
                chars_in = torch.empty(1, batch_size, dtype=torch.long, device=device).fill_(self.target_vocab.start_of_word)
                scores, dec_hidden = self.forward(chars_in, initialStates)
                chars_out = torch.argmax(scores, dim=-1)
            else:
                scores, dec_hidden = self.forward(chars_in, dec_hidden)
                chars_out = torch.argmax(scores, dim=-1)
            chars_in = chars_out
            decodedChars.append([self.target_vocab.id2char[char_idx.item()] for char_idx in chars_out[0]])
        
        decodedWords = []
        for batch_id in range(batch_size):
            word = ''
            for i in range(max_length):
                if self.target_vocab.char2id[decodedChars[i][batch_id]] != self.target_vocab.end_of_word:
                    word = word + decodedChars[i][batch_id]
                else:
                    break
            decodedWords.append(word)
            
        return decodedWords
        ### END YOUR CODE

