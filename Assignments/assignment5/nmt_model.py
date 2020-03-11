#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
nmt_model.py: NMT Model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
"""
from collections import namedtuple
import sys
from typing import List, Tuple, Dict, Set, Union
import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from model_embeddings import ModelEmbeddings
from char_decoder import CharDecoder

Hypothesis = namedtuple('Hypothesis', ['value', 'score'])

import random


class NMT(nn.Module):
    """ Simple Neural Machine Translation Model:
        - Bidrectional LSTM Encoder
        - Unidirection LSTM Decoder
        - Global Attention Model (Luong, et al. 2015)
    """

    def __init__(self, word_embed_size, hidden_size, vocab, dropout_rate=0.3, no_char_decoder=False):
        """ Init NMT Model.

        @param word_embed_size (int): Embedding size (dimensionality) of word
        @param hidden_size (int): Hidden Size (dimensionality)
        @param vocab (Vocab): Vocabulary object containing src and tgt languages
                              See vocab.py for documentation.
        @param dropout_rate (float): Dropout probability, for attention
        """
        super(NMT, self).__init__()

        self.model_embeddings_source = ModelEmbeddings(word_embed_size, vocab.src)
        self.model_embeddings_target = ModelEmbeddings(word_embed_size, vocab.tgt)

        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.vocab = vocab

        ### COPY OVER YOUR CODE FROM ASSIGNMENT 4

        # default values
        self.encoder = nn.LSTM(word_embed_size, self.hidden_size, bidirectional=True)
        self.decoder = nn.LSTMCell(word_embed_size + self.hidden_size, self.hidden_size)
        self.h_projection = nn.Linear(2*self.hidden_size, self.hidden_size, bias=False)
        self.c_projection = nn.Linear(2*self.hidden_size, self.hidden_size, bias=False)
        self.att_projection = nn.Linear(2*self.hidden_size, self.hidden_size, bias=False)
        self.combined_output_projection = nn.Linear(3*self.hidden_size, self.hidden_size, bias=False)
        self.target_vocab_projection = nn.Linear(self.hidden_size, len(vocab.tgt), bias=False)
        self.dropout = nn.Dropout(p=self.dropout_rate)

        # For sanity check only, not relevant to implementation
        self.gen_sanity_check = False
        self.counter = 0


        ### END YOUR CODE FROM ASSIGNMENT 4

        if not no_char_decoder:
            self.charDecoder = CharDecoder(hidden_size, target_vocab=vocab.tgt)
        else:
            self.charDecoder = None

    def forward(self, source: List[List[str]], target: List[List[str]]) -> torch.Tensor:
        """ Take a mini-batch of source and target sentences, compute the log-likelihood of
        target sentences under the language models learned by the NMT system.

        @param source (List[List[str]]): list of source sentence tokens
        @param target (List[List[str]]): list of target sentence tokens, wrapped by `<s>` and `</s>`

        @returns scores (Tensor): a variable/tensor of one number representing the
                                    log-likelihood of generating the gold-standard target sentence for
                                    each example in the input batch. Here b = batch size.
        """
        # Compute sentence lengths
        source_lengths = [len(s) for s in source]

        # Convert list of lists into tensors

        ### YOUR CODE HERE for part 1i
        ### TODO:
        ###     Modify the code lines above as needed to fetch the character-level tensor
        ###     to feed into encode() and decode(). You should:
        ###     - Keep `target_padded` from A4 code above for predictions
        ###     - Add `source_padded_chars` for character level padded encodings for source
        ###     - Add `target_padded_chars` for character level padded encodings for target
        ###     - Modify calls to encode() and decode() to use the character level encodings

        target_padded = self.vocab.tgt.to_input_tensor(target, device=self.device)   # Tensor: (tgt_len, b)
        source_padded_chars = self.vocab.src.to_input_tensor_char(source, device=self.device)
        target_padded_chars = self.vocab.tgt.to_input_tensor_char(target, device=self.device)

        enc_hiddens, dec_init_state = self.encode(source_padded_chars, source_lengths)
        enc_masks = self.generate_sent_masks(enc_hiddens, source_lengths)
        combined_outputs = self.decode(enc_hiddens, enc_masks, dec_init_state, target_padded_chars)
        ### END YOUR CODE

        P = F.log_softmax(self.target_vocab_projection(combined_outputs), dim=-1)

        # Zero out, probabilities for which we have nothing in the target text
        target_masks = (target_padded != self.vocab.tgt['<pad>']).float()

        # Compute log probability of generating true target words
        target_gold_words_log_prob = torch.gather(P, index=target_padded[1:].unsqueeze(-1), dim=-1).squeeze(
            -1) * target_masks[1:]
        scores = target_gold_words_log_prob.sum()  # mhahn2 Small modification from A4 code.

        if self.charDecoder is not None:
            max_word_len = target_padded_chars.shape[-1]

            target_words = target_padded[1:].contiguous().view(-1)
            target_chars = target_padded_chars[1:].contiguous().view(-1, max_word_len)
            target_outputs = combined_outputs.view(-1, 256)

            target_chars_oov = target_chars  # torch.index_select(target_chars, dim=0, index=oovIndices)
            rnn_states_oov = target_outputs  # torch.index_select(target_outputs, dim=0, index=oovIndices)
            oovs_losses = self.charDecoder.train_forward(target_chars_oov.t().contiguous(),
                                                         (rnn_states_oov.unsqueeze(0), rnn_states_oov.unsqueeze(0)))
            scores = scores - oovs_losses

        return scores

    def encode(self, source_padded: torch.Tensor, source_lengths: List[int]) -> Tuple[
        torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """ Apply the encoder to source sentences to obtain encoder hidden states.
            Additionally, take the final states of the encoder and project them to obtain initial states for decoder.
        @param source_padded (Tensor): Tensor of padded source sentences with shape (src_len, b, max_word_length), where
                                        b = batch_size, src_len = maximum source sentence length. Note that
                                       these have already been sorted in order of longest to shortest sentence.
        @param source_lengths (List[int]): List of actual lengths for each of the source sentences in the batch
        @returns enc_hiddens (Tensor): Tensor of hidden units with shape (b, src_len, h*2), where
                                        b = batch size, src_len = maximum source sentence length, h = hidden size.
        @returns dec_init_state (tuple(Tensor, Tensor)): Tuple of tensors representing the decoder's initial
                                                hidden state and cell.
        """
        enc_hiddens, dec_init_state = None, None

        ### COPY OVER YOUR CODE FROM ASSIGNMENT 4
        ### Except replace "self.model_embeddings.source" with "self.model_embeddings_source"
        # 通过对字符向量卷积得到输入的特征词向量
        X = self.model_embeddings_source(source_padded)             # src_len, batch_size, embed
        X = pack_padded_sequence(X, lengths=source_lengths)         #
        enc_hiddens, (last_hidden, last_cell) = self.encoder(X)     # 各层隐层，LSTM输出h_t,c_t
        enc_hiddens, length = pad_packed_sequence(enc_hiddens)      # src_len, b, h*2
        enc_hiddens = enc_hiddens.permute(1, 0, 2)                  # b, src_len, h*2->src_len, b, h*2,,使用tensor.transpose()也能达到效果。
                                                                    # 区别是transpose只能进行二维的转换, permute可以高维


        init_decoder_hidden = self.h_projection(
            torch.cat(
                (last_hidden[0], last_hidden[1]), 1)
        )                                                           # (b, h) -> (h, 2h)


        init_decoder_cell = self.c_projection(
            torch.cat(
                (last_cell[0], last_cell[1]), 1)
        )                                                           # (b, h) -> (h, 2h)

        dec_init_state = (init_decoder_hidden, init_decoder_cell)
        ### END YOUR CODE FROM ASSIGNMENT 4

        return enc_hiddens, dec_init_state

    def decode(self, enc_hiddens: torch.Tensor, enc_masks: torch.Tensor,
               dec_init_state: Tuple[torch.Tensor, torch.Tensor], target_padded: torch.Tensor) -> torch.Tensor:
        """Compute combined output vectors for a batch.
        @param enc_hiddens (Tensor): Hidden states (b, src_len, h*2), where
                                     b = batch size, src_len = maximum source sentence length, h = hidden size.
        @param enc_masks (Tensor): Tensor of sentence masks (b, src_len), where
                                     b = batch size, src_len = maximum source sentence length.
        @param dec_init_state (tuple(Tensor, Tensor)): Initial state and cell for decoder
        @param target_padded (Tensor): Gold-standard padded target sentences (tgt_len, b, max_word_length), where
                                       tgt_len = maximum target sentence length, b = batch size.
        @returns combined_outputs (Tensor): combined output tensor  (tgt_len, b,  h), where
                                        tgt_len = maximum target sentence length, b = batch_size,  h = hidden size
        """
        # Chop of the <END> token for max length sentences.
        target_padded = target_padded[:-1]

        # Initialize the decoder state (hidden and cell)
        dec_state = dec_init_state

        # Initialize previous combined output vector o_{t-1} as zeros
        batch_size = enc_hiddens.size(0)
        o_prev = torch.zeros(batch_size, self.hidden_size, device=self.device)

        # Initialize a list we will use to collect the combined output o_t on each step
        combined_outputs = []

        ### COPY OVER YOUR CODE FROM ASSIGNMENT 4
        ### Except replace "self.model_embeddings.target" with "self.model_embeddings_target"
        enc_hiddens_proj = self.att_projection(enc_hiddens)
        Y = self.model_embeddings_target(target_padded)

        for i in torch.split(Y, 1, dim=0):
            Y_t = i.squeeze(0)
            Ybar_t = torch.cat((Y_t, o_prev), 1)
            dec_state, o_t, e_t = self.step(Ybar_t, dec_state, enc_hiddens, enc_hiddens_proj, enc_masks)
            combined_outputs.append(o_t)
            o_prev = o_t

        combined_outputs = torch.stack(combined_outputs, dim=0)
        ### END YOUR CODE FROM ASSIGNMENT 4

        return combined_outputs

    def step(self, Ybar_t: torch.Tensor,
             dec_state: Tuple[torch.Tensor, torch.Tensor],
             enc_hiddens: torch.Tensor,
             enc_hiddens_proj: torch.Tensor,
             enc_masks: torch.Tensor) -> Tuple[Tuple, torch.Tensor, torch.Tensor]:
        """ Compute one forward step of the LSTM decoder, including the attention computation.
        @param Ybar_t (Tensor): Concatenated Tensor of [Y_t o_prev], with shape (b, e + h). The input for the decoder,
                                where b = batch size, e = embedding size, h = hidden size.
        @param dec_state (tuple(Tensor, Tensor)): Tuple of tensors both with shape (b, h), where b = batch size, h = hidden size.
                First tensor is decoder's prev hidden state, second tensor is decoder's prev cell.
        @param enc_hiddens (Tensor): Encoder hidden states Tensor, with shape (b, src_len, h * 2), where b = batch size,
                                    src_len = maximum source length, h = hidden size.
        @param enc_hiddens_proj (Tensor): Encoder hidden states Tensor, projected from (h * 2) to h. Tensor is with shape (b, src_len, h),
                                    where b = batch size, src_len = maximum source length, h = hidden size.
        @param enc_masks (Tensor): Tensor of sentence masks shape (b, src_len),
                                    where b = batch size, src_len is maximum source length.
        @returns dec_state (tuple (Tensor, Tensor)): Tuple of tensors both shape (b, h), where b = batch size, h = hidden size.
                First tensor is decoder's new hidden state, second tensor is decoder's new cell.
        @returns combined_output (Tensor): Combined output Tensor at timestep t, shape (b, h), where b = batch size, h = hidden size.
        @returns e_t (Tensor): Tensor of shape (b, src_len). It is attention scores distribution.
                                Note: You will not use this outside of this function.
                                      We are simply returning this value so that we can sanity check
                                      your implementation.
        """

        combined_output = None

        ### COPY OVER YOUR CODE FROM ASSIGNMENT 4
        dec_state = self.decoder(Ybar_t, dec_state)
        dec_hidden, dec_cell = dec_state
        e_t = torch.bmm(
            enc_hiddens_proj,                   # (b, src_len, h)
            dec_hidden.unsqueeze(2)             # (b, h, 1)
        ).squeeze(2)                            # (b, src_len)
        ### END YOUR CODE FROM ASSIGNMENT 4

        # Set e_t to -inf where enc_masks has 1
        if enc_masks is not None:
            e_t.data.masked_fill_(enc_masks.bool(), -float('inf'))

        ### COPY OVER YOUR CODE FROM ASSIGNMENT 4
        alpha_t = F.softmax(e_t, dim=1)
        a_t = torch.bmm(
            alpha_t.unsqueeze(1),               # (b, 1, src_len)
            enc_hiddens                         # (b, src_len, 2h)
        ).squeeze(1)                            # (b, 1, 2h) ->(b, 2h)
        U_t = torch.cat((dec_hidden, a_t), 1)   # (b, 3h)
        V_t = self.combined_output_projection(U_t)
        O_t = self.dropout(torch.tanh(V_t))
        ### END YOUR CODE FROM ASSIGNMENT 4

        combined_output = O_t
        return dec_state, combined_output, e_t

    def generate_sent_masks(self, enc_hiddens: torch.Tensor, source_lengths: List[int]) -> torch.Tensor:
        """ Generate sentence masks for encoder hidden states.

        @param enc_hiddens (Tensor): encodings of shape (b, src_len, 2*h), where b = batch size,
                                     src_len = max source length, h = hidden size.
        @param source_lengths (List[int]): List of actual lengths for each of the sentences in the batch.

        @returns enc_masks (Tensor): Tensor of sentence masks of shape (b, src_len),
                                    where src_len = max source length, h = hidden size.
        """
        enc_masks = torch.zeros(enc_hiddens.size(0), enc_hiddens.size(1), dtype=torch.float)
        for e_id, src_len in enumerate(source_lengths):
            enc_masks[e_id, src_len:] = 1
        return enc_masks.to(self.device)

    def beam_search(self, src_sent: List[str], beam_size: int = 5, max_decoding_time_step: int = 70) -> List[
        Hypothesis]:
        """ Given a single source sentence, perform beam search, yielding translations in the target language.
        @param src_sent (List[str]): a single source sentence (words)
        @param beam_size (int): beam size
        @param max_decoding_time_step (int): maximum number of time steps to unroll the decoding RNN
        @returns hypotheses (List[Hypothesis]): a list of hypothesis, each hypothesis has two fields:
                value: List[str]: the decoded target sentence, represented as a list of words
                score: float: the log-likelihood of the target sentence
        """

        # 字符型tensor,以最大单词字母数为单位，不足用0进行填充。size(max_sentence_length, batch_size, max_word_length)
        src_sents_var = self.vocab.src.to_input_tensor_char([src_sent], self.device)

        # LSTM输出隐状态dec和历史h_t矩阵
        src_encodings, dec_init_vec = self.encode(src_sents_var, [len(src_sent)])
        src_encodings_att_linear = self.att_projection(src_encodings)

        # 字符型LSTM输出的隐状态。
        h_tm1 = dec_init_vec
        att_tm1 = torch.zeros(1, self.hidden_size, device=self.device)

        eos_id = self.vocab.tgt['</s>']

        hypotheses = [['<s>']]
        hyp_scores = torch.zeros(len(hypotheses), dtype=torch.float, device=self.device)
        completed_hypotheses = []

        t = 0
        while len(completed_hypotheses) < beam_size and t < max_decoding_time_step:
            t += 1
            # 已翻译词作为输入文本
            hyp_num = len(hypotheses)

            # 附加新输入词空间
            exp_src_encodings = src_encodings.expand(hyp_num,
                                                     src_encodings.size(1),
                                                     src_encodings.size(2))
            exp_src_encodings_att_linear = src_encodings_att_linear.expand(hyp_num,
                                                                           src_encodings_att_linear.size(1),
                                                                           src_encodings_att_linear.size(2))

            # 已翻译文本中,以词为输入矩阵(sentence_length,batch_size,max_word_size)
            y_tm1 = self.vocab.tgt.to_input_tensor_char(list([hyp[-1]] for hyp in hypotheses), device=self.device)
            # 以词为单位，对每个词内每个字符替换成字符向量
            y_t_embed = self.model_embeddings_target(y_tm1)
            y_t_embed = torch.squeeze(y_t_embed, dim=0)

            x = torch.cat([y_t_embed, att_tm1], dim=-1)

            # 计算一次模型翻译输出,att_t即为o_t。
            (h_t, cell_t), att_t, _ = self.step(x, h_tm1,
                                                exp_src_encodings, exp_src_encodings_att_linear, enc_masks=None)

            # log probabilities over target words
            log_p_t = F.log_softmax(self.target_vocab_projection(att_t), dim=-1)

            live_hyp_num = beam_size - len(completed_hypotheses)
            contiuating_hyp_scores = (hyp_scores.unsqueeze(1).expand_as(log_p_t) + log_p_t).view(-1)
            # 取
            top_cand_hyp_scores, top_cand_hyp_pos = torch.topk(contiuating_hyp_scores, k=live_hyp_num)

            prev_hyp_ids = top_cand_hyp_pos / len(self.vocab.tgt)
            hyp_word_ids = top_cand_hyp_pos % len(self.vocab.tgt)

            new_hypotheses = []
            live_hyp_ids = []
            new_hyp_scores = []

            decoderStatesForUNKsHere = []
            for prev_hyp_id, hyp_word_id, cand_new_hyp_score in zip(prev_hyp_ids, hyp_word_ids, top_cand_hyp_scores):
                prev_hyp_id = prev_hyp_id.item()
                hyp_word_id = hyp_word_id.item()
                cand_new_hyp_score = cand_new_hyp_score.item()

                hyp_word = self.vocab.tgt.id2word[hyp_word_id]

                # Record output layer in case UNK was generated
                # 不同处, 预测的词出现了字典里没有的词,进行标记处理。
                if hyp_word == "<unk>":
                    hyp_word = "<unk>" + str(len(decoderStatesForUNKsHere))
                    # 将这次的 dec_t 送入字符向量 decoder 求得的 o_t 取出, 使用decode_greedy算法进行单词生成
                    decoderStatesForUNKsHere.append(att_t[prev_hyp_id])

                new_hyp_sent = hypotheses[prev_hyp_id] + [hyp_word]
                if hyp_word == '</s>':
                    completed_hypotheses.append(Hypothesis(value=new_hyp_sent[1:-1],
                                                           score=cand_new_hyp_score))
                else:
                    new_hypotheses.append(new_hyp_sent)
                    live_hyp_ids.append(prev_hyp_id)
                    new_hyp_scores.append(cand_new_hyp_score)


            # 和词向量翻译不同的地方
            if len(decoderStatesForUNKsHere) > 0 and self.charDecoder is not None:  # decode UNKs
                decoderStatesForUNKsHere = torch.stack(decoderStatesForUNKsHere, dim=0)
                # 通过字符向量计算字典里没有的词
                decodedWords = self.charDecoder.decode_greedy(
                    (decoderStatesForUNKsHere.unsqueeze(0), decoderStatesForUNKsHere.unsqueeze(0)), max_length=21,
                    device=self.device)
                assert len(decodedWords) == decoderStatesForUNKsHere.size()[0], "Incorrect number of decoded words"
                for hyp in new_hypotheses:
                    if hyp[-1].startswith("<unk>"):
                        hyp[-1] = decodedWords[int(hyp[-1][5:])]   # 预测词替换<unk>符号

            if len(completed_hypotheses) == beam_size:
                break

            live_hyp_ids = torch.tensor(live_hyp_ids, dtype=torch.long, device=self.device)
            h_tm1 = (h_t[live_hyp_ids], cell_t[live_hyp_ids])
            att_tm1 = att_t[live_hyp_ids]

            hypotheses = new_hypotheses
            hyp_scores = torch.tensor(new_hyp_scores, dtype=torch.float, device=self.device)

        if len(completed_hypotheses) == 0:
            completed_hypotheses.append(Hypothesis(value=hypotheses[0][1:],
                                                   score=hyp_scores[0].item()))
        # 按照分数排序的结果
        completed_hypotheses.sort(key=lambda hyp: hyp.score, reverse=True)
        return completed_hypotheses

    @property
    def device(self) -> torch.device:
        """ Determine which device to place the Tensors upon, CPU or GPU.
        """
        return self.att_projection.weight.device

    @staticmethod
    def load(model_path: str, no_char_decoder=False):
        """ Load the model from a file.
        @param model_path (str): path to model
        """
        params = torch.load(model_path, map_location=lambda storage, loc: storage)
        args = params['args']
        model = NMT(vocab=params['vocab'], no_char_decoder=no_char_decoder, **args)
        model.load_state_dict(params['state_dict'])

        return model

    def save(self, path: str):
        """ Save the odel to a file.
        @param path (str): path to the model
        """
        print('save model parameters to [%s]' % path, file=sys.stderr)

        params = {
            'args': dict(word_embed_size=self.model_embeddings_source.word_embed_size, hidden_size=self.hidden_size,
                         dropout_rate=self.dropout_rate),
            'vocab': self.vocab,
            'state_dict': self.state_dict()
        }

        torch.save(params, path)
