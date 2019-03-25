#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author: https://github.com/fastnlp/fastNLP/blob/master/fastNLP/modules/encoder/transformer.py
# @Email: liangshuailong@gmail.com
# @Date:   2019-03-23 11:56:53
# @Last Modified by:  Shuailong
# @Last Modified time: 2019-03-23 21:44:12

from overrides import overrides
import math
import torch
from torch import nn

from allennlp.common.registrable import FromParams


class TransformerEncoder(nn.Module, FromParams):
    """transformer的encoder模块，不包含embedding层
    :param num_layers: int, transformer的层数
    :param model_size: int, 输入维度的大小。同时也是输出维度的大小。
    :param inner_size: int, FFN层的hidden大小
    :param key_size: int, 每个head的维度大小。
    :param value_size: int，每个head中value的维度。
    :param num_head: int，head的数量。
    :param dropout: float。
    """

    def __init__(self, num_layers, model_size, inner_size, key_size, value_size, num_head, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList(
            [self.SubLayer(model_size, inner_size, key_size, value_size, num_head, dropout)
             for _ in range(num_layers)])
        self.model_size = model_size

    def get_input_dim(self):
        return self.model_size

    def get_output_dim(self):
        return self.model_size

    @overrides
    def forward(self, x, seq_mask=None):
        """
        :param x: [batch, seq_len, model_size] 输入序列
        :param seq_mask: [batch, seq_len] 输入序列的padding mask
        :return: [batch, seq_len, model_size] 输出序列
        """
        output = x
        if seq_mask is None:
            atte_mask_out = None
        else:
            atte_mask_out = (seq_mask < 1)[:, None, :]
            seq_mask = seq_mask[:, :, None]
        for layer in self.layers:
            output = layer(output, seq_mask, atte_mask_out)
        return output

    class SubLayer(nn.Module):
        def __init__(self, model_size, inner_size, key_size, value_size, num_head, dropout=0.1):
            super(TransformerEncoder.SubLayer, self).__init__()
            self.atte = MultiHeadAtte(
                model_size, key_size, value_size, num_head, dropout)
            self.norm1 = nn.LayerNorm(model_size)
            self.ffn = nn.Sequential(nn.Linear(model_size, inner_size),
                                     nn.ReLU(),
                                     nn.Linear(inner_size, model_size),
                                     TimestepDropout(dropout),)
            self.norm2 = nn.LayerNorm(model_size)

        def forward(self, input, seq_mask=None, atte_mask_out=None):
            """
            :param input: [batch, seq_len, model_size]
            :param seq_mask: [batch, seq_len]
            :return: [batch, seq_len, model_size]
            """
            attention = self.atte(input, input, input, atte_mask_out)
            norm_atte = self.norm1(attention + input)
            attention *= seq_mask
            output = self.ffn(norm_atte)
            output = self.norm2(output + norm_atte)
            output *= seq_mask
            return output


class DotAtte(nn.Module):
    def __init__(self, key_size, value_size, dropout=0.1):
        super(DotAtte, self).__init__()
        self.key_size = key_size
        self.value_size = value_size
        self.scale = math.sqrt(key_size)
        self.drop = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, Q, K, V, mask_out=None):
        """
        :param Q: [batch, seq_len, key_size]
        :param K: [batch, seq_len, key_size]
        :param V: [batch, seq_len, value_size]
        :param mask_out: [batch, seq_len]
        """
        output = torch.matmul(Q, K.transpose(1, 2)) / self.scale
        if mask_out is not None:
            output.masked_fill_(mask_out, -float('inf'))
        output = self.softmax(output)
        output = self.drop(output)
        return torch.matmul(output, V)


class MultiHeadAtte(nn.Module):
    def __init__(self, input_size, key_size, value_size, num_head, dropout=0.1):
        """
        :param input_size: int, 输入维度的大小。同时也是输出维度的大小。
        :param key_size: int, 每个head的维度大小。
        :param value_size: int，每个head中value的维度。
        :param num_head: int，head的数量。
        :param dropout: float。
        """
        super(MultiHeadAtte, self).__init__()
        self.input_size = input_size
        self.key_size = key_size
        self.value_size = value_size
        self.num_head = num_head

        in_size = key_size * num_head
        self.q_in = nn.Linear(input_size, in_size)
        self.k_in = nn.Linear(input_size, in_size)
        self.v_in = nn.Linear(input_size, in_size)
        self.attention = DotAtte(key_size=key_size, value_size=value_size)
        self.out = nn.Linear(value_size * num_head, input_size)
        self.drop = TimestepDropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        sqrt = math.sqrt
        nn.init.normal_(self.q_in.weight, mean=0, std=sqrt(
            2.0 / (self.input_size + self.key_size)))
        nn.init.normal_(self.k_in.weight, mean=0, std=sqrt(
            2.0 / (self.input_size + self.key_size)))
        nn.init.normal_(self.v_in.weight, mean=0, std=sqrt(
            2.0 / (self.input_size + self.value_size)))
        nn.init.xavier_normal_(self.out.weight)

    def forward(self, Q, K, V, atte_mask_out=None):
        """
        :param Q: [batch, seq_len, model_size]
        :param K: [batch, seq_len, model_size]
        :param V: [batch, seq_len, model_size]
        :param seq_mask: [batch, seq_len]
        """
        batch, seq_len, _ = Q.size()
        d_k, d_v, n_head = self.key_size, self.value_size, self.num_head
        # input linear
        q = self.q_in(Q).view(batch, seq_len, n_head, d_k)
        k = self.k_in(K).view(batch, seq_len, n_head, d_k)
        v = self.v_in(V).view(batch, seq_len, n_head, d_k)

        # transpose q, k and v to do batch attention
        q = q.permute(2, 0, 1, 3).contiguous().view(-1, seq_len, d_k)
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, seq_len, d_k)
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, seq_len, d_v)
        if atte_mask_out is not None:
            atte_mask_out = atte_mask_out.repeat(n_head, 1, 1)
        atte = self.attention(q, k, v, atte_mask_out).view(
            n_head, batch, seq_len, d_v)

        # concat all heads, do output linear
        atte = atte.permute(1, 2, 0, 3).contiguous().view(batch, seq_len, -1)
        output = self.drop(self.out(atte))
        return output


class TimestepDropout(torch.nn.Dropout):
    """This module accepts a ``[batch_size, num_timesteps, embedding_dim)]`` and use a single
    dropout mask of shape ``(batch_size, embedding_dim)`` to apply on every time step.
    """

    def forward(self, x):
        dropout_mask = x.new_ones(x.shape[0], x.shape[-1])
        torch.nn.functional.dropout(
            dropout_mask, self.p, self.training, inplace=True)
        dropout_mask = dropout_mask.unsqueeze(
            1)  # [batch_size, 1, embedding_dim]
        if self.inplace:
            x *= dropout_mask
            return
        else:
            return x * dropout_mask
