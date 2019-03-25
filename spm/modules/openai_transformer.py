#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author: Shuailong
# @Email: liangshuailong@gmail.com
# @Date:   2019-03-24 12:45:50
# @Last Modified by:  Shuailong
# @Last Modified time: 2019-03-24 15:01:15

"""
An implementation of the OpenAI Transformer Language Model.

Mostly just a slightly modified version of
https://github.com/huggingface/pytorch-openai-transformer-lm
so thanks to them!

Some of these modules duplicate code elsewhere in AllenNLP,
but the serialized weights depend on the exact parameter setup
here, so it's easiest to just reimplement them.
"""

# pylint: disable=invalid-name,arguments-differ
from typing import NamedTuple, List
import copy
import io
import json
import logging
import math
import pathlib
import re
import tarfile

import numpy as np
import torch
from torch.nn import Parameter

from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.common.from_params import FromParams

logger = logging.getLogger(__name__)


def gelu(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


def swish(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


_ACTIVATION_FUNCTIONS = {
    'relu': torch.nn.ReLU,
    'swish': swish,
    'gelu': gelu
}


class TransformerConfig(NamedTuple):
    """
    The transformer has to pass a bunch of params to its submodules,
    this bundles them together to make things easier.
    """
    embedding_dim: int = 768
    num_heads: int = 12
    embedding_dropout_probability: float = 0.1
    attention_dropout_probability: float = 0.1
    residual_dropout_probability: float = 0.1
    activation_function: str = 'gelu'


class LayerNorm(torch.nn.Module):
    "Construct a layernorm module in the OpenAI style (epsilon inside the square root)."

    def __init__(self, n_state, e=1e-5):
        super().__init__()
        self.g = torch.nn.Parameter(torch.ones(n_state))
        self.b = torch.nn.Parameter(torch.zeros(n_state))
        self.e = e

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.e)
        return self.g * x + self.b


class Conv1D(torch.nn.Module):
    def __init__(self, nf: int, rf: int, nx: int) -> None:
        super().__init__()
        self.rf = rf
        self.nf = nf
        if rf == 1:
            w = torch.empty(nx, nf)
            torch.nn.init.normal_(w, std=0.02)
            self.w = Parameter(w)
            self.b = Parameter(torch.zeros(nf))
        else:
            raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.rf == 1:
            size_out = x.size()[:-1] + (self.nf,)
            x = torch.addmm(self.b, x.view(-1, x.size(-1)), self.w)
            x = x.view(*size_out)
        else:
            raise NotImplementedError
        return x


class Attention(torch.nn.Module):
    def __init__(self,
                 nx: int,
                 n_ctx: int,
                 config: TransformerConfig,
                 scale: bool = False) -> None:
        super().__init__()
        n_state = nx  # in Attention: n_state=768 (nx=n_embd)
        # [switch nx => n_state from Block to Attention to keep identical to TF implem]
        assert n_state % config.num_heads == 0
        self.register_buffer('b', torch.tril(
            torch.ones(n_ctx, n_ctx)).view(1, 1, n_ctx, n_ctx))
        self.n_head = config.num_heads
        self.split_size = n_state
        self.scale = scale
        self.c_attn = Conv1D(n_state * 3, 1, nx)
        self.c_proj = Conv1D(n_state, 1, nx)
        self.attn_dropout = torch.nn.Dropout(
            config.attention_dropout_probability)
        self.resid_dropout = torch.nn.Dropout(
            config.residual_dropout_probability)

    def _attn(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        w = torch.matmul(q, k)
        if self.scale:
            w = w / math.sqrt(v.size(-1))
        # TF implem method: mask_attn_weights
        w = w * self.b + -1e9 * (1 - self.b)
        w = torch.nn.Softmax(dim=-1)(w)
        w = self.attn_dropout(w)
        return torch.matmul(w, v)

    def merge_heads(self, x: torch.Tensor):
        # pylint: disable=no-self-use
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)  # in Tensorflow implem: fct merge_states

    def split_heads(self, x: torch.Tensor, k: bool = False):
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(*new_x_shape)  # in Tensorflow implem: fct split_states
        if k:
            return x.permute(0, 2, 3, 1)
        else:
            return x.permute(0, 2, 1, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_attn(x)
        query, key, value = x.split(self.split_size, dim=2)
        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)
        a = self._attn(query, key, value)
        a = self.merge_heads(a)
        a = self.c_proj(a)
        a = self.resid_dropout(a)
        return a


class MLP(torch.nn.Module):
    # in MLP: n_state=3072 (4 * n_embd)
    def __init__(self, n_state: int, config: TransformerConfig) -> None:
        super().__init__()
        self.c_fc = Conv1D(n_state, 1, config.embedding_dim)
        self.c_proj = Conv1D(config.embedding_dim, 1, n_state)
        self.act = _ACTIVATION_FUNCTIONS[config.activation_function]
        self.dropout = torch.nn.Dropout(config.residual_dropout_probability)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.c_fc(x))
        h2 = self.c_proj(h)
        return self.dropout(h2)


class Block(torch.nn.Module):
    def __init__(self,
                 n_ctx: int,
                 config: TransformerConfig,
                 scale: bool = False) -> None:
        super().__init__()
        nx = config.embedding_dim
        self.attn = Attention(nx, n_ctx, config, scale)
        self.ln_1 = LayerNorm(nx)
        self.mlp = MLP(4 * nx, config)
        self.ln_2 = LayerNorm(nx)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.attn(x)
        n = self.ln_1(x + a)
        m = self.mlp(n)
        h = self.ln_2(n + m)
        return h


class OpenaiTransformer(torch.nn.Module, FromParams):
    """
    Openai transformer, as per https://blog.openai.com/language-unsupervised/.
    Default parameters are the ones for their pretrained model.

    Parameters
    ----------
    vocab_size: ``int`` (optional, default: 40478)
            The size of the vocabulary (number of byte pair embeddings)
            excluding the n_special embeddings (if any), and the positional embeddings.
    n_ctx: ``int`` (optional, default: 512)
            The number of positional encodings to use for evaluation.
    embedding_dim: ``int`` (optional, default: 768)
            The dimension of the output embeddings.
    num_heads: ``int`` (optional, default: 12)
            How many "heads" the attention has.
    num_layers: ``int`` (optional, default: 12)
            How many layers of "blocks" the transformer has.
    embedding_dropout_probability: ``float`` (optional, default: 0.1)
            Dropout for the embedding.
    attention_dropout_probability: ``float`` (optional, default: 0.1)
            Dropout for attention.
    residual_dropout_probability: ``float`` (optional, default: 0.1)
            Dropout for residual
    activation_function: ``str`` (optional, default: ``'gelu'``)
            Activation function for the multi-layer perceptron.
    model_path: ``str`` (optional, default: ``None``)
            A tar.gz file containing serialized model weights. If supplied,
            the weights will be loaded from that file.
    requires_grad: ``bool`` (optional, default: ``False``)
            If true, the transformer will be fine-tuneable.
    n_special: ``int`` (optional, default: ``-1``)
            The number of special tokens added to the byte pair vocabulary
            (via ``OpenaiTransformerBytePairIndexer``).
    """

    def __init__(self,
                 n_ctx: int = 512,
                 embedding_dim: int = 768,
                 num_heads: int = 12,
                 num_layers: int = 12,
                 embedding_dropout_probability: float = 0.1,
                 attention_dropout_probability: float = 0.1,
                 residual_dropout_probability: float = 0.1,
                 activation_function: str = 'gelu') -> None:
        super().__init__()

        config = TransformerConfig(
            embedding_dim,
            num_heads,
            embedding_dropout_probability,
            attention_dropout_probability,
            residual_dropout_probability,
            activation_function,
        )

        self.embedding_dim = embedding_dim

        block = Block(n_ctx, config, scale=True)
        self.h = torch.nn.ModuleList(
            [copy.deepcopy(block) for _ in range(num_layers)])

    def get_input_dim(self) -> int:
        return self.embedding_dim

    def get_output_dim(self) -> int:
        return self.embedding_dim

    def forward(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # h is (batch_size, sequence_length, embedding_dim)
        # Shuailnog: ignore mask for now
        all_layers = [h]
        for block in self.h:
            h = block(h)
            all_layers.append(h)

        # result is list of (batch_size, sequence_length, embedding_dim)
        return all_layers[-1]
