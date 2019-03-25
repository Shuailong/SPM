#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author: Shuailong
# @Email: liangshuailong@gmail.com
# @Date:   2019-03-16 11:38:55
# @Last Modified by:  Shuailong
# @Last Modified time: 2019-03-24 15:52:57

'''
Implementation of the paper: Sentence-State LSTM for Text Representation
(https://arxiv.org/abs/1805.02474)
Modified according to CONLL 2003 NER dataset
'''

from overrides import overrides

from torch import nn
import torch
import torch.nn.functional as F

from allennlp.nn.initializers import block_orthogonal
from allennlp.common.registrable import FromParams
from allennlp.modules import LayerNorm
from spm.modules.utils import mean_with_mask


class SLSTMEncoder(nn.Module, FromParams):

    def __init__(self,
                 hidden_size: int,
                 num_layers: int = 7,
                 ):
        super(SLSTMEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # h_t updates
        self.h_context_linearity = torch.nn.Linear(
            2 * hidden_size, 7 * hidden_size, bias=False)
        self.h_current_linearity = torch.nn.Linear(
            hidden_size, 7 * hidden_size, bias=False)
        self.h_input_linearity = torch.nn.Linear(
            hidden_size, 7 * hidden_size, bias=True)
        self.h_global_linearity = torch.nn.Linear(
            hidden_size, 7 * hidden_size, bias=False)

        # global updates
        self.g_input_linearity = torch.nn.Linear(
            hidden_size, 3 * hidden_size, bias=True)
        self.g_hidden_linearity = torch.nn.Linear(
            hidden_size, hidden_size, bias=False)
        self.g_avg_linearity = torch.nn.Linear(
            hidden_size, 2 * hidden_size, bias=False)

        # layer normalization layer
        self.layer_norms = torch.nn.ModuleList(
            [LayerNorm(hidden_size) for _ in range(10)])

        self.reset_parameters()

    def reset_parameters(self):
        # Use sensible default initializations for parameters.
        block_orthogonal(self.h_context_linearity.weight.data,
                         [self.hidden_size, self.hidden_size])
        block_orthogonal(self.h_input_linearity.weight.data, [
            self.hidden_size, self.hidden_size])
        block_orthogonal(self.h_global_linearity.weight.data,
                         [self.hidden_size, self.hidden_size])

        block_orthogonal(self.g_input_linearity.weight.data,
                         [self.hidden_size, self.hidden_size])
        block_orthogonal(self.g_hidden_linearity.weight.data,
                         [self.hidden_size, self.hidden_size])
        block_orthogonal(self.g_avg_linearity.weight.data, [
            self.hidden_size, self.hidden_size])

        self.h_input_linearity.bias.data.fill_(0.0)
        self.g_input_linearity.bias.data.fill_(0.0)

    def get_input_dim(self):
        return self.hidden_size

    def get_output_dim(self):
        return self.hidden_size

    @overrides
    def forward(self,  # type: ignore
                inputs: torch.FloatTensor,
                mask: torch.FloatTensor):
        """
        Parameters
        ----------
        inputs : ``torch.FloatTensor``
            A tensor of shape (batch_size, seq_len, hidden_size)
        mask : ``torch.FloatTensor``
            A tensor of shape (batch_size, seq_len)
        Returns
        -------
        An output dictionary consisting of:
        hiddens: ``torch.FloatTensor``
            A tensor of shape (batch_size, seq_len, hidden_size)
        global_hiddens: ``torch.FloatTensor``
            A tensor of shape (batch_size, hidden_size)
        """
        batch_size, _, hidden_size = inputs.size()

        # filters for attention
        mask_softmax_score = mask * 1e25 - 1e25
        mask_softmax_score = mask_softmax_score.unsqueeze(-1)

        ############################################################################
        # Init states
        ############################################################################
        # randomly initialize the states
        hidden = torch.rand_like(inputs) - 0.5
        cell = torch.rand_like(inputs) - 0.5

        global_hidden = mean_with_mask(hidden, mask)
        global_cell = mean_with_mask(cell, mask)

        for _ in range(self.num_layers):
            #############################
            # update global node states #
            #############################
            hidden_avg = mean_with_mask(hidden, mask)
            projected_input = self.g_input_linearity(global_hidden)
            projected_hiddens = self.g_hidden_linearity(hidden)
            projected_avg = self.g_avg_linearity(hidden_avg)

            input_gate = torch.sigmoid(
                self.layer_norms[0](projected_input[:, 0 * hidden_size: 1 * hidden_size] +
                                    projected_avg[:, 0 * hidden_size: 1 * hidden_size]))
            hidden_gates = torch.sigmoid(
                self.layer_norms[1](projected_input[:, 1 * hidden_size: 2 * hidden_size].unsqueeze(1).expand_as(hidden) +
                                    projected_hiddens))
            output_gate = torch.sigmoid(
                self.layer_norms[2](projected_input[:, 2 * hidden_size: 3 * hidden_size] +
                                    projected_avg[:, 1 * hidden_size: 2 * hidden_size]))

            # softmax on each hidden dimension
            hidden_gates = hidden_gates + mask_softmax_score
            # Combine
            gates_normalized = F.softmax(
                torch.cat([torch.unsqueeze(input_gate, dim=1), hidden_gates], dim=1), dim=1)

            # split the softmax scores
            input_gate_normalized = gates_normalized[:, 0, :]
            hidden_gates_normalized = gates_normalized[:, 1:, :]

            # new global states
            global_cell = (hidden_gates_normalized * cell).sum(1) + \
                global_cell * input_gate_normalized
            global_hidden = output_gate * torch.tanh(global_cell)

            #############################
            # update hidden node states #
            #############################

            # Note: add <bos> and <eos> before hand in case that the valid words are omitted!
            hidden_l = torch.cat(
                [hidden.new_zeros(batch_size, 1, hidden_size), hidden[:, :-1, :]], dim=1)
            hidden_r = torch.cat(
                [hidden[:, 1:, :], hidden.new_zeros(batch_size, 1, hidden_size)], dim=1)
            cell_l = torch.cat(
                [cell.new_zeros(batch_size, 1, hidden_size), cell[:, :-1, :]], dim=1)
            cell_r = torch.cat(
                [cell[:, 1:, :], cell.new_zeros(batch_size, 1, hidden_size)], dim=1)

            # concat with neighbors
            contexts = torch.cat([hidden_l, hidden_r], dim=-1)

            projected_contexts = self.h_context_linearity(contexts)
            projected_current = self.h_current_linearity(hidden)
            projected_input = self.h_input_linearity(inputs)
            projected_global = self.h_global_linearity(global_hidden)

            gates = []
            for offset in range(6):
                gates.append(torch.sigmoid(
                    self.layer_norms[offset + 3](projected_contexts[..., offset * hidden_size:(offset + 1) * hidden_size] +
                                                 projected_current[..., offset * hidden_size:(offset + 1) * hidden_size] +
                                                 projected_input[..., offset * hidden_size:(offset + 1) * hidden_size] +
                                                 projected_global[..., offset * hidden_size:(offset + 1) * hidden_size].unsqueeze(1).expand_as(inputs))))
            memory_init = torch.tanh(
                self.layer_norms[-1](projected_contexts[..., 6 * hidden_size:7 * hidden_size] +
                                     projected_current[..., 6 * hidden_size: 7 * hidden_size] +
                                     projected_input[..., 6 * hidden_size:7 * hidden_size] +
                                     projected_global[..., 6 * hidden_size:7 * hidden_size].unsqueeze(1).expand_as(inputs)))

            # gate: batch x seq_len x hidden_size
            gates_normalized = F.softmax(torch.stack(gates), dim=0)
            input_gate = gates_normalized[0, ...]
            left_gate = gates_normalized[1, ...]
            right_gate = gates_normalized[2, ...]
            forget_gate = gates_normalized[3, ...]
            global_gate = gates_normalized[4, ...]
            output_gate = gates_normalized[5, ...]

            cell = left_gate * cell_l +\
                right_gate * cell_r +\
                forget_gate * cell +\
                input_gate * memory_init +\
                global_gate * global_cell.unsqueeze(1).expand_as(global_gate)

            hidden = output_gate * torch.tanh(cell)

            hidden = hidden * mask.unsqueeze(-1)
            cell = cell * mask.unsqueeze(-1)

        # output_dict = {
        #     'hiddens': hidden,
        #     'global_hidden': global_hidden
        # }

        return hidden
