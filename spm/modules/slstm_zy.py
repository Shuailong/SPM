#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author: Shuailong
# @Email: liangshuailong@gmail.com
# @Date:   2019-03-16 11:38:55
# @Last Modified by:  Shuailong
# @Last Modified time: 2019-03-20 15:57:23

'''
Adapted from zeeeyang's implementation.
'''

from overrides import overrides

from torch import nn
import torch
import torch.nn.functional as F
import torch.nn.init as init

from allennlp.common.registrable import FromParams
from allennlp.modules import LayerNorm


class SLSTMEncoderZY(nn.Module, FromParams):

    def __init__(self,
                 hidden_size: int,
                 SLSTM_step: int = 1,
                 num_layers: int = 7,
                 dropout: int = 0.5,
                 ):
        super(SLSTMEncoderZY, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.SLSTM_step = SLSTM_step

        # forget gate for left
        self.Wxf1, self.Whf1, self.Wif1, self.Wdf1 = self.create_a_lstm_gate(
            hidden_size)
        # forget gate for right
        self.Wxf2, self.Whf2, self.Wif2, self.Wdf2 = self.create_a_lstm_gate(
            hidden_size)
        # forget gate for inital states
        self.Wxf3, self.Whf3, self.Wif3, self.Wdf3 = self.create_a_lstm_gate(
            hidden_size)
        # forget gate for global states
        self.Wxf4, self.Whf4, self.Wif4, self.Wdf4 = self.create_a_lstm_gate(
            hidden_size)
        # input gate for current state
        self.Wxi, self.Whi, self.Wii, self.Wdi = self.create_a_lstm_gate(
            hidden_size)
        # input gate for output gate
        self.Wxo, self.Who, self.Wio, self.Wdo = self.create_a_lstm_gate(
            hidden_size)

        self.bi, self.bo, self.bf1, self.bf2, self.bf3, self.bf4 = \
            (nn.Parameter(torch.Tensor(hidden_size)) for i in range(6))

        self.gated_Wxd = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.gated_Whd = nn.Parameter(
            torch.Tensor(hidden_size, hidden_size))

        self.gated_Wxo = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.gated_Who = nn.Parameter(
            torch.Tensor(hidden_size, hidden_size))

        self.gated_Wxf1, self.gated_Whf1 = \
            (nn.Parameter(torch.Tensor(hidden_size, hidden_size))
             for i in range(2))

        self.gated_bd, self.gated_bo, self.gated_bf1, self.gated_bf2 = \
            (nn.Parameter(torch.Tensor(hidden_size)) for i in range(4))

        self.i_norm, self.o_norm, \
            self.f1_norm, self.f2_norm, self.f3_norm, self.f4_norm, \
            self.gd_norm, self.go_norm, self.gf_norm = \
            (LayerNorm(hidden_size, eps=1e-3) for i in range(9))

        self.dropout = nn.Dropout(dropout)

        self.reset_parameters()

    def reset_parameters(self):
        mean, stdv = 0.0, 0.1
        for name, param in self.named_parameters():
            if 'norm' not in name:
                init.normal_(param, mean, stdv)

    def get_input_dim(self):
        return self.hidden_size

    def get_output_dim(self):
        return self.hidden_size

    def create_a_lstm_gate(self, hidden_size, mean=0.0, stddev=0.1):

        wxf = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        whf = nn.Parameter(torch.Tensor(hidden_size * 2, hidden_size))
        wif = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        wdf = nn.Parameter(torch.Tensor(hidden_size, hidden_size))

        return wxf, whf, wif, wdf

    def get_hidden_states_before(self, padding, hidden_states, step):
        # padding zeros
        if step < hidden_states.size(1):
            # remove last steps
            displaced_hidden_states = hidden_states[:, :-step, :]
            # concat padding
            return torch.cat([padding, displaced_hidden_states], dim=1)
        else:
            return torch.cat([padding] * hidden_states.size(1), dim=1)

    def get_hidden_states_after(self, padding, hidden_states, step):
        # padding zeros
        # remove last steps
        if step < hidden_states.size(1):
            displaced_hidden_states = hidden_states[:, step:, :]
            # concat padding
            return torch.cat([displaced_hidden_states, padding], dim=1)
        else:
            return torch.cat([padding] * hidden_states.size(1), dim=1)

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
        batch_size, seq_len, hidden_size = inputs.size()
        input_shape = inputs.size()

        # filters for attention
        mask_softmax_score = mask * 1e25 - 1e25
        mask_softmax_score = mask_softmax_score.unsqueeze(-1)

        # filter invalid steps
        mask = mask.unsqueeze(-1)  # batch, seq_len, 1

        ############################################################################
        # Init states
        ############################################################################
        # filter embedding states
        emb_hidden = inputs * mask  # batch, seq_len, hidden
        emb_cell = inputs * mask  # batch, seq_len, hidden
        # initial embedding states
        # batch * seq_len, hidden
        emb_hidden = emb_hidden.view(-1, hidden_size)
        # batch * seq_len, hidden
        emb_cell = emb_cell.view(-1, hidden_size)

        # randomly initialize the states
        hidden = torch.rand_like(inputs) - 0.5
        cell = torch.rand_like(inputs) - 0.5
        # filter it
        hidden = hidden * mask
        cell = cell * mask

        global_hidden = hidden.mean(1)
        global_cell = cell.mean(1)

        padding_list = [torch.zeros(batch_size, step + 1, hidden_size, device=inputs.device) for step in
                        range(self.SLSTM_step)]

        for _ in range(self.num_layers):
            #############################
            # update global node states #
            #############################
            # combine states
            combined_hidden = hidden.mean(dim=1)
            # input gate
            gated_d_t = torch.sigmoid(
                self.gd_norm(global_hidden @ self.gated_Wxd +
                             combined_hidden @ self.gated_Whd) + self.gated_bd)
            # output gate
            gated_o_t = torch.sigmoid(
                self.go_norm(global_hidden @ self.gated_Wxo +
                             combined_hidden @ self.gated_Who) + self.gated_bo)

            # copy global states for computing forget gate
            transformed_global_hidden = global_hidden.unsqueeze(
                1).repeat(1, seq_len, 1).view(-1, hidden_size)
            # forget gate for hidden states
            reshaped_hidden_output = hidden.view(-1, hidden_size)
            gated_f1_t = torch.sigmoid(
                self.gf_norm(transformed_global_hidden @ self.gated_Wxf1 +
                             reshaped_hidden_output @ self.gated_Whf1) + self.gated_bf1)
            
            # softmax on each hidden dimension
            reshaped_gated_f_t = gated_f1_t.view_as(
                inputs) + mask_softmax_score
            # Combine
            gated_softmax_scores = F.softmax(
                torch.cat([reshaped_gated_f_t, torch.unsqueeze(gated_d_t, dim=1)], dim=1), dim=1)

            gated_softmax_scores = self.dropout(
                gated_softmax_scores.permute(0, 2, 1)).permute(0, 2, 1) ## different from Zhang

            # split the softmax scores
            new_reshaped_gated_f_t = gated_softmax_scores[:, :seq_len, :]
            new_gated_d_t = gated_softmax_scores[:, seq_len:, :]
            transformed_global_cell = global_cell.unsqueeze(
                1).repeat(1, seq_len, 1).view(-1, hidden_size)

            # new global states
            global_cell = torch.sum(new_reshaped_gated_f_t * cell, dim=1) + \
                torch.squeeze(new_gated_d_t, dim=1) * global_cell
            global_hidden = gated_o_t * torch.tanh(global_cell)

            #############################
            # update other node states  #
            #############################

            # local lstm
            hidden_before = sum([
                self.get_hidden_states_before(
                    padding_list[step], hidden, step + 1).view(-1, hidden_size)
                for step in range(self.SLSTM_step)])
            hidden_after = sum([
                self.get_hidden_states_after(
                    padding_list[step], hidden, step + 1).view(-1, hidden_size)
                for step in range(self.SLSTM_step)])

            # get states after
            cell_before = sum([
                self.get_hidden_states_before(
                    padding_list[step], cell, step + 1).view(-1, hidden_size)
                for step in range(self.SLSTM_step)])

            cell_after = sum([
                self.get_hidden_states_after(
                    padding_list[step], cell, step + 1).view(-1, hidden_size)
                for step in range(self.SLSTM_step)])

            # reshape for matmul
            hidden = hidden.view(-1, hidden_size)
            cell = cell.view(-1, hidden_size)

            # concat before and after hidden states  ### different from Zhang et al 2018
            concat_before_after = torch.cat(
                [hidden_before, hidden_after], dim=1)

            f1_t = torch.sigmoid(   ### Different from Zhang et al 2018 layernorm
                self.f1_norm(
                    hidden @ self.Wxf1 +
                    concat_before_after @ self.Whf1 +
                    emb_hidden @ self.Wif1 +
                    transformed_global_hidden @ self.Wdf1) + self.bf1
            )

            f2_t = torch.sigmoid(
                self.f2_norm(
                    hidden @ self.Wxf2 +
                    concat_before_after @ self.Whf2 +
                    emb_hidden @ self.Wif2 +
                    transformed_global_hidden @ self.Wdf2) + self.bf2
            )

            f3_t = torch.sigmoid(
                self.f3_norm(hidden @ self.Wxf3 +
                             concat_before_after @ self.Whf3 +
                             emb_hidden @ self.Wif3 +
                             transformed_global_hidden @ self.Wdf3) + self.bf3
            )

            f4_t = torch.sigmoid(
                self.f4_norm(
                    hidden @ self.Wxf4 +
                    concat_before_after @ self.Whf4 +
                    emb_hidden @ self.Wif4 +
                    transformed_global_hidden @ self.Wdf4) + self.bf4
            )

            i_t = torch.sigmoid(
                self.i_norm(
                    hidden @ self.Wxi +
                    concat_before_after @ self.Whi +
                    emb_hidden @ self.Wii +
                    transformed_global_hidden @ self.Wdi) + self.bi
            )

            o_t = torch.sigmoid(
                self.o_norm(
                    hidden @ self.Wxo +
                    concat_before_after @ self.Who +
                    emb_hidden @ self.Wio +
                    transformed_global_hidden @ self.Wdo) + self.bo
            )

            f1_t, f2_t, f3_t, f4_t, i_t = torch.unsqueeze(f1_t, dim=1), torch.unsqueeze(f2_t, dim=1), torch.unsqueeze(
                f3_t, dim=1), torch.unsqueeze(f4_t, dim=1), torch.unsqueeze(i_t, dim=1)

            five_gates = torch.cat([f1_t, f2_t, f3_t, f4_t, i_t], dim=1)
            five_gates = F.softmax(five_gates, dim=1)

            f1_t, f2_t, f3_t, f4_t, i_t = torch.chunk(five_gates, 5, dim=1)

            f1_t, f2_t, f3_t, f4_t, i_t = torch.squeeze(f1_t, dim=1), torch.squeeze(f2_t, dim=1), torch.squeeze(f3_t, dim=1), torch.squeeze(
                f4_t, dim=1), torch.squeeze(i_t, dim=1)

            c_t = (cell_before * f1_t) + \
                  (cell_after * f2_t) + \
                  (cell * i_t) + \
                  (emb_cell * f3_t) + \
                  (transformed_global_cell * f4_t)   ### different from Zhang et al 2018

            reshaped_hidden_output = hidden.view(-1, hidden_size)
            h_t = o_t * (c_t + reshaped_hidden_output)  ### different from Zhang et al 2018

            # update states
            hidden = h_t.view(input_shape)
            cell = c_t.view(input_shape)
            hidden = hidden * mask
            cell = cell * mask

        output_dict = {
            'hiddens': hidden,
            'global_hidden': global_hidden
        }

        return output_dict
