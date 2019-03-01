#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author: zeeeyang
# @Email: liangshuailong@gmail.com
# @Date:   2019-02-27 23:00:11
# @Last Modified by:  Shuailong
# @Last Modified time: 2019-02-28 21:13:08

# -*- coding: utf-8 -*-
from overrides import overrides

from torch import nn
import torch
import torch.nn.functional as F
import torch.nn.init as init

from allennlp.common.registrable import FromParams
from allennlp.nn.util import replace_masked_values
from allennlp.modules import LayerNorm


class SentencePairSLSTMEncoder(nn.Module, FromParams):

    def __init__(self,
                 hidden_size: int,
                 SLSTM_step: int = 1,
                 num_layers: int = 7,
                 dropout: int = 0.5,
                 ):
        super(SentencePairSLSTMEncoder, self).__init__()
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

        self.bi = nn.Parameter(torch.Tensor(hidden_size))
        self.bo = nn.Parameter(torch.Tensor(hidden_size))
        self.bf1 = nn.Parameter(torch.Tensor(hidden_size))
        self.bf2 = nn.Parameter(torch.Tensor(hidden_size))
        self.bf3 = nn.Parameter(torch.Tensor(hidden_size))
        self.bf4 = nn.Parameter(torch.Tensor(hidden_size))

        self.gated_Wxd = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.gated_Whd = nn.Parameter(
            torch.Tensor(hidden_size * 4, hidden_size))

        self.gated_Wxo = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.gated_Who = nn.Parameter(
            torch.Tensor(hidden_size * 4, hidden_size))

        self.gated_Wxf1 = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.gated_Whf1 = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.gated_Wxf2 = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.gated_Whf2 = nn.Parameter(torch.Tensor(hidden_size, hidden_size))

        self.gated_bd = nn.Parameter(torch.Tensor(hidden_size))
        self.gated_bo = nn.Parameter(torch.Tensor(hidden_size))
        self.gated_bf1 = nn.Parameter(torch.Tensor(hidden_size))
        self.gated_bf2 = nn.Parameter(torch.Tensor(hidden_size))

        self.dropout = nn.Dropout(dropout)

        self.i_norm = LayerNorm(hidden_size)
        self.o_norm = LayerNorm(hidden_size)
        self.f1_norm = LayerNorm(hidden_size)
        self.f2_norm = LayerNorm(hidden_size)
        self.f3_norm = LayerNorm(hidden_size)
        self.f4_norm = LayerNorm(hidden_size)
        self.gd_norm = LayerNorm(hidden_size)
        self.go_norm = LayerNorm(hidden_size)
        self.gf_norm = LayerNorm(hidden_size)

        self.reset_parameters()

    def reset_parameters(self):
        mean, stdv = 0.0, 0.1
        for weight in self.parameters():
            init.normal_(weight, mean, stdv)

    def get_input_dim(self):
        return self.hidden_size

    def get_output_dim(self):
        return self.hidden_size * 5

    def create_a_lstm_gate(self, hidden_size, mean=0.0, stddev=0.1):

        wxf = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        whf = nn.Parameter(torch.Tensor(hidden_size * 2, hidden_size))
        wif = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        wdf = nn.Parameter(torch.Tensor(hidden_size, hidden_size))

        return wxf, whf, wif, wdf

    def combine_hidden_states(self, s1_hiddens, s1_mask, s2_hiddens, s2_mask, pooling="mean"):
        '''
        s1_hiddens: batch x seq_len x hidden
        s1_mask: batch x seq_len
        s2_hiddens: batch x seq_len x hidden
        s2_mask: batch x seq_len
        '''
        # The pooling layer -- max and avg pooling.
        # (batch_size, model_dim)
        if pooling == "mean":
            s1_pooled = torch.sum(s1_hiddens * s1_mask.unsqueeze(-1), dim=1) / torch.sum(
                s1_mask, 1, keepdim=True
            )
            s2_pooled = torch.sum(s2_hiddens * s2_mask.unsqueeze(-1), dim=1) / torch.sum(
                s2_mask, 1, keepdim=True
            )
        else:
            s1_pooled, _ = replace_masked_values(
                s1_hiddens, s1_mask.unsqueeze(-1), -1e7
            ).max(dim=1)
            s2_pooled, _ = replace_masked_values(
                s2_hiddens, s2_mask.unsqueeze(-1), -1e7
            ).max(dim=1)

        # Only use the last layer
        features = torch.cat([s1_pooled,
                              s2_pooled,
                              torch.abs(s1_pooled - s2_pooled),
                              s1_pooled * s2_pooled],
                             dim=1)
        return features

    def get_hidden_states_before(self, padding, hidden_states, step):
        # padding zeros
        # padding = create_padding_variable(self.training, (shape[0], step, hidden_size))
        if step < hidden_states.size(1):
            # remove last steps
            displaced_hidden_states = hidden_states[:, :-step, :]
            # concat padding
            return torch.cat([padding, displaced_hidden_states], dim=1)
        else:
            return torch.cat([padding] * hidden_states.size(1), dim=1)

    def get_hidden_states_after(self, padding, hidden_states, step):
        # padding zeros
        # padding = create_padding_variable(self.training, (shape[0], step, hidden_size))
        # remove last steps
        if step < hidden_states.size(1):
            displaced_hidden_states = hidden_states[:, step:, :]
            # concat padding
            return torch.cat([displaced_hidden_states, padding], dim=1)
        else:
            return torch.cat([padding] * hidden_states.size(1), dim=1)

    @overrides
    def forward(self,  # type: ignore
                s1_inputs: torch.FloatTensor,
                s1_mask: torch.FloatTensor,
                s2_inputs: torch.FloatTensor,
                s2_mask: torch.FloatTensor):
        """
        Parameters
        ----------
        s1_inputs : ``torch.FloatTensor``
            A tensor of shape (batch_size, seq_len, hidden_size)
        s1_mask : ``torch.FloatTensor``
            A tensor of shape (batch_size, seq_len)
        s2_input : ``torch.FloatTensor``
            A tensor of shape (batch_size, seq_len, hidden_size)
        s2_mask : ``torch.FloatTensor``
            A tensor of shape (batch_size, seq_len)
        Returns
        -------
        An output dictionary consisting of:
        premise_hiddens: ``torch.FloatTensor``
            A tensor of shape (batch_size, hidden_size)
        hypothesis_hiddens: ``torch.FloatTensor``
            A tensor of shape (batch_size, hidden_size)
        global_hiddens: ``torch.FloatTensor``
            A tensor of shape (batch_size, hidden_size)
        """
        batch_size, s1_seq_len, hidden_size = s1_inputs.size()
        s2_seq_len = s2_inputs.size(1)
        s1_shape, s2_shape = s1_inputs.size(), s2_inputs.size()

        # filters for attention
        s1_mask_softmax_score = s1_mask * 1e25 - 1e25
        s1_mask_softmax_score = s1_mask_softmax_score.unsqueeze(-1)
        s2_mask_softmax_score = s2_mask * 1e25 - 1e25
        s2_mask_softmax_score = s2_mask_softmax_score.unsqueeze(-1)

        # filter invalid steps
        s1_mask = s1_mask.unsqueeze(-1)  # batch, seq_len, 1
        s2_mask = s2_mask.unsqueeze(-1)

        ############################################################################
        # Init states for s1
        ############################################################################
        # filter embedding states
        s1_emb_hidden = s1_inputs * s1_mask  # batch, seq_len, hidden
        s1_emb_cell = s1_inputs * s1_mask  # batch, seq_len, hidden
        # initial embedding states
        # batch * seq_len, hidden
        s1_emb_hidden = s1_emb_hidden.view(-1, hidden_size)
        # batch * seq_len, hidden
        s1_emb_cell = s1_emb_cell.view(-1, hidden_size)

        # randomly initialize the states
        s1_hidden = torch.rand_like(s1_inputs) - 0.5
        s1_cell = torch.rand_like(s1_inputs) - 0.5
        # filter it
        s1_hidden = s1_hidden * s1_mask
        s1_cell = s1_cell * s1_mask

        # do the same thing for sentence 2, should check it carefully
        ############################################################################
        # Init states for s2
        ############################################################################
        # filter embedding states
        s2_emb_hidden = s2_inputs * s2_mask  # batch, seq_len, hidden
        s2_emb_cell = s2_inputs * s2_mask  # batch, seq_len, hidden
        # initial embedding states
        # batch * seq_len, hidden
        s2_emb_hidden = s2_emb_hidden.view(-1, hidden_size)
        # batch * seq_len, hidden
        s2_emb_cell = s2_emb_cell.view(-1, hidden_size)

        s2_hidden = torch.rand_like(s2_inputs) - 0.5
        s2_cell = torch.rand_like(s2_inputs) - 0.5
        # filter it
        s2_hidden = s2_hidden * s2_mask
        s2_cell = s2_cell * s2_mask

        global_hidden = torch.cat(
            [s1_hidden, s2_hidden], dim=1).mean(1)
        global_cell = torch.cat(
            [s1_cell, s2_cell], dim=1).mean(1)

        padding_list = [torch.zeros(batch_size, step + 1, hidden_size, device=s1_inputs.device) for step in
                        range(self.SLSTM_step)]

        for _ in range(self.num_layers):
            # update global node states
            # combine states
            combined_hidden = self.combine_hidden_states(
                s1_hidden, s1_mask.squeeze(-1), s2_hidden, s2_mask.squeeze(-1))  # batch  * (4h)
            # input gate
            gated_d_t = torch.sigmoid(
                self.gd_norm(global_hidden @ self.gated_Wxd +
                             combined_hidden @ self.gated_Whd) + self.gated_bd)
            # output gate
            gated_o_t = torch.sigmoid(
                self.go_norm(global_hidden @ self.gated_Wxo +
                             combined_hidden @ self.gated_Who) + self.gated_bo)
            # S1
            # copy global states for computing forget gate
            s1_transformed_global_hidden = torch.unsqueeze(
                global_hidden, dim=1).repeat(1, s1_seq_len, 1).view(-1, hidden_size)
            # forget gate for hidden states
            s1_reshaped_hidden_output = s1_hidden.view(-1, hidden_size)
            gated_f1_t = torch.sigmoid(
                self.gf_norm(s1_transformed_global_hidden @ self.gated_Wxf1 +
                             s1_reshaped_hidden_output @ self.gated_Whf1) + self.gated_bf1)
            # softmax on each hidden dimension
            s1_reshaped_gated_f_t = gated_f1_t.view_as(
                s1_inputs) + s1_mask_softmax_score

            # S2
            s2_transformed_global_hidden = torch.unsqueeze(
                global_hidden, dim=1).repeat(1, s2_seq_len, 1).view(-1, hidden_size)
            s2_reshaped_hidden_output = s2_hidden.view(-1, hidden_size)
            gated_f2_t = torch.sigmoid(
                self.gf_norm(s2_transformed_global_hidden @ self.gated_Wxf2 +
                             s2_reshaped_hidden_output @ self.gated_Whf2) + self.gated_bf2)
            s2_reshaped_gated_f_t = gated_f2_t.view_as(
                s2_inputs) + s2_mask_softmax_score

            # Combine
            gated_softmax_scores = F.softmax(
                torch.cat([s1_reshaped_gated_f_t, s2_reshaped_gated_f_t,
                           torch.unsqueeze(gated_d_t, dim=1)], dim=1), dim=1)

            gated_softmax_scores = self.dropout(gated_softmax_scores)

            # split the softmax scores
            s1_new_reshaped_gated_f_t = gated_softmax_scores[:, :s1_seq_len, :]
            s2_new_reshaped_gated_f_t = gated_softmax_scores[:,
                                                             s1_seq_len:s1_seq_len + s2_seq_len, :]
            new_gated_d_t = gated_softmax_scores[:,
                                                 s1_seq_len + s2_seq_len:, :]
            s1_transformed_global_cell = global_cell.unsqueeze(
                1).repeat(1, s1_seq_len, 1).view(-1, hidden_size)
            s2_transformed_global_cell = global_cell.unsqueeze(
                1).repeat(1, s2_seq_len, 1).view(-1, hidden_size)

            # new global states
            global_cell = torch.sum(s1_new_reshaped_gated_f_t * s1_cell, dim=1) + \
                torch.sum(s2_new_reshaped_gated_f_t * s2_cell, dim=1) + \
                torch.squeeze(new_gated_d_t, dim=1) * global_cell
            global_hidden = gated_o_t * torch.tanh(global_cell)

            def local_lstm(input_shape,
                           mask,
                           hidden,
                           cell,
                           emb_hidden,
                           transformed_global_hidden,
                           emb_cell,
                           transformed_global_cell):

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

                # concat before and after hidden states
                concat_before_after = torch.cat(
                    [hidden_before, hidden_after], dim=1)

                f1_t = torch.sigmoid(
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
                      (emb_cell * f3_t) + \
                      (transformed_global_cell * f4_t) + \
                      (cell * i_t)

                reshaped_hidden_output = hidden.view(-1, hidden_size)
                h_t = o_t * (c_t + reshaped_hidden_output)

                # update states
                hidden = h_t.view(input_shape)
                cell = c_t.view(input_shape)
                hidden = hidden * mask
                cell = cell * mask

                return hidden, cell

            s1_hidden, s1_cell = local_lstm(s1_shape,
                                            s1_mask,
                                            s1_hidden,
                                            s1_cell,
                                            s1_emb_hidden,
                                            s1_transformed_global_hidden,
                                            s1_emb_cell,
                                            s1_transformed_global_cell)
            s2_hidden, s2_cell = local_lstm(s2_shape,
                                            s2_mask,
                                            s2_hidden,
                                            s2_cell,
                                            s2_emb_hidden,
                                            s2_transformed_global_hidden,
                                            s2_emb_cell,
                                            s2_transformed_global_cell)

        s1_hidden = self.dropout(s1_hidden)
        s2_hidden = self.dropout(s2_hidden)
        global_hidden = self.dropout(global_hidden)

        output_dict = {
            'premise_hiddens': s1_hidden,
            'hypothesis_hiddens': s2_hidden,
            'global_hiddens': global_hidden
        }

        return output_dict
