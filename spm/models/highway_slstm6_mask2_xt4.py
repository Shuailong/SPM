# -*- coding: utf-8 -*-

from torch import nn
from torch import autograd
import numpy as np
import torch
import sys
import torch.nn.functional as F
from utils.locked_dropout import LockedDropout
#from allennlp.modules import ScalarMix

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class LayerNorm2(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-3):
        super(LayerNorm2, self).__init__()

    def forward(self, x):
        return x


class SLSTM(nn.Module):

    def __init__(self, config):
        #current
        super(SLSTM, self).__init__()
        self.config = config
        hidden_size = config.HP_hidden_dim
        #self.mixture = ScalarMix(config.HP_SLSTM_layer)
        #self.mixture.gamma.requires_grad = False

        print(f"hidden_size: {hidden_size}")
        #sys.exit(0)

        # forget gate for left
        self.Wxf1, self.Whf1, self.Wif1, self.Wdf1 = self.create_a_lstm_gate(hidden_size, self.config.HP_gpu)
        #forget gate for right
        self.Wxf2, self.Whf2, self.Wif2, self.Wdf2 = self.create_a_lstm_gate(hidden_size, self.config.HP_gpu)
        #forget gate for inital states
        self.Wxf3, self.Whf3, self.Wif3, self.Wdf3 = self.create_a_lstm_gate(hidden_size, self.config.HP_gpu)
        #forget gate for dummy states
        self.Wxf4, self.Whf4, self.Wif4, self.Wdf4 = self.create_a_lstm_gate(hidden_size, self.config.HP_gpu)
        #input gate for current state
        self.Wxi, self.Whi, self.Wii, self.Wdi = self.create_a_lstm_gate(hidden_size, self.config.HP_gpu)
        #input gate for output gate
        self.Wxo, self.Who, self.Wio, self.Wdo = self.create_a_lstm_gate(hidden_size, self.config.HP_gpu)

        self.bi = self.create_bias_variable(hidden_size, self.config.HP_gpu)
        self.bo = self.create_bias_variable(hidden_size, self.config.HP_gpu)
        self.bf1 = self.create_bias_variable(hidden_size, self.config.HP_gpu)
        self.bf2 = self.create_bias_variable(hidden_size, self.config.HP_gpu)
        self.bf3 = self.create_bias_variable(hidden_size, self.config.HP_gpu)
        self.bf4 = self.create_bias_variable(hidden_size, self.config.HP_gpu)

        self.gated_Wxd = self.create_to_hidden_variable(hidden_size, hidden_size, self.config.HP_gpu)
        self.gated_Whd = self.create_to_hidden_variable(hidden_size, hidden_size, self.config.HP_gpu)

        self.gated_Wxo = self.create_to_hidden_variable(hidden_size, hidden_size, self.config.HP_gpu)
        self.gated_Who = self.create_to_hidden_variable(hidden_size, hidden_size, self.config.HP_gpu)

        self.gated_Wxf = self.create_to_hidden_variable(hidden_size, hidden_size, self.config.HP_gpu)
        self.gated_Whf = self.create_to_hidden_variable(hidden_size, hidden_size, self.config.HP_gpu)

        self.gated_bd = self.create_bias_variable(hidden_size, self.config.HP_gpu)
        self.gated_bo = self.create_bias_variable(hidden_size, self.config.HP_gpu)
        self.gated_bf = self.create_bias_variable(hidden_size, self.config.HP_gpu)

        self.h_drop = nn.Dropout(config.HP_dropout)
        self.c_drop = nn.Dropout(config.HP_dropout)

        self.g_drop = nn.Dropout(config.HP_dropout)

        #self.input_norm = LayerNorm(hidden_size)
        self.i_norm = LayerNorm(hidden_size)
        self.o_norm = LayerNorm(hidden_size)
        self.f1_norm = LayerNorm(hidden_size)
        self.f2_norm = LayerNorm(hidden_size)
        self.f3_norm = LayerNorm(hidden_size)
        self.f4_norm = LayerNorm(hidden_size)

        self.gd_norm = LayerNorm(hidden_size)
        self.go_norm = LayerNorm(hidden_size)
        self.gf_norm = LayerNorm(hidden_size)

        #self.c_norm = LayerNorm(hidden_size)
        #self.gc_norm = LayerNorm(hidden_size)

        if self.config.HP_gpu:
            self.to(self.config.device)

    def create_bias_variable(self, size, gpu=False, mean=0.0, stddev=0.1):
        data = torch.zeros(size)
        if gpu:
            data = data.to(self.config.device)
        var = nn.Parameter(data, requires_grad=True)
        var.data.normal_(mean, std=stddev) #standard
        #print(f"[tlog] bias: {var}")
        #var.data.zero_()
        #if gpu:
        #    var = var.to(config.device)
        return var

    def create_to_hidden_variable(self, size1, size2, gpu=False, mean=0.0, stddev=0.1):
        data = torch.zeros((size1, size2))
        if gpu:
            data = data.to(self.config.device)
        var = nn.Parameter(data, requires_grad=True)
        var.data.normal_(mean, std=stddev) #standard
        #print(f"[tlog] hidden: {var}")
        #torch.nn.init.xavier_normal(var)
        #torch.nn.init.xavier_uniform(var)
        #if gpu:
        #    var = var.to(config.device)
        return var

    def create_a_lstm_gate(self, hidden_size, gpu=False, mean=0.0, stddev=0.1):

        wxf = self.create_to_hidden_variable(hidden_size, hidden_size, gpu, mean, stddev)
        whf = self.create_to_hidden_variable(2 * hidden_size, hidden_size, gpu, mean, stddev)
        wif = self.create_to_hidden_variable(hidden_size, hidden_size, gpu, mean, stddev)
        wdf = self.create_to_hidden_variable(hidden_size, hidden_size, gpu, mean, stddev)

        return wxf, whf, wif, wdf

    def create_nograd_variable(self, minval, maxval, gpu, *shape):
        data = torch.zeros(*shape)
        if gpu:
            data = data.to(self.config.device)
        var = autograd.Variable(data, requires_grad=False)
        var.data.uniform_(minval, maxval)#standard
        #torch.nn.init.xavier_normal(var)
        #torch.nn.init.xavier_uniform(var)
        #if gpu:
        #    var = var.to(config.device)
        return var

    def create_padding_variable(self, gpu, *shape):
        data = torch.zeros(*shape)
        if gpu:
            data = data.to(self.config.device)
        var = autograd.Variable(data, requires_grad=False)
        #if gpu:
        #    var = var.to(config.device)
        return var


    def get_hidden_states_before(self, padding, hidden_states, step):
        #padding zeros
        #padding = create_padding_variable(self.training, self.config.HP_gpu, (shape[0], step, hidden_size))
        if step < hidden_states.size()[1]:
            #remove last steps
            displaced_hidden_states = hidden_states[:, :-step, :]
            #concat padding
            return torch.cat([padding, displaced_hidden_states], dim=1)
        else:
            return torch.cat([padding]*hidden_states.size()[1], dim=1)

    def get_hidden_states_after(self, padding, hidden_states, step):
        #padding zeros
        #padding = create_padding_variable(self.training, self.config.HP_gpu, (shape[0], step, hidden_size))
        #remove last steps
        if step < hidden_states.size()[1]:
            displaced_hidden_states = hidden_states[:, step:, :]
            #concat padding
            return torch.cat([displaced_hidden_states, padding], dim=1)
        else:
            return torch.cat([padding]*hidden_states.size()[1], dim=1)

    def sum_together(self, l):
        return sum(l)

    def forward(self, word_inputs, mask,  num_layers, seq_length=None):

        # filters for attention
        #print("[tlog] mask: " + str(mask))
        mask_softmax_score = mask.float() * 1e25 - 1e25  # 10, 40
        #print("[tlog] mask_softmax_score: " + str(mask_softmax_score))

        mask_softmax_score_expanded = torch.unsqueeze(mask_softmax_score, dim=2)  # 10, 40, 1
        #print("[tlog] mask_softmax_expanded: " + str(mask_softmax_score_expanded))
        # filter invalid steps
        sequence_mask = torch.unsqueeze(mask.float(), dim=2)  # 10, 40, 1
        sequence_lengths = torch.sum(sequence_mask, dim=1)  # 10, 40, 1
        #print("[tlog] sequence_mask: " + str(mask_softmax_score_expanded))

        #word_inputs = self.input_norm(word_inputs) #//maybe we can remove this one

        #word_inputs = self.i_drop(word_inputs)
        # filter embedding states

        filtered_word_inputs = word_inputs * sequence_mask  # 10, 40, 600

        # record shape of the batch
        shape = word_inputs.size()
        ##print("[tlog] shape: " + str(shape)) # 10, 37, 100

        # initial embedding states
        embedding_hidden_state = filtered_word_inputs.view(-1, shape[-1])
        embedding_cell_state = filtered_word_inputs.view(-1, shape[-1])

        # randomly initialize the states
        initial_hidden_states = filtered_word_inputs
        initial_cell_states = filtered_word_inputs
        #print("[tlog] initial_hidden_states: " + str(initial_hidden_states))
        #print("[tlog] initial_cell_states: " + str(initial_cell_states))
        #sys.exit(0)
        # inital dummy node states
        dummynode_hidden_states =torch.sum(initial_hidden_states, dim=1)/sequence_lengths 
        #dummynode_hidden_states = self.i_drop(dummynode_hidden_states)
        # self.debug = dummynode_hidden_states
        ##print("[tlog] dummynode_hidden_states: " + str(dummynode_hidden_states))
        dummynode_cell_states = torch.sum(initial_cell_states, dim=1)/sequence_lengths
        ##print("[tlog] dummynode_cell_states: " + str(dummynode_cell_states)) # batch_size * hidden_dim

        hidden_size = self.config.HP_hidden_dim

        padding_list = [self.create_padding_variable(self.config.HP_gpu, (shape[0], step+1, hidden_size)) for step in range(self.config.HP_SLSTM_step)]


        for i in range(num_layers):
            #print("[tlog] layers: " + str(i))

            # update word node states
            # get states before

            initial_hidden_states_before = [(self.get_hidden_states_before(padding_list[step], initial_hidden_states, step + 1) * sequence_mask).view(-1, hidden_size) \
                                            for step in range(self.config.HP_SLSTM_step)]

            initial_hidden_states_before = self.sum_together(initial_hidden_states_before)

            ##print("[tlog] initial_hidden_states_before: " + str(initial_hidden_states_before))

            initial_hidden_states_after = [(self.get_hidden_states_after(padding_list[step], initial_hidden_states, step + 1) * sequence_mask).view(-1, hidden_size) \
                                           for step in range(self.config.HP_SLSTM_step)]

            initial_hidden_states_after = self.sum_together(initial_hidden_states_after)
            ##print("[tlog] initial_hidden_states_after: " + str(initial_hidden_states_after))
            #sys.exit(0)
            # get states after
            initial_cell_states_before = [(self.get_hidden_states_before(padding_list[step], initial_cell_states, step + 1) * sequence_mask).view(-1, hidden_size) \
                                          for step in range(self.config.HP_SLSTM_step)]

            initial_cell_states_before = self.sum_together(initial_cell_states_before)
            ##print("[tlog] initial_cell_states_before: " + str(initial_cell_states_before))
            #sys.exit(0)
            initial_cell_states_after = [(self.get_hidden_states_after(padding_list[step], initial_cell_states, step + 1) * sequence_mask).view(-1, hidden_size) \
                                         for step in range(self.config.HP_SLSTM_step)]

            initial_cell_states_after = self.sum_together(initial_cell_states_after)
            ##print("[tlog] initial_cell_states_after: " + str(initial_cell_states_after))

            #sys.exit(0)
            # reshape for matmul
            initial_hidden_states = initial_hidden_states.view(-1, hidden_size)
            initial_cell_states = initial_cell_states.view(-1, hidden_size)

            # concat before and after hidden states
            concat_before_after = torch.cat([initial_hidden_states_before, initial_hidden_states_after], dim=1)
            ##print("[tlog] concat_before_after: " + str(concat_before_after))

            # copy dummy node states

            transformed_dummynode_cell_states = torch.unsqueeze(dummynode_cell_states, dim=1).repeat(1, shape[1], 1).view(-1, hidden_size)

            transformed_dummynode_hidden_states = torch.unsqueeze(dummynode_hidden_states, dim=1).repeat(1, shape[1], 1)
            transformed_dummynode_hidden_states = (transformed_dummynode_hidden_states * sequence_mask).view(-1, hidden_size)#add 2019-03-12
            #print("[tlog] concat_before_after: " + str(transformed_dummynode_hidden_states))

            f1_t = torch.sigmoid(
                self.f1_norm(
                torch.matmul(initial_hidden_states, self.Wxf1) + torch.matmul(concat_before_after, self.Whf1) +
                torch.matmul(embedding_hidden_state, self.Wif1) + torch.matmul(transformed_dummynode_hidden_states, self.Wdf1) + self.bf1)
            )

            f2_t = torch.sigmoid(
                self.f2_norm(
                torch.matmul(initial_hidden_states, self.Wxf2) + torch.matmul(concat_before_after, self.Whf2) +
                torch.matmul(embedding_hidden_state, self.Wif2) + torch.matmul(transformed_dummynode_hidden_states, self.Wdf2) + self.bf2)
            )

            f3_t = torch.sigmoid(
                self.f3_norm(torch.matmul(initial_hidden_states, self.Wxf3) + torch.matmul(concat_before_after, self.Whf3) +
                torch.matmul(embedding_hidden_state, self.Wif3) + torch.matmul(transformed_dummynode_hidden_states, self.Wdf3) + self.bf3)
            )

            f4_t = torch.sigmoid(
                self.f4_norm(
                torch.matmul(initial_hidden_states, self.Wxf4) + torch.matmul(concat_before_after, self.Whf4) +
                torch.matmul(embedding_hidden_state, self.Wif4) + torch.matmul(transformed_dummynode_hidden_states, self.Wdf4) + self.bf4)
            )

            i_t = torch.sigmoid(
                self.i_norm(
                torch.matmul(initial_hidden_states, self.Wxi) + torch.matmul(concat_before_after, self.Whi) +
                torch.matmul(embedding_hidden_state, self.Wii) + torch.matmul(transformed_dummynode_hidden_states, self.Wdi) + self.bi)
            )

            o_t = torch.sigmoid(
                self.o_norm(
                torch.matmul(initial_hidden_states, self.Wxo) + torch.matmul(concat_before_after, self.Who) +
                torch.matmul(embedding_hidden_state, self.Wio) + torch.matmul(transformed_dummynode_hidden_states, self.Wdo) + self.bo)
            )

            f1_t, f2_t, f3_t = torch.unsqueeze(f1_t, dim=1), torch.unsqueeze(f2_t, dim=1), torch.unsqueeze(f3_t, dim=1)

            first_three_gates = torch.cat([f1_t, f2_t, f3_t], dim=1)
            first_three_gates = F.softmax(first_three_gates, dim=1)
            f1_t, f2_t, f3_t = torch.chunk(first_three_gates, 3, dim=1)
            f1_t, f2_t, f3_t = torch.squeeze(f1_t, dim=1), torch.squeeze(f2_t, dim=1), torch.squeeze(f3_t, dim=1)

            local_c_t = ( initial_cell_states_before * f1_t) + ( initial_cell_states_after * f2_t) + (initial_cell_states * i_t)
            local_and_global_c_t =  local_c_t * (1.0 - f4_t) + ( transformed_dummynode_cell_states * f4_t ) 
            c_t =  embedding_cell_state * i_t + local_and_global_c_t * (1.0 - i_t)

            #h_t = o_t * torch.tanh(self.c_norm(c_t)) #+ (1.0 - o_t) * embedding_hidden_state
            #h_t = o_t * torch.tanh(c_t) #+ (1.0 - o_t) * embedding_hidden_state
            #h_t = o_t * c_t +  reshaped_hidden_output # 92.75
            #c_t =  c_t +  reshaped_hidden_output # 92.75
            #h_t = o_t * c_t 
            #h_t = o_t * (c_t +  reshaped_hidden_output)
            reshaped_hidden_output = initial_hidden_states.view(-1, hidden_size)
            h_t = o_t * (c_t +  reshaped_hidden_output)
            #h_t = o_t * (c_t +  reshaped_hidden_output) + (1.0 - o_t) * embedding_hidden_state

            ##print("[tlog] c_t: " + str(c_t))
            ##print("[tlog] h_t: " + str(h_t))
            #sys.exit(0)

            # update states
            initial_hidden_states = h_t.view(shape[0], shape[1], hidden_size)
            initial_cell_states = c_t.view(shape[0], shape[1], hidden_size)

            initial_hidden_states = initial_hidden_states * sequence_mask
            initial_cell_states = initial_cell_states * sequence_mask
            ##################################################################################################################################
            # update dummy node states
            # average states
            combined_word_hidden_state = torch.sum(initial_hidden_states, dim=1)/sequence_lengths

            ##print("[tlog] combined_word_hidden_state: " + str(combined_word_hidden_state))
            reshaped_hidden_output = initial_hidden_states.view(-1, hidden_size)

            # input gate
            gated_d_t = torch.sigmoid(
                self.gd_norm(torch.matmul(dummynode_hidden_states, self.gated_Wxd) + torch.matmul(combined_word_hidden_state,
                                                                          self.gated_Whd) + self.gated_bd)
            )
            ##print("[tlog] gated_d_t: " + str(gated_d_t))
            #sys.exit(0)
            # output gate
            gated_o_t = torch.sigmoid(
                self.go_norm(torch.matmul(dummynode_hidden_states, self.gated_Wxo) + torch.matmul(combined_word_hidden_state,
                                                                          self.gated_Who) + self.gated_bo)
            )
            ##print("[tlog] gated_o_t: " + str(gated_o_t))
            # forget gate for hidden states
            gated_f_t = torch.sigmoid(
                self.gf_norm(torch.matmul(transformed_dummynode_hidden_states, self.gated_Wxf) + torch.matmul(reshaped_hidden_output,
                                                                                      self.gated_Whf) + self.gated_bf)
            )
            ##print("[tlog] gated_f_t: " + str(gated_f_t))
            #sys.exit(0)

            # softmax on each hidden dimension
            reshaped_gated_f_t = gated_f_t.view(shape[0], shape[1], hidden_size) + mask_softmax_score_expanded
            ##print("[tlog] reshaped_gated_f_t: " + str(reshaped_gated_f_t))

            gated_softmax_scores = F.softmax(
                torch.cat([reshaped_gated_f_t, torch.unsqueeze(gated_d_t, dim=1)], dim=1), dim=1)

            #print("[tlog] gated_softmax_scores: " + str(gated_softmax_scores))
            gated_softmax_scores = self.g_drop(gated_softmax_scores.permute(0, 2, 1)).permute(0,2,1)
            #print("[tlog] gated_softmax_scores: " + str(gated_softmax_scores))
            #sys.exit(0)

            # self.debug = gated_softmax_scores
            # split the softmax scores
            new_reshaped_gated_f_t = gated_softmax_scores[:, :shape[1], :]
            new_gated_d_t = gated_softmax_scores[:, shape[1]:, :]
            ##print("[tlog] new_reshaped_gated_f_t: " + str(new_reshaped_gated_f_t))
            ##print("[tlog] new_gated_d_t: " + str(new_gated_d_t))

            # new dummy states
            dummy_c_t = torch.sum(new_reshaped_gated_f_t * initial_cell_states, dim=1) + torch.squeeze(new_gated_d_t, dim=1) * dummynode_cell_states

            #dummy_h_t = gated_o_t * torch.tanh(self.gc_norm(dummy_c_t))
            dummy_h_t = gated_o_t * torch.tanh(dummy_c_t)
            #sys.exit(0)
            ##########################################################################################################################################################
            ##################################################################################################################################

            dummynode_hidden_states = dummy_h_t
            dummynode_cell_states = dummy_c_t

            #initial_hidden_states = self.locked_dropout(initial_hidden_states, self.config.HP_dropout)
            #dummynode_hidden_states = self.i_drop(dummynode_hidden_states)
            #hidden_buffer.append(initial_hidden_states)


        initial_hidden_states = self.h_drop(initial_hidden_states)
        #initial_hidden_states = self.h_drop(self.mixture(hidden_buffer))
        #initial_cell_states = self.c_drop(initial_cell_states)
        ##print("[tlog] initial_hidden_states: " + str(initial_hidden_states))
        return initial_hidden_states, initial_cell_states


