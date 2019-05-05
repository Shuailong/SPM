#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author: Shuailong
# @Email: liangshuailong@gmail.com
# @Date: 2019-04-16 22:10:41
# @Last Modified by: Shuailong
# @Last Modified time: 2019-05-05 20:27:54

from allennlp.modules.seq2seq_encoders.seq2seq_encoder import Seq2SeqEncoder

from spm.modules.slstm import SLSTMEncoder

Seq2SeqEncoder.register("slstm")(
    SLSTMEncoder
)
