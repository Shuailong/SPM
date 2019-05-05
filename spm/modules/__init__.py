#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author: Shuailong
# @Email: liangshuailong@gmail.com
# @Date:   2019-03-01 10:23:59
# @Last Modified by:  Shuailong
# @Last Modified time: 2019-04-10 12:12:43
from allennlp.modules.seq2seq_encoders.seq2seq_encoder import Seq2SeqEncoder

from spm.modules.slstm import SLSTMEncoder
# from spm.modules.allennlp_transformer import BidirectionalLanguageModelTransformer
# from spm.modules.openai_transformer import OpenaiTransformer
# from spm.modules.bert_token_embedder import PretrainedBertEmbedder
from spm.modules.lstm import LSTMEncoder
# # fastNLP implementation
# from spm.modules.star_transformer import StarTransformerEncoder
# from spm.modules.transformer import TransformerEncoder

# # zeeeyang implementation
# from spm.modules.slstm_zy import SLSTMEncoderZY
# from spm.modules.pair_slstm_zy import SentencePairSLSTMEncoderZY

# Seq2SeqEncoder.register("allennlp_transformer")(
#     BidirectionalLanguageModelTransformer
# )
# Seq2SeqEncoder.register("openai_transformer")(
#     OpenaiTransformer
# )
Seq2SeqEncoder.register("slstm")(
    SLSTMEncoder
)
# Seq2SeqEncoder.register("star_transformer")(
#     StarTransformerEncoder
# )
# Seq2SeqEncoder.register("fast_transformer")(
#     TransformerEncoder
# )
