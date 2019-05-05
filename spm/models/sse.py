#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author: Shuailong
# @Email: liangshuailong@gmail.com
# @Date:   2019-03-01 10:11:48
# @Last Modified by:  Shuailong
# @Last Modified time: 2019-03-20 16:42:54

from typing import Dict, Optional, List, Any
from overrides import overrides
import torch

# from allennlp.common.checks import check_dimensions_match
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import FeedForward
from allennlp.modules import TextFieldEmbedder
from allennlp.modules import Seq2SeqEncoder
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import get_text_field_mask, masked_max
from allennlp.training.metrics import CategoricalAccuracy


@Model.register("sse")
class StackBiLSTMMaxout(Model):
    """
    This ``Model`` implements the StackBiLSTMMaxout(SSE) model (https://arxiv.org/abs/1708.02312)

    Parameters
    ----------
    vocab : ``Vocabulary``
    text_field_embedder : ``TextFieldEmbedder``
        Used to embed the ``premise`` and ``hypothesis`` ``TextFields`` we get as input to the
        model.
    encoder1 : ``Seq2SeqEncoder``
        Used to encode the premise and hypothesis using LSTM with shortcut.
    encoder2 : ``Seq2SeqEncoder``
        Used to encode the premise and hypothesis using LSTM with shortcut.
    encoder3 : ``Seq2SeqEncoder``
        Used to encode the premise and hypothesis using LSTM with shortcut.    
    output_feedforward1 : ``FeedForward``
        Used to prepare the concatenated premise and hypothesis for prediction.
    output_feedforward2 : ``FeedForward``
        Used to prepare the concatenated premise and hypothesis for prediction.
    output_logit : ``FeedForward``
        This feedforward network computes the output logits.
    dropout : ``float``, optional (default=0.5)
        Dropout percentage to use.
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.
    """

    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 encoder1: Seq2SeqEncoder,
                 encoder2: Seq2SeqEncoder,
                 encoder3: Seq2SeqEncoder,
                 output_feedforward1: FeedForward,
                 output_feedforward2: FeedForward,
                 output_logit: FeedForward,
                 dropout: float = 0.5,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super().__init__(vocab, regularizer)

        self._text_field_embedder = text_field_embedder

        self._encoder1 = encoder1
        self._encoder2 = encoder2
        self._encoder3 = encoder3

        if dropout:
            self.dropout = torch.nn.Dropout(dropout)
        else:
            self.dropout = None

        self._output_feedforward1 = output_feedforward1
        self._output_feedforward2 = output_feedforward2
        self._output_logit = output_logit

        self._num_labels = vocab.get_vocab_size(namespace="labels")

        self._accuracy = CategoricalAccuracy()
        self._loss = torch.nn.CrossEntropyLoss()

        initializer(self)

    @overrides
    def forward(self,  # type: ignore
                premise: Dict[str, torch.LongTensor],
                hypothesis: Dict[str, torch.LongTensor],
                label: torch.IntTensor = None,
                metadata: List[Dict[str, Any]
                               ] = None  # pylint:disable=unused-argument
                ) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        premise : Dict[str, torch.LongTensor]
            From a ``TextField``
        hypothesis : Dict[str, torch.LongTensor]
            From a ``TextField``
        label : torch.IntTensor, optional (default = None)
            From a ``LabelField``
        metadata : ``List[Dict[str, Any]]``, optional, (default = None)
            Metadata containing the original tokenization of the premise and
            hypothesis with 'premise_tokens' and 'hypothesis_tokens' keys respectively.

        Returns
        -------
        An output dictionary consisting of:

        label_logits : torch.FloatTensor
            A tensor of shape ``(batch_size, num_labels)`` representing unnormalised log
            probabilities of the entailment label.
        label_probs : torch.FloatTensor
            A tensor of shape ``(batch_size, num_labels)`` representing probabilities of the
            entailment label.
        loss : torch.FloatTensor, optional
            A scalar loss to be optimised.
        """
        embedded_premise = self._text_field_embedder(premise)
        embedded_hypothesis = self._text_field_embedder(hypothesis)
        premise_mask = get_text_field_mask(premise).float()
        hypothesis_mask = get_text_field_mask(hypothesis).float()

        s1_layer_1_out = self._encoder1(embedded_premise, premise_mask)
        s2_layer_1_out = self._encoder1(embedded_hypothesis, hypothesis_mask)

        s1_layer_2_out = self._encoder2(
            torch.cat([embedded_premise, s1_layer_1_out], dim=2), premise_mask)
        s2_layer_2_out = self._encoder2(
            torch.cat([embedded_hypothesis, s2_layer_1_out], dim=2), hypothesis_mask)

        s1_layer_3_out = self._encoder3(torch.cat(
            [embedded_premise, s1_layer_1_out, s1_layer_2_out], dim=2), premise_mask)
        s2_layer_3_out = self._encoder3(torch.cat(
            [embedded_hypothesis, s2_layer_1_out, s2_layer_2_out], dim=2), hypothesis_mask)

        premise_max = masked_max(s1_layer_3_out, premise_mask.unsqueeze(-1))
        hypothesis_max = masked_max(
            s2_layer_3_out, hypothesis_mask.unsqueeze(-1))

        features = torch.cat([premise_max,
                              hypothesis_max,
                              torch.abs(premise_max - hypothesis_max),
                              premise_max * hypothesis_max],
                             dim=1)

        # the final MLP -- apply dropout to input, and MLP applies to output & hidden
        output_hidden1 = self._output_feedforward1(features)
        output_hidden2 = self._output_feedforward2(output_hidden1)
        label_logits = self._output_logit(output_hidden2)
        label_probs = torch.nn.functional.softmax(label_logits, dim=-1)

        output_dict = {"label_logits": label_logits,
                       "label_probs": label_probs}

        if label is not None:
            loss = self._loss(label_logits, label.long().view(-1))
            self._accuracy(label_logits, label)
            output_dict["loss"] = loss

        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {'accuracy': self._accuracy.get_metric(reset)}
