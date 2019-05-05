#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author: Shuailong
# @Email: liangshuailong@gmail.com
# @Date:   2019-02-27 22:09:48
# @Last Modified by:  Shuailong
# @Last Modified time: 2019-04-07 15:50:47

from typing import Dict, Optional, List, Any
from overrides import overrides
import torch

from allennlp.common.checks import check_dimensions_match
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import FeedForward
from allennlp.modules import Seq2SeqEncoder
from allennlp.modules import TextFieldEmbedder
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import get_text_field_mask, masked_max
from allennlp.training.metrics import CategoricalAccuracy


@Model.register("slstm_share")
class SLSTMShare(Model):
    """
    This ``Model`` implements the SLSTMShare model...

    Parameters
    ----------
    vocab : ``Vocabulary``
    text_field_embedder : ``TextFieldEmbedder``
        Used to embed the ``premise`` and ``hypothesis`` ``TextFields`` we get as input to the
        model.
    embedding_project: ``Feedforward``, optional
        Used to cast ELMo embeddings (1024) to lower dimensions for computation efficiency.
        (may decrease performance?)
    encoder : ``Seq2SeqEncoder``
        Used to encode the premise and hypothesis using SLSTM and GNN.
    output_feedforward : ``FeedForward``
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
                 encoder: Seq2SeqEncoder,
                 output_feedforward: FeedForward,
                 output_logit: FeedForward,
                 dropout: float = 0.5,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super().__init__(vocab, regularizer)

        self._text_field_embedder = text_field_embedder
        self._encoder = encoder

        if dropout:
            self.dropout = torch.nn.Dropout(dropout)
        else:
            self.dropout = None

        self._output_feedforward = output_feedforward
        self._output_logit = output_logit

        self._num_labels = vocab.get_vocab_size(namespace="labels")

        check_dimensions_match(text_field_embedder.get_output_dim(), encoder.get_input_dim(),
                               "text field embedding dim", "encoder input dim")
        check_dimensions_match(encoder.get_output_dim() * 4, output_feedforward.get_input_dim(),
                               "encoder output dim", "output_feedforward input dim")
        check_dimensions_match(output_feedforward.get_output_dim(), output_logit.get_input_dim(),
                               "output_feedforward output dim", "output_logit input dim")
        check_dimensions_match(output_logit.get_output_dim(), self._num_labels,
                               "output_logit output dim", "number of labels")

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
        premise_embeded = self._text_field_embedder(premise)
        hypothesis_embeded = self._text_field_embedder(hypothesis)
        premise_mask = get_text_field_mask(premise).float()
        hypothesis_mask = get_text_field_mask(hypothesis).float()

        premise_seq_len = premise_embeded.size(1)
        concat_inputs = torch.cat(
            [premise_embeded, hypothesis_embeded], dim=1)
        concat_mask = torch.cat([premise_mask, hypothesis_mask], dim=1)
        hiddens = self._encoder(concat_inputs, concat_mask, premise_seq_len)

        premise_hidden = hiddens[:, :premise_seq_len, :]
        hypothesis_hidden = hiddens[:, premise_seq_len:, :]

        premise_feats = masked_max(premise_hidden, premise_mask.unsqueeze(-1))
        hypothesis_feats = masked_max(
            hypothesis_hidden, hypothesis_mask.unsqueeze(-1))

        fusion = torch.cat([premise_feats,
                            hypothesis_feats,
                            torch.abs(premise_feats - hypothesis_feats),
                            premise_feats * hypothesis_feats], dim=-1)

        if self.dropout:
            fusion = self.dropout(fusion)

        # the final MLP -- apply dropout to input, and MLP applies to output & hidden
        output_hidden = self._output_feedforward(fusion)
        label_logits = self._output_logit(output_hidden)
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
