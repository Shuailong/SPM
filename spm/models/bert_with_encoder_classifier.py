#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author: Shuailong
# @Email: liangshuailong@gmail.com
# @Date:   2019-04-05 22:00:49
# @Last Modified by:  Shuailong
# @Last Modified time: 2019-04-09 13:56:57

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
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.nn.util import masked_max


@Model.register("bert_with_encoder_classifier")
class BertEncoderClassifier(Model):
    """
    This ``Model`` implements the BertEncoderClassifier model...

    Parameters
    ----------
    vocab : ``Vocabulary``
    bert : ``TextFieldEmbedder``
        Used to embed the ``tokens`` ``TextFields`` we get as input to the model.
    encoder : ``Seq2SeqEncoder``
        Used to encode the output of the embedding layer
    aggregation: ``str``
        Choose from ['cls', 'max', 'cls+max']
    classifier : ``FeedForward``
        This feedforward network computes the output logits.
    dropout : ``float``, optional (default=0.1)
        Dropout percentage to use.
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.
    """

    def __init__(self, vocab: Vocabulary,
                 bert: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 classifier: FeedForward,
                 aggregation: str = 'cls+max',
                 dropout: float = 0.1,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super().__init__(vocab, regularizer)

        self._bert = bert
        self._encoder = encoder
        self._aggregation = aggregation
        self._dropout = torch.nn.Dropout(dropout)
        self._classifier = classifier
        self._num_labels = vocab.get_vocab_size(namespace="labels")
        self._pooler = FeedForward(input_dim=bert.get_output_dim(),
                                   num_layers=1,
                                   hidden_dims=bert.get_output_dim(),
                                   activations=torch.tanh)

        check_dimensions_match(bert.get_output_dim(), encoder.get_input_dim(),
                               "bert output dim", "encoder input dim")
        check_dimensions_match(encoder.get_output_dim() * 2, classifier.get_input_dim(),
                               "encoder output dim", "classifier input dim")
        check_dimensions_match(classifier.get_output_dim(), self._num_labels,
                               "classifier output dim", "number of labels")

        self._accuracy = CategoricalAccuracy()
        self._loss = torch.nn.CrossEntropyLoss()

        initializer(self)

    @overrides
    def forward(self,  # type: ignore
                tokens: Dict[str, torch.LongTensor],
                label: torch.IntTensor = None,
                metadata: List[Dict[str, Any]
                               ] = None  # pylint:disable=unused-argument
                ) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        tokens : Dict[str, torch.LongTensor]
            From a ``TextField``
        label : torch.IntTensor, optional (default = None)
            From a ``LabelField``
        metadata : ``List[Dict[str, Any]]``, optional, (default = None)
            Metadata containing the original tokenization of the text.

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
        embedded = self._bert(tokens)

        first_token = embedded[:, 0, :]
        pooled_first = self._pooler(first_token)
        pooled_first = self._dropout(pooled_first)

        mask = tokens['mask'].float()
        encoded = self._encoder(embedded, mask)
        encoded = self._dropout(encoded)
        pooled_encoded = masked_max(encoded, mask.unsqueeze(-1), dim=1)

        concat = torch.cat([pooled_first, pooled_encoded], dim=-1)
        label_logits = self._classifier(concat)
        label_probs = torch.nn.functional.softmax(label_logits, dim=-1)

        output_dict = {"label_logits": label_logits,
                       "label_probs": label_probs}

        if label is not None:
            loss = self._loss(
                label_logits.view(-1, self._num_labels), label.view(-1))
            self._accuracy(label_logits, label)
            output_dict["loss"] = loss

        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {'accuracy': self._accuracy.get_metric(reset)}
