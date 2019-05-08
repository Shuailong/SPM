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
from allennlp.training.metrics import CategoricalAccuracy, F1Measure


@Model.register("bert_sequence_classifier")
class BertSequenceClassifier(Model):
    """
    This ``Model`` implements the BertSequenceClassifier model...

    Parameters
    ----------
    vocab : ``Vocabulary``
    bert : ``TextFieldEmbedder``
        Used to embed the ``tokens`` ``TextFields`` we get as input to the model.
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
                 classifier: FeedForward,
                 dropout: float = 0.1,
                 num_labels: int = None,
                 metrics: List[str] = ['acc'],
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super().__init__(vocab, regularizer)

        self._bert = bert
        self._dropout = torch.nn.Dropout(dropout)
        self._classifier = classifier
        if num_labels is None:
            self._num_labels = vocab.get_vocab_size(namespace="labels")
        else:
            self._num_labels = num_labels

        self._pooler = FeedForward(input_dim=bert.get_output_dim(),
                                   num_layers=1,
                                   hidden_dims=bert.get_output_dim(),
                                   activations=torch.tanh)
        check_dimensions_match(bert.get_output_dim(), classifier.get_input_dim(),
                               "bert output dim", "classifier input dim")
        check_dimensions_match(classifier.get_output_dim(), self._num_labels,
                               "classifier output dim", "number of labels")
        self.metrics = metrics
        self._accuracy = CategoricalAccuracy()
        if 'f1' in self.metrics:
            self._f1 = F1Measure(positive_label=1)
        self._loss = torch.nn.CrossEntropyLoss()

        initializer(self)

    @overrides
    def forward(self,  # type: ignore
                tokens: Dict[str, torch.LongTensor] = None,
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
        pooled_output = self._pooler(first_token)
        pooled_output = self._dropout(pooled_output)

        label_logits = self._classifier(pooled_output)
        label_probs = torch.nn.functional.softmax(label_logits, dim=-1)

        output_dict = {"label_logits": label_logits,
                       "label_probs": label_probs}

        if label is not None:
            loss = self._loss(
                label_logits.view(-1, self._num_labels), label.view(-1))
            self._accuracy(label_logits, label)
            if 'f1' in self.metrics:
                self._f1(label_logits, label)
            output_dict["loss"] = loss

        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Does a simple argmax over the probabilities, converts index to string label, and
        add ``"label"`` key to the dictionary with the result.
        """
        predictions = output_dict["label_probs"]
        if predictions.dim() == 2:
            predictions_list = [predictions[i]
                                for i in range(predictions.shape[0])]
        else:
            predictions_list = [predictions]
        classes = []
        for prediction in predictions_list:
            label_idx = prediction.argmax(dim=-1).item()
            label_str = self.vocab.get_token_from_index(
                label_idx, namespace="labels")
            classes.append(label_str)
        output_dict["label"] = classes
        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:

        acc = self._accuracy.get_metric(reset)
        metrics = {'accuracy': acc}
        if 'f1' in self.metrics:
            p, r, f1 = self._f1.get_metric(reset)
            metrics['precision'] = p
            metrics['recall'] = r
            metrics['f1'] = f1
        return metrics
