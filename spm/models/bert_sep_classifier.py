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
from allennlp.modules import InputVariationalDropout
from allennlp.modules.similarity_functions import DotProductSimilarity, SimilarityFunction
from allennlp.modules.matrix_attention.legacy_matrix_attention import LegacyMatrixAttention
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.training.metrics import CategoricalAccuracy, F1Measure
from allennlp.nn.util import get_text_field_mask, masked_softmax, weighted_sum, replace_masked_values
from allennlp.nn.util import masked_max


@Model.register("bert_sep_sequence_classifier")
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
                 similarity_function: SimilarityFunction,
                 projection_feedforward: FeedForward,
                 inference_encoder: Seq2SeqEncoder,
                 output_feedforward: FeedForward,
                 classifier: FeedForward,
                 dropout: float = 0.1,
                 num_labels: int = None,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super().__init__(vocab, regularizer)

        self._bert = bert
        self._dropout = torch.nn.Dropout(dropout)
        self._matrix_attention = LegacyMatrixAttention(similarity_function)
        self._projection_feedforward = projection_feedforward

        self._inference_encoder = inference_encoder

        if dropout:
            self.dropout = torch.nn.Dropout(dropout)
            self.rnn_input_dropout = InputVariationalDropout(dropout)
        else:
            self.dropout = None
            self.rnn_input_dropout = None

        self._output_feedforward = output_feedforward
        self._classifier = classifier
        if num_labels is None:
            self._num_labels = vocab.get_vocab_size(namespace="labels")
        else:
            self._num_labels = num_labels

        check_dimensions_match(bert.get_output_dim() * 4, projection_feedforward.get_input_dim(),
                               "bert output dim", "projection_feedforward input dim")
        check_dimensions_match(projection_feedforward.get_output_dim(), inference_encoder.get_input_dim(),
                               "proj feedforward output dim", "inference lstm input dim")
        check_dimensions_match(inference_encoder.get_output_dim() * 4, output_feedforward.get_input_dim(),
                               "inference encoder output dim", "output feedforward input dim")
        check_dimensions_match(output_feedforward.get_output_dim(), classifier.get_input_dim(),
                               "output feedforward output dim", "classifier input dim")
        check_dimensions_match(classifier.get_output_dim(), self._num_labels,
                               "classifier output dim", "number of labels")

        self._accuracy = CategoricalAccuracy()
        self._f1 = F1Measure(positive_label=1)
        self._loss = torch.nn.CrossEntropyLoss()

        initializer(self)

    @overrides
    def forward(self,  # type: ignore
                s1: Dict[str, torch.LongTensor] = None,
                s2: Dict[str, torch.LongTensor] = None,
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
        embedded_s1 = self._bert(s1)
        embedded_s2 = self._bert(s2)
        mask_s1 = s1['mask'].float()
        mask_s2 = s2['mask'].float()

        # Shape: (batch_size,s1_length, s2_length)
        similarity_matrix = self._matrix_attention(
            embedded_s1, embedded_s2)

        # Shape: (batch_size,s1_length, s2_length)
        p2h_attention = masked_softmax(similarity_matrix, mask_s2)
        # Shape: (batch_size,s1_length, embedding_dim)
        attended_s2 = weighted_sum(
            embedded_s2, p2h_attention)

        # Shape: (batch_size, s2_length,s1_length)
        h2p_attention = masked_softmax(
            similarity_matrix.transpose(1, 2).contiguous(), mask_s1)
        # Shape: (batch_size, s2_length, embedding_dim)
        attended_s1 = weighted_sum(embedded_s1, h2p_attention)

        # the "enhancement" layer
        s1_enhanced = torch.cat(
            [embedded_s1, attended_s2,
                embedded_s1 - attended_s2,
                embedded_s1 * attended_s2],
            dim=-1
        )
        s2_enhanced = torch.cat(
            [embedded_s2, attended_s1,
                embedded_s2 - attended_s1,
                embedded_s2 * attended_s1],
            dim=-1
        )
        # The projection layer down to the model dimension.  Dropout is not applied before
        # projection.
        projected_enhanced_s1 = self._projection_feedforward(
            s1_enhanced)
        projected_enhanced_s2 = self._projection_feedforward(
            s2_enhanced)

        # Run the inference layer
        if self.rnn_input_dropout:
            projected_enhanced_s1 = self.rnn_input_dropout(
                projected_enhanced_s1)
            projected_enhanced_s2 = self.rnn_input_dropout(
                projected_enhanced_s2)
        v_ai = self._inference_encoder(
            projected_enhanced_s1, mask_s1)
        v_bi = self._inference_encoder(
            projected_enhanced_s2, mask_s2)

        s1_feats = masked_max(v_ai, mask_s1.unsqueeze(-1), dim=1)
        s2_feats = masked_max(v_bi, mask_s2.unsqueeze(-1), dim=1)
        v_all = torch.cat([s1_feats,
                           s2_feats,
                           torch.abs(s1_feats - s2_feats),
                           s1_feats * s2_feats], dim=-1)

        # # The pooling layer -- max and avg pooling.
        # # (batch_size, model_dim)
        # v_a_max, _ = replace_masked_values(
        #     v_ai, mask_s1.unsqueeze(-1), -1e7
        # ).max(dim=1)
        # v_b_max, _ = replace_masked_values(
        #     v_bi, mask_s2.unsqueeze(-1), -1e7
        # ).max(dim=1)

        # v_a_avg = torch.sum(v_ai * mask_s1.unsqueeze(-1), dim=1) / torch.sum(
        #     mask_s1, 1, keepdim=True
        # )
        # v_b_avg = torch.sum(v_bi * mask_s2.unsqueeze(-1), dim=1) / torch.sum(
        #     mask_s2, 1, keepdim=True
        # )

        # # Now concat
        # # (batch_size, model_dim * 2 * 4)
        # v_all = torch.cat([v_a_avg, v_a_max, v_b_avg, v_b_max], dim=1)

        # the final MLP -- apply dropout to input, and MLP applies to output & hidden
        if self.dropout:
            v_all = self.dropout(v_all)

        output_hidden = self._output_feedforward(v_all)
        label_logits = self._classifier(output_hidden)
        label_probs = torch.nn.functional.softmax(label_logits, dim=-1)

        output_dict = {"label_logits": label_logits,
                       "label_probs": label_probs}

        if label is not None:
            loss = self._loss(
                label_logits.view(-1, self._num_labels), label.view(-1))
            self._accuracy(label_logits, label)
            self._f1(label_logits, label)
            output_dict["loss"] = loss

        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        p, r, f1 = self._f1.get_metric(reset)
        return {'accuracy': self._accuracy.get_metric(reset),
                'f1': f1,
                'precision': p,
                'recall': r}
