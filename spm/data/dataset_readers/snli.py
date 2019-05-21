#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author: Shuailong
# @Email: liangshuailong@gmail.com
# @Date:   2019-04-06 17:08:20
# @Last Modified by:  Shuailong
# @Last Modified time: 2019-04-07 00:01:36

from typing import Dict
import json
import logging

from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, LabelField, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Tokenizer, WordTokenizer, Token

from spm.data.fields import WeightField

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("mysnli")
class SnliReader(DatasetReader):
    """
    Reads a file from the Stanford Natural Language Inference (SNLI) dataset.  This data is
    formatted as jsonl, one json-formatted instance per line.  The keys in the data are
    "gold_label", "sentence1", and "sentence2".  We convert these keys into fields named "tokens"
    and "label", along with a metadata field containing the tokenized strings of the
    premise and hypothesis. The "token" fields contains the concatenation of tokens of premise and
    hypothesis, joined by "[SEP]" token, to be used in BERT model.

    Parameters
    ----------
    tokenizer : ``Tokenizer``, optional (default=``WordTokenizer()``)
        We use this ``Tokenizer`` for both the premise and the hypothesis.  See :class:`Tokenizer`.
    token_indexers : ``Dict[str, TokenIndexer]``, optional (default=``{"tokens": SingleIdTokenIndexer()}``)
        We similarly use this for both the premise and the hypothesis.  See :class:`TokenIndexer`.
    """

    def __init__(self,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 mode: str = "merge",
                 weighted_training: bool = False,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {
            'tokens': SingleIdTokenIndexer()}
        self.mode = mode
        self.weighted_training = weighted_training

    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        with open(file_path, 'r') as snli_file:
            logger.info(
                "Reading SNLI instances from jsonl dataset at: %s", file_path)
            for line in snli_file:
                example = json.loads(line)

                label = example["gold_label"]
                if label == '-':
                    # These were cases where the annotators disagreed; we'll just skip them.  It's
                    # like 800 out of 500k examples in the training data.
                    continue

                premise = example["sentence1"]
                hypothesis = example["sentence2"]

                if self.weighted_training:
                    annotator_labels = example["annotator_labels"]
                    if len(annotator_labels) == 1:
                        label_confidence = 1
                    else:
                        label_confidence = annotator_labels.count(label)\
                            / len(annotator_labels)
                else:
                    label_confidence = None

                yield self.text_to_instance(premise, hypothesis, label, label_confidence)

    @overrides
    def text_to_instance(self,  # type: ignore
                         premise: str,
                         hypothesis: str,
                         label: str = None,
                         label_confidence: float = None) -> Instance:
        # pylint: disable=arguments-differ
        fields: Dict[str, Field] = {}
        premise_tokens = self._tokenizer.tokenize(premise)
        hypothesis_tokens = self._tokenizer.tokenize(hypothesis)

        if self.mode == 'merge':
            sentence_pair_tokens = premise_tokens + \
                [Token("[SEP]")] + hypothesis_tokens
            fields['tokens'] = TextField(
                sentence_pair_tokens, self._token_indexers)
        else:
            fields['s1'] = TextField(premise_tokens, self._token_indexers)
            fields['s2'] = TextField(hypothesis_tokens, self._token_indexers)
        if label:
            fields['label'] = LabelField(label)
        if label_confidence:
            fields['weight'] = WeightField(label_confidence)

        metadata = {"premise_tokens": [x.text for x in premise_tokens],
                    "hypothesis_tokens": [x.text for x in hypothesis_tokens]}
        fields["metadata"] = MetadataField(metadata)
        return Instance(fields)
