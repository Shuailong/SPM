#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author: Shuailong
# @Email: liangshuailong@gmail.com
# @Date: 2019-04-24 15:43:30
# @Last Modified by: Shuailong
# @Last Modified time: 2019-04-24 15:43:46


from typing import Dict
import csv
import json
import logging

from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, LabelField, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Tokenizer, WordTokenizer, Token

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("quora-bert")
class QuoraReader(DatasetReader):
    """
    Reads a file from the Quora Paraphrase dataset. The train/validation/test split of the data
    comes from the paper `Bilateral Multi-Perspective Matching for Natural Language Sentences
    <https://arxiv.org/abs/1702.03814>`_ by Zhiguo Wang et al., 2017. Each file of the data
    is a tsv file without header. The columns are is_duplicate, question1, question2, and id.
    All questions are pre-tokenized and tokens are space separated. We convert these keys into 
    fields named "tokens" and "label", along with a metadata field containing the tokenized string
    of the premise and hypothesis. The "token" fields contains the concatenation of tokens of 
    premise and hypothesis, joined by "[SEP]" token, to be used in BERT model.

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
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {
            'tokens': SingleIdTokenIndexer()}

    @overrides
    def _read(self, file_path):
        logger.info(
            "Reading quora instances from lines in file at: %s", file_path)
        with open(cached_path(file_path), "r") as data_file:
            tsv_in = csv.reader(data_file, delimiter='\t')
            for row in tsv_in:
                if len(row) == 4:
                    yield self.text_to_instance(premise=row[1], hypothesis=row[2], label=row[0])

    @overrides
    def text_to_instance(self,  # type: ignore
                         premise: str,
                         hypothesis: str,
                         label: str = None) -> Instance:
        # pylint: disable=arguments-differ
        fields: Dict[str, Field] = {}
        premise_tokens = self._tokenizer.tokenize(premise)
        hypothesis_tokens = self._tokenizer.tokenize(hypothesis)
        sentence_pair_tokens = premise_tokens + \
            [Token("[SEP]")] + hypothesis_tokens
        fields['tokens'] = TextField(
            sentence_pair_tokens, self._token_indexers)
        if label:
            fields['label'] = LabelField(label)

        metadata = {"premise_tokens": [x.text for x in premise_tokens],
                    "hypothesis_tokens": [x.text for x in hypothesis_tokens]}
        fields["metadata"] = MetadataField(metadata)
        return Instance(fields)
