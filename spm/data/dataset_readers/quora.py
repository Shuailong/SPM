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


@DatasetReader.register("quora")
class QuoraReader(DatasetReader):
    """
    Reads a file from the Quora Paraphrase dataset. The train/validation/test split of the data
    comes from the paper `Bilateral Multi-Perspective Matching for Natural Language Sentences
    <https://arxiv.org/abs/1702.03814>`_ by Zhiguo Wang et al., 2017. Each file of the data
    is a tsv file without header. The columns are is_duplicate, question1, question2, and id.
    All questions are pre-tokenized and tokens are space separated. We convert these keys into 
    fields named "tokens" and "label", along with a metadata field containing the tokenized string
    of the s1 and s2. The "token" fields contains the concatenation of tokens of 
    s1 and s2, joined by "[SEP]" token, to be used in BERT model.

    Parameters
    ----------
    tokenizer : ``Tokenizer``, optional (default=``WordTokenizer()``)
        We use this ``Tokenizer`` for both the s1 and the s2.  See :class:`Tokenizer`.
    token_indexers : ``Dict[str, TokenIndexer]``, optional (default=``{"tokens": SingleIdTokenIndexer()}``)
        We similarly use this for both the s1 and the s2.  See :class:`TokenIndexer`.
    """

    def __init__(self,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False,
                 mode: str = 'merge') -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {
            'tokens': SingleIdTokenIndexer()}
        self.mode = mode

    @overrides
    def _read(self, file_path):
        logger.info(
            "Reading quora instances from lines in file at: %s", file_path)
        with open(cached_path(file_path), "r") as data_file:
            tsv_in = csv.reader(data_file, delimiter='\t')
            for row in tsv_in:
                if len(row) == 4:
                    yield self.text_to_instance(s1=row[1], s2=row[2], label=row[0])

    @overrides
    def text_to_instance(self,  # type: ignore
                         s1: str,
                         s2: str,
                         label: str = None) -> Instance:
        # pylint: disable=arguments-differ
        fields: Dict[str, Field] = {}
        s1_tokens = self._tokenizer.tokenize(s1)
        s2_tokens = self._tokenizer.tokenize(s2)

        if self.mode == 'merge':
            sentence_pair_tokens = s1_tokens + \
                [Token("[SEP]")] + s2_tokens
            fields['tokens'] = TextField(
                sentence_pair_tokens, self._token_indexers)
        else:
            fields['s1'] = TextField(
                s1_tokens, self._token_indexers)
            fields['s2'] = TextField(
                s1_tokens, self._token_indexers)

        if label:
            fields['label'] = LabelField(label)

        metadata = {"s1_tokens": [x.text for x in s1_tokens],
                    "s2_tokens": [x.text for x in s2_tokens]}
        fields["metadata"] = MetadataField(metadata)
        return Instance(fields)
