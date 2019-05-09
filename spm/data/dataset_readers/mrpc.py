#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author: Shuailong
# @Email: liangshuailong@gmail.com
# @Date: 2019-04-30 17:43:29
# @Last Modified by: Shuailong
# @Last Modified time: 2019-04-30 21:00:17


from typing import Dict, Union
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


@DatasetReader.register("mrpc")
class MRPCReader(DatasetReader):
    """
    Parameters
    ----------
    tokenizer : ``Tokenizer``, optional (default=``WordTokenizer()``)
        We use this ``Tokenizer`` for both the premise and the s2.  See :class:`Tokenizer`.
    token_indexers : ``Dict[str, TokenIndexer]``, optional (default=``{"tokens": SingleIdTokenIndexer()}``)
        We similarly use this for both the premise and the s2.  See :class:`TokenIndexer`.
    mode: ``str``, ['merge', 'seperate'], whether to return merged s1+s2 tokens or seperate s1, s2 tokens.
    """

    def __init__(self,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False,
                 skip_label_indexing: bool = False,
                 mode: str = 'merge') -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {
            'tokens': SingleIdTokenIndexer()}
        self._skip_label_indexing = skip_label_indexing
        self.mode = mode

    @overrides
    def _read(self, file_path):
        logger.info(
            "Reading instances from lines in file at: %s", file_path)
        with open(cached_path(file_path), "r") as data_file:
            for i, line in enumerate(data_file):
                row = line.strip().split('\t')
                if i == 0:
                    continue
                if len(row) == 5:
                    label = row[0]
                    s1, s2 = row[3], row[4]
                    if self._skip_label_indexing:
                        try:
                            label = int(row[0])
                        except ValueError:
                            raise ValueError(
                                'Labels must be integers if skip_label_indexing is True.')
                    yield self.text_to_instance(s1=s1, s2=s2, label=label)
                else:
                    raise ValueError(f'Malformed input in file {file_path}!')

    @overrides
    def text_to_instance(self,  # type: ignore
                         s1: str,
                         s2: str,
                         label: Union[int, str] = None) -> Instance:
        # pylint: disable=arguments-differ
        fields: Dict[str, Field] = {}
        s1_tokens = self._tokenizer.tokenize(s1)
        s2_tokens = self._tokenizer.tokenize(s2)
        tokens = s1_tokens + [Token("[SEP]")] + s2_tokens

        if self.mode == 'merge':
            fields['tokens'] = TextField(tokens, self._token_indexers)
        else:
            fields['s1'] = TextField(s1_tokens, self._token_indexers)
            fields['s2'] = TextField(s2_tokens, self._token_indexers)

        if label is not None:
            fields['label'] = LabelField(
                label, skip_indexing=self._skip_label_indexing)

        metadata = {"s1_tokens": [x.text for x in s1_tokens],
                    "s2_tokens": [x.text for x in s2_tokens]}
        fields["metadata"] = MetadataField(metadata)
        return Instance(fields)
