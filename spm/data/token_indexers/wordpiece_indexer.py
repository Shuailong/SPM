#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author: Shuailong
# @Email: liangshuailong@gmail.com
# @Date:   2019-04-02 17:57:25
# @Last Modified by:  Shuailong
# @Last Modified time: 2019-04-07 15:14:57

# pylint: disable=no-self-use
from typing import List
import logging


from spm.data.tokenizers import BertTokenizer

from allennlp.data.token_indexers.token_indexer import TokenIndexer
from allennlp.data.token_indexers import WordpieceIndexer

logger = logging.getLogger(__name__)


@TokenIndexer.register("bert-pretrained-sl")
class PretrainedBertIndexer(WordpieceIndexer):
    # pylint: disable=line-too-long
    """
    A ``TokenIndexer`` corresponding to a pretrained BERT model.

    Parameters
    ----------
    pretrained_model: ``str``
        Either the name of the pretrained model to use (e.g. 'bert-base-uncased'),
        or the path to the .txt file with its vocabulary.

        If the name is a key in the list of pretrained models at
        https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/pytorch_pretrained_bert/tokenization.py#L33
        the corresponding path will be used; otherwise it will be interpreted as a path or URL.
    use_starting_offsets: bool, optional (default: False)
        By default, the "offsets" created by the token indexer correspond to the
        last wordpiece in each word. If ``use_starting_offsets`` is specified,
        they will instead correspond to the first wordpiece in each word.
    do_lowercase: ``bool``, optional (default = True)
        Whether to lowercase the tokens before converting to wordpiece ids.
    never_lowercase: ``List[str]``, optional
        Tokens that should never be lowercased. Default is
        ['[UNK]', '[SEP]', '[PAD]', '[CLS]', '[MASK]'].
    max_pieces: int, optional (default: 512)
        The BERT embedder uses positional embeddings and so has a corresponding
        maximum length for its input ids. Currently any inputs longer than this
        will be truncated. If this behavior is undesirable to you, you should
        consider filtering them out in your dataset reader.
    """

    def __init__(self,
                 pretrained_model: str,
                 use_starting_offsets: bool = False,
                 do_lowercase: bool = True,
                 never_lowercase: List[str] = None,
                 max_pieces: int = 512,
                 start_tokens: List[str] = None,
                 end_tokens: List[str] = None,
                 separator_token: str = "[SEP]") -> None:
        if pretrained_model.endswith("-cased") and do_lowercase:
            logger.warning("Your BERT model appears to be cased, "
                           "but your indexer is lowercasing tokens.")
        elif pretrained_model.endswith("-uncased") and not do_lowercase:
            logger.warning("Your BERT model appears to be uncased, "
                           "but your indexer is not lowercasing tokens.")

        bert_tokenizer = BertTokenizer.from_pretrained(pretrained_model, do_lower_case=do_lowercase, do_basic_tokenize=False)
        super().__init__(vocab=bert_tokenizer.vocab,
                         wordpiece_tokenizer=bert_tokenizer.wordpiece_tokenizer.tokenize,
                         namespace="bert",
                         use_starting_offsets=use_starting_offsets,
                         max_pieces=max_pieces,
                         do_lowercase=do_lowercase,
                         never_lowercase=never_lowercase,
                         start_tokens=start_tokens,
                         end_tokens=end_tokens,
                         separator_token=separator_token)

