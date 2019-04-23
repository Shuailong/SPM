from typing import Dict, List, Union
import logging

import csv
from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import LabelField, TextField, Field
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token, WordTokenizer, Tokenizer
from allennlp.common.checks import ConfigurationError

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("glue_sst2")
class GLUESST2DatasetReader(DatasetReader):
    """
    Reads tokens and their sentiment labels from the GLUE SST-2.

    The output of ``read`` is a list of ``Instance`` s with the fields:
        tokens: ``TextField`` and
        label: ``LabelField``

    Parameters
    ----------
    skip_label_indexing: ``bool``, optional (default = ``False``)
        Whether or not to skip label indexing. You might want to skip label indexing if your
        labels are numbers, so the dataset reader doesn't re-number them starting from 0.
    token_indexers : ``Dict[str, TokenIndexer]``, optional (default=``{"tokens": SingleIdTokenIndexer()}``)
        We use this to define the input representation for the text.  See :class:`TokenIndexer`.
    tokenizer : ``Tokenizer``, optional (default=``WordTokenizer()``)
        We use this ``Tokenizer`` to tokenize the tokens into wordpiece used in bert model.  See :class:`Tokenizer`.

    lazy : ``bool``, optional, (default = ``False``)
        Whether or not instances can be read lazily.
    """

    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 tokenizer: Tokenizer = None,
                 skip_label_indexing: bool = False,
                 lazy: bool = False) -> None:
        super().__init__(lazy=lazy)
        self._token_indexers = token_indexers or {
            'tokens': SingleIdTokenIndexer()}
        self._tokenizer = tokenizer or WordTokenizer()
        self._skip_label_indexing = skip_label_indexing

    @overrides
    def _read(self, file_path):
        for i, line in enumerate(self._read_tsv(cached_path(file_path))):
            if i == 0:
                continue
            text, label = line
            if self._skip_label_indexing:
                try:
                    label = int(label)
                except ValueError:
                    raise ValueError(
                        'Labels must be integers if skip_label_indexing is True.')
            instance = self.text_to_instance(text, label)
            if instance is not None:
                yield instance

    @overrides
    # type: ignore
    def text_to_instance(self, text: str, sentiment: Union[int, str] = None) -> Instance:
        """
        Parameters
        ----------
        text : ``str``, required.
            The given sentence.
        sentiment ``int``, optional, (default = None).
            The sentiment for this sentence.

        Returns
        -------
        An ``Instance`` containing the following fields:
            tokens : ``TextField``
                The tokens in the sentence or phrase.
            label : ``LabelField``
                The sentiment label of the sentence or phrase.
        """
        # pylint: disable=arguments-differ
        tokens = self._tokenizer.tokenize(text)
        text_field = TextField(tokens, self._token_indexers)
        label_field = LabelField(
            sentiment, skip_indexing=self._skip_label_indexing)
        fields: Dict[str, Field] = {"tokens": text_field,
                                    "label": label_field}
        return Instance(fields)

    def _read_tsv(self, input_file, quotechar=None):
        """Reads a tab separated value file."""
        logger.info(
            "Reading instances from lines in file at: %s", input_file)
        with open(input_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines
