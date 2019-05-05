# pylint: disable=no-self-use,invalid-name
import pytest
import pathlib

from spm.data.dataset_readers import MRPCReader
from allennlp.common.util import ensure_list
from allennlp.common.testing import AllenNlpTestCase
from allennlp.data.tokenizers import WordTokenizer
from allennlp.data.tokenizers.word_splitter import BertBasicWordSplitter, JustSpacesWordSplitter
from allennlp.data.token_indexers.wordpiece_indexer import PretrainedBertIndexer


class TestMRPCReader():

    FIXTURES_ROOT = (pathlib.Path(__file__).parent /
                     ".." / ".." / "tests" / "fixtures").resolve()

    @pytest.mark.parametrize("lazy", (True, False))
    def test_read_from_file(self, lazy):
        reader = MRPCReader(tokenizer=WordTokenizer(word_splitter=JustSpacesWordSplitter()),
                            token_indexers={"bert":
                                            PretrainedBertIndexer(pretrained_model='data/bert/bert-base-uncased-vocab.txt')},
                            lazy=lazy,
                            mode='seperate',
                            skip_label_indexing=False)
        instances = reader.read(
            str(self.FIXTURES_ROOT / 'mrpc_dev.tsv'))
        instances = ensure_list(instances)

        instance1 = {"s1": "He said the foodservice pie business doesn 't fit the company 's long-term growth strategy .".split(),
                     "s2": "\" The foodservice pie business does not fit our long-term growth strategy .".split(),
                     "label": '1'}

        instance2 = {"s1": "Magnarelli said Racicot hated the Iraqi regime and looked forward to using his long years of training in the war .".split(),
                     "s2": "His wife said he was \" 100 percent behind George Bush \" and looked forward to using his years of training in the war .".split(),
                     "label": '0'}

        instance3 = {"s1": "The dollar was at 116.92 yen against the yen , flat on the session , and at 1.2891 against the Swiss franc , also flat .".split(),
                     "s2": "The dollar was at 116.78 yen JPY = , virtually flat on the session , and at 1.2871 against the Swiss franc CHF = , down 0.1 percent .".split(),
                     "label": '0'}

        for instance, expected_instance in zip(instances, [instance1, instance2, instance3]):
            fields = instance.fields
            assert [
                t.text for t in fields["s1"].tokens] == expected_instance["s1"]
            assert [
                t.text for t in fields["s2"].tokens] == expected_instance["s2"]
            assert fields["label"].label == expected_instance["label"]

    @pytest.mark.parametrize("lazy", (True, False))
    def test_read_from_file(self, lazy):
        reader = MRPCReader(tokenizer=WordTokenizer(word_splitter=JustSpacesWordSplitter()),
                            token_indexers={"bert":
                                            PretrainedBertIndexer(pretrained_model='data/bert/bert-base-uncased-vocab.txt')},
                            lazy=lazy,
                            skip_label_indexing=False,
                            mode='merge')
        instances = reader.read(
            str(self.FIXTURES_ROOT / 'mrpc_dev.tsv'))
        instances = ensure_list(instances)

        instance1 = {"tokens": "He said the foodservice pie business doesn 't fit the company 's long-term growth strategy .".split() + ["[SEP]"] +
                     "\" The foodservice pie business does not fit our long-term growth strategy .".split(),
                     "label": '1'}

        instance2 = {"tokens": "Magnarelli said Racicot hated the Iraqi regime and looked forward to using his long years of training in the war .".split() + ["[SEP]"] +
                     "His wife said he was \" 100 percent behind George Bush \" and looked forward to using his years of training in the war .".split(),
                     "label": '0'}

        instance3 = {"tokens": "The dollar was at 116.92 yen against the yen , flat on the session , and at 1.2891 against the Swiss franc , also flat .".split() + ["[SEP]"] +
                     "The dollar was at 116.78 yen JPY = , virtually flat on the session , and at 1.2871 against the Swiss franc CHF = , down 0.1 percent .".split(),
                     "label": '0'}

        for instance, expected_instance in zip(instances, [instance1, instance2, instance3]):
            fields = instance.fields
            assert [
                t.text for t in fields["tokens"].tokens] == expected_instance["tokens"]
            assert fields["label"].label == expected_instance["label"]
