# pylint: disable=no-self-use,invalid-name
import pytest
import pathlib
import random
from allennlp.common import Params
from allennlp.common.util import ensure_list
from allennlp.common.testing import ModelTestCase
from allennlp.data.dataset import Batch
from allennlp.data.fields import TextField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers.wordpiece_indexer import PretrainedBertIndexer
from allennlp.data.tokenizers import WordTokenizer, Token
from allennlp.data.tokenizers.word_splitter import BertBasicWordSplitter
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.token_embedders.bert_token_embedder import PretrainedBertEmbedder

from spm.data.dataset_readers import SnliReader


class TestSNLIReader:
    FIXTURES_ROOT = (pathlib.Path(__file__).parent /
                     ".." / ".." / "tests" / "fixtures").resolve()

    @pytest.mark.parametrize("lazy", (True, False))
    def test_read(self, lazy):
        reader = SnliReader(
            tokenizer=WordTokenizer(word_splitter=BertBasicWordSplitter()),
            token_indexers={'bert': PretrainedBertIndexer(
                pretrained_model="data/bert/bert-base-uncased-vocab.txt")},
        )

        instances = reader.read(
            str(self.FIXTURES_ROOT / 'snli_1.0_sample.jsonl'))
        instances = ensure_list(instances)
        example = instances[0]
        tokens = [t.text for t in example.fields['tokens'].tokens]
        label = example.fields['label'].label
        print(label)
        print(tokens)
        batch = Batch(instances)
        vocab = Vocabulary.from_instances(instances)
        batch.index_instances(vocab)

        padding_lengths = batch.get_padding_lengths()
        tensor_dict = batch.as_tensor_dict(padding_lengths)
        tokens = tensor_dict["tokens"]

        print(tokens['mask'].tolist()[0])
        print(tokens["bert"].tolist()[0])
        print([vocab.get_token_from_index(i, "bert")
               for i in tokens["bert"].tolist()[0]])
        print(len(tokens['bert'][0]))
        print(tokens["bert-offsets"].tolist()[0])
        print(tokens['bert-type-ids'].tolist()[0])

    def test_can_build_from_params(self):
        reader = SnliReader.from_params(Params({}))
        # pylint: disable=protected-access
        assert reader._token_indexers['tokens'].__class__.__name__ == 'SingleIdTokenIndexer'
