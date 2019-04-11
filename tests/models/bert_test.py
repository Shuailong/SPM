# pylint: disable=no-self-use,invalid-name
from allennlp.common.testing import ModelTestCase
from allennlp.data.dataset import Batch
from allennlp.data.fields import TextField
from allennlp.data.instance import Instance
from spm.data.token_indexers.wordpiece_indexer import PretrainedBertIndexer
from allennlp.data.tokenizers import WordTokenizer, Token
from allennlp.data.tokenizers.word_splitter import BertBasicWordSplitter
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.token_embedders.bert_token_embedder import PretrainedBertEmbedder


class TestBertEmbedder(ModelTestCase):
    def setUp(self):
        super().setUp()

        vocab_path = 'data/bert/vocab.txt'
        model_path = 'data/bert/bert-base-uncased.tar.gz'
        self.token_indexer = PretrainedBertIndexer(vocab_path)
        self.token_embedder = PretrainedBertEmbedder(model_path, requires_grad=True)

    def test_end_to_end(self):
        tokenizer = WordTokenizer(word_splitter=BertBasicWordSplitter())

        #            2   3    4   3     5     6   8      9    2   14   12
        sentence1 = "the quickest quick brown fox jumped over the lazy dog"
        tokens1 = tokenizer.tokenize(sentence1)

        #            2   3     5     6   8      9    2  15 10 11 14   1
        sentence2 = "the quick brown fox jumped over the laziest lazy elmo"
        tokens2 = tokenizer.tokenize(sentence2)

        assert len(tokens1) == 10
        assert len(tokens2) == 10

        tokens = [Token('[CLS]')] + tokens1 + [Token('[SEP]')] + tokens2

        assert len(tokens) == 22

        vocab = Vocabulary()

        instance = Instance({"sentence_pair": TextField(
            tokens, {"bert": self.token_indexer})})

        batch = Batch([instance])
        batch.index_instances(vocab)

        padding_lengths = batch.get_padding_lengths()

        tensor_dict = batch.as_tensor_dict(padding_lengths)

        tokens = tensor_dict["sentence_pair"]
        assert tokens['mask'].tolist()[0] == [1] * 22
        assert tokens["bert"].tolist()[0] == [101, 1996, 4248, 4355, 4248, 2829, 4419, 5598, 2058, 1996, 13971, 3899, 102, 1996, 4248, 2829, 4419, 5598, 2058, 1996, 2474, 14272, 3367, 13971, 17709, 2080]
        assert [vocab.get_token_from_index(i, "bert") for i in tokens["bert"].tolist()[0]] == ['[CLS]', 'the', 'quick', '##est', 'quick', 'brown', 'fox', 'jumped', 'over', 'the', 'lazy', 'dog', '[SEP]', 'the', 'quick', 'brown', 'fox', 'jumped', 'over', 'the', 'la', '##zie', '##st', 'lazy', 'elm', '##o']
        assert len(tokens['bert'][0]) == 26
        assert tokens["bert-offsets"].tolist()[0] == [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 22, 23, 25]
        assert tokens['bert-type-ids'].tolist()[0] == [0] * 13 + [1] * 13

        bert_vectors = self.token_embedder(
            tokens["bert"], offsets=tokens["bert-offsets"], token_type_ids=tokens['bert-type-ids'])
        assert list(bert_vectors.shape) == [1, 22, 768]
