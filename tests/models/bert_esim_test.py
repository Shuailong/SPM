# pylint: disable=no-self-use,invalid-name
import pathlib

import numpy
from numpy.testing import assert_almost_equal

from allennlp.common.testing import ModelTestCase


class TestESIM(ModelTestCase):
    FIXTURES_ROOT = (pathlib.Path(__file__).parent /
                     ".." / ".." / "tests" / "fixtures").resolve()

    def setUp(self):
        super(TestESIM, self).setUp()
        self.set_up_model(self.FIXTURES_ROOT / 'bert_esim.jsonnet',
                          self.FIXTURES_ROOT / 'snli_5.jsonl')

    def test_forward_pass_runs_correctly(self):
        training_tensors = self.dataset.as_tensor_dict()
        output_dict = self.model(**training_tensors)
        assert_almost_equal(
            numpy.sum(output_dict["label_probs"][0].data.numpy(), -1), 1, decimal=6)
