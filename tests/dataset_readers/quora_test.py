# pylint: disable=no-self-use,invalid-name
import pytest
import pathlib

from spm.data.dataset_readers import QuoraReader
from allennlp.common.util import ensure_list
from allennlp.common.testing import AllenNlpTestCase


class TestQuoraParaphraseReader():

    FIXTURES_ROOT = (pathlib.Path(__file__).parent /
                     ".." / ".." / "tests" / "fixtures").resolve()

    @pytest.mark.parametrize("lazy", (True, False))
    def test_read_from_file(self, lazy):
        reader = QuoraReader(lazy=lazy)
        instances = reader.read(
            str(self.FIXTURES_ROOT / 'quora_paraphrase.tsv'))
        instances = ensure_list(instances)

        instance1 = {"tokens": "What should I do to avoid sleeping in class ? [SEP] How do I not sleep in a boring class ?".split(),
                     "label": "1"}

        instance2 = {"tokens": "Do women support each other more than men do ? [SEP] Do women need more compliments than men ?".split(),
                     "label": "0"}

        instance3 = {"tokens": "How can one root android devices ? [SEP] How do I root an Android device ?".split(),
                     "label": "1"}

        assert len(instances) == 3

        for instance, expected_instance in zip(instances, [instance1, instance2, instance3]):
            fields = instance.fields
            assert [
                t.text for t in fields["tokens"].tokens] == expected_instance["tokens"]
            assert fields["label"].label == expected_instance["label"]
