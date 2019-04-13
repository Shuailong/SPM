### Sentence Pair Models

Codebase for sentence pair task such as SNLI based on allennlp.

#### Data preparation
File structure
<pre>
|
|--data
|    |
|    |--elmo
|    |    |--elmo_2x4096_512_2048cnn_2xhighway_options.json
|    |    |--elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5
|    |
|    |--glove
|    |    |--glove.840B.300d.txt.gz
|    |
|    |--snli
|    |    |--snli_1.0_dev.jsonl
|    |    |--snli_1.0_train.jsonl
|    |    |--snli_1.0_test.jsonl
|    |
|    |--bert
|         |--vocab.txt
|         |--bert-base-uncased.tar.gz
|
|--models
     |
     |--snli

</pre>

Links: [elmo](https://github.com/allenai/allennlp/blob/master/tutorials/how_to/elmo.md), [glove](https://nlp.stanford.edu/projects/glove/), [snli](https://nlp.stanford.edu/projects/snli/), [bert model](https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/pytorch_pretrained_bert/modeling.py), [bert vocab](https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/pytorch_pretrained_bert/tokenization.py)

#### Installation
Follow the instructions [here](https://github.com/allenai/allennlp).
TLDR: 
``` bash
# install conda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
sh Miniconda3-latest-Linux-x86_64.sh
# create virtual environment
conda create -n allennlp python=3
conda activate allennlp
# install!
pip install allennlp
```

#### Training and Evaluation
Make sure to `cd` to the project root dir.
Train
```bash
allennlp train experiments/snli/bert_slstm.jsonnet -s models/snli/bert-slstm --include-package spm
```
Evaluation _(optional, used when training is interrupted by user)_
```bash
allennlp evaluate models/snli/slstm-bert/model.tar.gz data/snli/snli_1.0_test.jsonl --include-package spm --cuda-device 0
```

#### Visualization
Model and training logs as well as results are stored in `models/snli/bert-slstm`.
To visualize the log by `tensorboard`:
```bash
tensorboard --logdir models/snli
```
