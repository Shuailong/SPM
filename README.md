# Sentence Pair Models

Codebase for sentence pair task such as SNLI based on allennlp.

## Data preparation

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

Links:

- pretrained models: [glove](https://nlp.stanford.edu/projects/glove/), [elmo](https://github.com/allenai/allennlp/blob/master/tutorials/how_to/elmo.md), [bert model](https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/pytorch_pretrained_bert/modeling.py), [bert vocab](https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/pytorch_pretrained_bert/tokenization.py)
- data: [snli](https://nlp.stanford.edu/projects/snli/), [mnli](https://www.nyu.edu/projects/bowman/multinli/), [quora](https://data.quora.com/First-Quora-Dataset-Release-Question-Pairs), [quora-split](https://drive.google.com/file/d/0B0PlTAo--BnaQWlsZl9FZ3l1c28/view), [sts-2014](http://alt.qcri.org/semeval2014/task10/index.php?id=data-and-tools), [wikiqa](http://aka.ms/WikiQA), [trecqa](http://aka.ms/WikiQA)

## Installation

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

## Training and Evaluation

Make sure to `cd` to the project root dir.
Train

```bash
allennlp train experiments/snli/bert_slstm.jsonnet -s models/snli/bert-slstm --include-package spm
```

on docker

```bash
nvidia-docker run --rm -v "/home/shuailong/SPM:/mnt/SPM" shuailongliang/spm:latest train -s /mnt/SPM/models/20190417-bert-base-finetune-slstm /mnt/SPM/experiments/snli/bert_slstm.jsonnet --include-package spm
```

Evaluation _(optional, used when training is interrupted by user)_

```bash
allennlp evaluate models/snli/slstm-bert/model.tar.gz data/snli/snli_1.0_test.jsonl --include-package spm --cuda-device 0
```

## Visualization

Model and training logs as well as results are stored in `models/snli/bert-slstm`.
To visualize the log by `tensorboard`:

```bash
tensorboard --logdir models/snli
```

## Experiments

| Model                 | MRPC(F1/acc) | SNLI(acc) |
| --------------------- | :----------: | :-------: |
| BERT-Base             |  88.9/84.8   |   89.2    |
| BERT-Large            |  89.3/85.4   |   90.4    |
| BERT-Base(reproduce)  |  88.5/84.3   |   90.9    |
| BERT-Large(reproduce) |  89.3/85.2   |   91.2    |
