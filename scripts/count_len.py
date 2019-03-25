#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author: Shuailong
# @Email: liangshuailong@gmail.com
# @Date:   2019-03-24 16:15:23
# @Last Modified by:  Shuailong
# @Last Modified time: 2019-03-24 16:31:34

from __future__ import division
import os
import json
import argparse

from tqdm import tqdm
import numpy as np


def read_data(datadir, data_file):
    res = []
    with open(os.path.join(datadir, data_file)) as f:
        for line in tqdm(f):
            d = json.loads(line.strip())
            sentence1 = d['sentence1']
            sentence2 = d['sentence2']
            res.append(len(sentence1.split()))
            res.append(len(sentence2.split()))
    return res


def main(args):
    print(args)
    train_lens = read_data(args.data_dir, 'snli_1.0_train.jsonl')
    dev_lens = read_data(args.data_dir, 'snli_1.0_dev.jsonl')
    test_lens = read_data(args.data_dir, 'snli_1.0_test.jsonl')

    print(
        f'train max/min/avg: {max(train_lens)}/{min(train_lens)}/{np.mean(train_lens):.2f}')
    print(
        f'dev max/min/avg: {max(dev_lens)}/{min(dev_lens)}/{np.mean(dev_lens):.2f}')
    print(
        f'test max/min/avg: {max(test_lens)}/{min(test_lens)}/{np.mean(test_lens):.2f}')
    print(
        f'overall max/min/avg: {max(train_lens + dev_lens + test_lens)}/{min(train_lens + dev_lens + test_lens)}/{np.mean(train_lens + dev_lens + test_lens):.2f}')

    c = 0
    for l in train_lens:
        if l > 60:
            c += 1
    print(f'{c} train instances are longer than 60')

def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1', 'y')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Analyze sentence length for SNLI dataset')
    parser.register('type', 'bool', str2bool)
    parser.add_argument('--data-dir', type=str, default='./data/snli')
    parser.set_defaults()
    args = parser.parse_args()
    main(args)
