#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author: Shuailong
# @Email: liangshuailong@gmail.com
# @Date: 2019-05-11 00:27:12
# @Last Modified by: Shuailong
# @Last Modified time: 2019-05-11 00:27:34

'''
Remove example with "-" label.
Augument example by the following rule:

A, B entailment
+ B, A neutral ?
A, B contradiction
+ B, A contradiction
A, B neutral
+ B, A neutral ?

'''

import json
import argparse

from tqdm import tqdm


def main(args):
    with open(args.input) as f:
        lines = sum((1 for _ in f))
    print(f'Reading {lines} examples from {args.input}')
    with open(args.input) as infile, open(args.output, 'w') as outf:
        aug_counts = 0
        for line in tqdm(infile, total=lines, desc='read'):
            jsondict = json.loads(line)
            sentence1 = jsondict['sentence1']
            sentence2 = jsondict['sentence2']
            gold_label = jsondict['gold_label']
            if gold_label == '-':
                continue
            original = {
                'sentence1': sentence1,
                'sentence2': sentence2,
                'gold_label': gold_label
            }
            outf.write(json.dumps(original) + '\n')
            aug_counts += 1
            if gold_label == 'contradiction':
                mirror = {
                    'sentence1': sentence2,
                    'sentence2': sentence1,
                    'gold_label': gold_label
                }
                outf.write(json.dumps(mirror) + '\n')
                aug_counts += 1
    print(f'Written {aug_counts} examples into {args.output}.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Augument SNLI dataset')
    parser.add_argument('input', help='input file name')
    parser.add_argument('output', help='output file name')
    args = parser.parse_args()
    main(args)
