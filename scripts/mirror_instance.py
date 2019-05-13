#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author: Shuailong
# @Email: liangshuailong@gmail.com
# @Date: 2019-05-11 00:27:12
# @Last Modified by: Shuailong
# @Last Modified time: 2019-05-11 00:27:34

'''
Swap s1 and s2.
'''

import json
import argparse

from tqdm import tqdm


def main(args):
    with open(args.input) as f:
        lines = sum((1 for _ in f))
    print(f'Reading {lines} examples from {args.input}')
    with open(args.input) as infile, open(args.output, 'w') as outf:
        count = 0
        for line in tqdm(infile, total=lines, desc='read'):
            jsondict = json.loads(line)
            if jsondict['gold_label'] == '-':
                continue
            mirror = {
                'sentence1': jsondict['sentence2'],
                'sentence2': jsondict['sentence1'],
                'gold_label': 'neutral'  # default, to avoid UNK label error
            }
            outf.write(json.dumps(mirror) + '\n')
            count += 1
    print(f'Written {count} mirror examples into {args.output}.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='SWAP s1 and s2 for SNLI dataset')
    parser.add_argument('input', help='input file name')
    parser.add_argument('output', help='output file name')
    args = parser.parse_args()
    main(args)
