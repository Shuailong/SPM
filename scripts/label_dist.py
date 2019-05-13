#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author: Shuailong
# @Email: liangshuailong@gmail.com
# @Date: 2019-05-09 19:50:30
# @Last Modified by: Shuailong
# @Last Modified time: 2019-05-09 19:50:35

from collections import Counter
import json
import argparse


def main(args):
    with open(args.input) as f:
        label_counter = Counter()
        for line in f:
            jsondict = json.loads(line)
            label = jsondict['gold_label']
            label_counter[label] += 1
    for label, label_c in label_counter.items():
        print(
            f'{label:<10}\t{label_c:<10}\t{label_c/sum(label_counter.values())*100:.2f}%')
    print(
        f'{"total":<10}\t{sum(label_counter.values()):<10}\t{100:.2f}%')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Label distribution of SNLI corpus')
    parser.add_argument('input', help='input file name')
    args = parser.parse_args()
    main(args)
