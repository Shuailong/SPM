#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author: Shuailong
# @Email: liangshuailong@gmail.com
# @Date: 2019-05-12 00:55:17
# @Last Modified by: Shuailong
# @Last Modified time: 2019-05-12 00:55:22

import json
import argparse
from collections import Counter


def main(args):
    with open(args.input) as f:
        gold_label_counts = Counter()
        for line in f:
            jsondict = json.loads(line)
            gold_label = jsondict['gold_label']
            if gold_label == '-':
                continue
            anno_labels = jsondict['annotator_labels']
            gold_label_count = anno_labels.count(gold_label)
            gold_label_counts[gold_label_count] += 1

        for label, label_c in sorted(gold_label_counts.items()):
            print(
                f'{label:<10}\t{label_c:<10}\t{label_c/sum(gold_label_counts.values())*100:.2f}%')
        print(f'{"total":<10}\t{sum(gold_label_counts.values()):<10}\t{100:.2f}%')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Calculate annotator agreement')
    parser.add_argument('input', help='input file (SNLI train/dev/test)')
    args = parser.parse_args()
    main(args)
