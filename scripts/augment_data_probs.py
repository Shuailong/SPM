#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author: Shuailong
# @Email: liangshuailong@gmail.com
# @Date: 2019-05-12 15:18:06
# @Last Modified by: Shuailong
# @Last Modified time: 2019-05-12 15:18:13

import json
import argparse

from tqdm import tqdm


def main(args):
    print('Reading original dataset...')
    original_data = []
    with open(args.original) as f:
        for line in tqdm(f):
            sample = json.loads(line)
            original_data.append({
                'sentence1': sample['sentence1'],
                'sentence2': sample['sentence2'],
                'gold_label': sample['gold_label']
            })
    print(f'Read {len(original_data)} original instances.')
    print('-'*100)
    print('Reading mirror instance and filtering...')
    aug_data = []
    count = 0
    with open(args.mirror) as mf:
        total = sum((1 for _ in mf))
    with open(args.mirror) as mf, open(args.prediction) as pf:
        for instance, prediction in tqdm(zip(mf, pf), total=total):
            ins = json.loads(instance)
            pred = json.loads(prediction)
            if max(pred['label_probs']) < args.confidence_threshold:
                continue
            aug_data.append({
                'sentence1': ins['sentence1'],
                'sentence2': ins['sentence2'],
                'gold_label': pred['label']
            })
            count += 1
    print(f'From {total} mirror instances get {count} valid ones.')
    print('-'*100)
    print('Combine and writing...')
    count = 0
    with open(args.output, 'w') as of:
        for sample in original_data + aug_data:
            of.write(json.dumps(sample) + '\n')
            count += 1
    print(f'From {len(original_data)} origial samples and {len(aug_data)} mirror samples created {count} samples.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Augment data by using the high prediction label confidence mirror instance')
    parser.add_argument('--original', '-i', help='original dataset')
    parser.add_argument(
        '--mirror', '-m', help='mirror dataset with dummy labels')
    parser.add_argument('--prediction', '-p',
                        help='mirror dataset predictions')
    parser.add_argument('--output', '-o', help='augmented dataset')
    parser.add_argument('--confidence_threshold', '-t', type=float, default=0.9,
                        help='use instances with confidence >= threshold.')
    args = parser.parse_args()
    main(args)
