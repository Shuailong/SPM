#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author: Shuailong
# @Email: liangshuailong@gmail.com
# @Date: 2019-05-13 14:37:40
# @Last Modified by: Shuailong
# @Last Modified time: 2019-05-13 14:37:45

import json
import argparse

from tqdm import tqdm


def main(args):
    print('Reading original dataset...')
    original_data = []
    with open(args.original) as f:
        total = sum((1 for _ in f))
    with open(args.original) as f:
        for line in tqdm(f, total=total):
            sample = json.loads(line)
            if sample['gold_label'] != '-':
                original_data.append({
                    'sentence1': sample['sentence1'],
                    'sentence2': sample['sentence2'],
                    'gold_label': sample['gold_label']
                })

    print(f'Read {len(original_data)} original instances.')
    print('-'*100)
    print('Reading mirror instance...')
    mirror_data = []
    count = 0
    with open(args.mirror) as mf:
        total = sum((1 for _ in mf))
    with open(args.mirror) as mf, open(args.prediction) as pf:
        for instance, prediction in tqdm(zip(mf, pf), total=total):
            ins = json.loads(instance)
            pred = json.loads(prediction)
            mirror_data.append({
                'sentence1': ins['sentence1'],
                'sentence2': ins['sentence2'],
                'gold_label': pred['label'],
                'confidence': max(pred['label_probs'])
            })
            count += 1
    print(f'From {total} mirror instances.')
    print('-'*100)
    print('Combine and writing...')
    count = 0
    assert len(original_data) == len(mirror_data),\
        'original dataset size != mirror dataset size'
    with open(args.output, 'w') as outf:
        for original, mirror in tqdm(zip(original_data, mirror_data), total=len(original_data)):
            assert original['sentence1'] == mirror['sentence2']
            assert original['sentence2'] == mirror['sentence1']
            if original['gold_label'] == 'entailment' and mirror['gold_label'] == 'entailment'\
                    and mirror['confidence'] >= args.confidence_threshold:
                pair = {
                    'sentence1': original['sentence1'],
                    'sentence2': original['sentence2']
                }
                outf.write(json.dumps(pair) + '\n')
                count += 1

    print(
        f'Written {count} pairs of paraphrase into {args.output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Based on mirror instance labels, extract paraphrase sentences')
    parser.add_argument('--original', '-i', help='original dataset')
    parser.add_argument(
        '--mirror', '-m', help='mirror dataset with dummy labels')
    parser.add_argument('--prediction', '-p',
                        help='mirror dataset predictions')
    parser.add_argument('--output', '-o', help='paraphrase dataset')
    parser.add_argument('--confidence_threshold', '-t', type=float, default=0.9,
                        help='use instances with confidence >= threshold.')
    args = parser.parse_args()
    main(args)
