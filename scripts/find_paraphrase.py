#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author: Shuailong
# @Email: liangshuailong@gmail.com
# @Date: 2019-05-13 14:37:40
# @Last Modified by: Shuailong
# @Last Modified time: 2019-05-13 14:37:45

import json
import argparse
import random

from tqdm import tqdm

from allennlp.data.tokenizers.word_tokenizer import WordTokenizer


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
    print('Finding paraphrase samples...')
    assert len(original_data) == len(mirror_data),\
        'original dataset size != mirror dataset size'
    positive_samples, negative_samples = [], []

    for original, mirror in tqdm(zip(original_data, mirror_data), total=len(original_data)):
        assert original['sentence1'] == mirror['sentence2']
        assert original['sentence2'] == mirror['sentence1']
        if original['gold_label'] == 'entailment' and mirror['gold_label'] == 'entailment'\
                and mirror['confidence'] >= args.confidence_threshold:
            positive_samples.append({
                'sentence1': original['sentence1'],
                'sentence2': original['sentence2'],
                'label': 1
            })
        else:
            negative_samples.append({
                'sentence1': original['sentence1'],
                'sentence2': original['sentence2'],
                'label': 0
            })

    print('-'*100)
    print('Tokenize and write into output')
    negative_samples = random.sample(negative_samples, len(positive_samples))
    samples = positive_samples + negative_samples
    random.shuffle(samples)

    tokenizer = WordTokenizer()
    with open(args.output, 'w') as outf:
        # MRPC format
        outf.write(f'Quality\t#1 ID\t#2 ID\t#1 String\t#2 String\n')

        for sample in tqdm(samples, total=len(samples)):
            label = sample['label']
            sentence1, sentence2 = sample['sentence1'], sample['sentence2']
            s1_tokens = ' '.join(
                (t.text for t in tokenizer.tokenize(sentence1)))
            s2_tokens = ' '.join(
                (t.text for t in tokenizer.tokenize(sentence2)))
            outf.write(
                f'{label}\tsentence1\tsentence2\t{s1_tokens}\t{s2_tokens}\n')

    print(
        f'Written {len(samples)} pairs of paraphrase into {args.output}')


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
