#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author: Shuailong
# @Email: liangshuailong@gmail.com
# @Date: 2019-06-23 14:50:10
# @Last Modified by: Shuailong
# @Last Modified time: 2019-06-23 14:50:12

import json
import random
import statistics as stats
import numpy as np
import argparse
from collections import Counter
from collections import defaultdict
from itertools import groupby

LABELS = ['entailment', 'contradiction', 'neutral']


def main():
    all_predictions = []
    for pred_file in args.predictions:
        predictions = []
        with open(pred_file) as pred_f:
            for line in pred_f:
                predictions.append(json.loads(line)['label_probs'])
        all_predictions.append(predictions)

    n_predictions = len(args.predictions)
    n_samples = len(all_predictions[0])

    pred_labels = []
    for i in range(n_samples):
        probs = [all_predictions[j][i] for j in range(n_predictions)]
        # n_predictions x 3
        if args.mode == 'soft':
            probs_transpose = list(map(list, zip(*probs)))  # 3 x n_predictions
            label_probs = [stats.mean(p) for p in probs_transpose]
            label_index = np.argmax(label_probs).item()
        elif args.mode == 'hard':
            label_indices = [np.argmax(p).item() for p in probs]
            try:
                label_index = stats.mode(label_indices)
            except stats.StatisticsError as exc:
                rand_index = 2
                print(
                    f'No unique mode. Select random label: {LABELS[rand_index]}.')
        label = LABELS[label_index]
        pred_labels.append(label)

    gold_labels = []
    with open(args.dataset) as data_f:
        for line in data_f:
            sample = json.loads(line)
            if sample['gold_label'] != '-':
                gold_labels.append(sample['gold_label'])
    assert len(pred_labels) == len(gold_labels),\
        f'prediction labels ({len(pred_labels)}) != gold labels ({len(gold_labels)})'

    correct = 0
    for gold_label, pred_label in zip(gold_labels, pred_labels):
        if gold_label == pred_label:
            correct += 1

    print(
        f'Accuracy: {correct} / {n_samples} = {correct/n_samples*100:.2f}%.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Ensemble predictions')
    parser.add_argument('--predictions', '-p', nargs='+',
                        help='prediction files in json format')
    parser.add_argument('--dataset', '-d', help='dataset file')
    parser.add_argument('--mode', choices=['hard', 'soft'], default='soft')
    args = parser.parse_args()
    main()
