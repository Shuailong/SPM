#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author: Shuailong
# @Email: liangshuailong@gmail.com
# @Date: 2019-05-08 16:52:48
# @Last Modified by: Shuailong
# @Last Modified time: 2019-05-08 16:52:53
import json
import argparse
from sklearn.metrics import confusion_matrix


def main(args):
    with open(args.gold_file) as gf, open(args.predictions_file) as pf,\
            open(args.output_file, 'w') as of:
        incorrect, total = 0, 0
        gold_instances = []
        for line in gf:
            jsondict = json.loads(line)
            if jsondict['gold_label'] != '-':
                gold_instances.append(jsondict)
        pred_instances = []
        for line in pf:
            pred_instances.append(json.loads(line))
        assert len(gold_instances) == len(pred_instances),\
            f'gold instances != prediction instances: {len(gold_instances)}!={len(pred_instances)}'

        predict_labels = []
        gold_labels = []

        for gold_ins, pred_ins in zip(gold_instances, pred_instances):
            sentence1, sentence2 = gold_ins['sentence1'], gold_ins['sentence2']
            gold_label = gold_ins['gold_label']
            gold_labels.append(gold_label)
            anno_labels = gold_ins['annotator_labels']
            if gold_label == '-':
                continue
            predict_label = pred_ins['label']
            predict_labels.append(predict_label)
            predict_prob = max(pred_ins['label_probs'])
            total += 1

            filter_condition = gold_label != predict_label
            if args.reverse:
                filter_condition = not filter_condition
            if args.lenient:
                filter_condition = filter_condition and predict_label not in anno_labels
            if filter_condition:
                incorrect += 1
                if args.format == 'txt':
                    of.write(
                        f'Premise: {sentence1}\nHypothesis: {sentence2}\n')
                    of.write(
                        f'Label: {gold_label}\nPrediction: {predict_label} (prob={predict_prob:.3f})\n')
                    anno_labels = ", ".join(anno_labels)
                    of.write(f'Annotator labels: {anno_labels}\n')
                    of.write('-'*100+'\n')
                else:
                    of.write(json.dumps({
                        'premise': sentence1,
                        'hypothesis': sentence2,
                        'gold_label': gold_label,
                        'pred_label': predict_label,
                        'label_prob': predict_prob
                    }) + '\n')
        print(f'Error instances written to {args.output_file}.')
        acc = 1-incorrect/total
        if args.reverse:
            acc = 1 - acc
        print(f'Accuracy = 1 - {incorrect}/{total} = {acc*100:.8f}%.')

        labels = ['entailment', 'contradiction', 'neutral']
        cm = confusion_matrix(gold_labels, predict_labels,
                              labels=labels)
        print(cm)
        if args.plot_cm:
            import matplotlib.pyplot as plt
            import seaborn as sns
            import pandas as pd
            sns.set()
            cm = pd.DataFrame(cm, index=labels, columns=labels)
            sns.heatmap(cm, annot=True, fmt='d', cmap=sns.light_palette(
                (210, 90, 60), input="husl"), cbar=False)
            plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Find error cases')
    parser.add_argument('--gold-file', '-g', type=str, help='gold file')
    parser.add_argument('--predictions-file', '-p',
                        type=str, help='predictions')
    parser.add_argument('--output-file', '-o', type=str, help='output file')
    parser.add_argument('--format', type=str, default='txt',
                        choices=['txt', 'json'])
    parser.add_argument('--reverse', action='store_true',
                        help='output corrent instances')
    parser.add_argument('--lenient', action='store_true',
                        help='if there is any annotator agreeing with model, consider true')
    parser.add_argument('--plot-cm', action='store_true',
                        help='plot confusion matrix')
    args = parser.parse_args()
    main(args)
