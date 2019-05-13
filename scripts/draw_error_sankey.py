#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author: Shuailong
# @Email: liangshuailong@gmail.com
# @Date: 2019-05-12 23:55:58
# @Last Modified by: Shuailong
# @Last Modified time: 2019-05-12 23:56:01

import json
import argparse

from pysankey import sankey


def main(args):
    gold_labels, pred_labels = [], []
    with open(args.prediction) as f:
        for line in f:
            sample = json.loads(line)
            gold_labels.append(sample['gold_label'])
            pred_labels.append(sample['pred_label'])

    colorDict = {
        "neutral": '#FEC925',
        "contradiction": '#FA1E44',
        "entailment": '#5AB190'
    }
    sankey(
        gold_labels, pred_labels,
        aspect=20, figureName="gold_pred",
        leftLabels=["neutral", "contradiction", "entailment"],
        rightLabels=["neutral", "contradiction", "entailment"],
        colorDict=colorDict
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Draw sankey diagram for error instances gold to prediction')
    parser.add_argument('prediction', help='prediction file')
    args = parser.parse_args()
    main(args)
