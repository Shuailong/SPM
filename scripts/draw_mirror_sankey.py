#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author: Shuailong
# @Email: liangshuailong@gmail.com
# @Date: 2019-05-11 23:08:35
# @Last Modified by: Shuailong
# @Last Modified time: 2019-05-11 23:08:50


import json
import argparse
import pandas as pd

import seaborn as sns
from pysankey import sankey


def main(args):
    original_labels = []
    with open(args.original) as f:
        for line in f:
            original_labels.append(json.loads(line)['label'])

    mirror_labels = []
    with open(args.mirror) as f:
        for line in f:
            mirror_labels.append(json.loads(line)['label'])

    colorDict = {
        "neutral": '#FEC925',
        "contradiction": '#FA1E44',
        "entailment": '#5AB190'
    }
    sankey(
        original_labels, mirror_labels,
        aspect=20, figureName="original_mirror",
        leftLabels=["neutral", "contradiction", "entailment"],
        rightLabels=["neutral", "contradiction", "entailment"],
        colorDict=colorDict
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Draw sankey diagram for original and mirror instance prediction labels')
    parser.add_argument('original', help='original predictions')
    parser.add_argument('mirror', help='mirror predictions')
    args = parser.parse_args()
    main(args)
