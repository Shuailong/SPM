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

# TODO: add seaborn fig support.


def main(args):
    data = []
    for input_f in args.input:
        with open(input_f) as f:
            probs = []
            for line in f:
                jsondict = json.loads(line)
                if 'label_prob' in jsondict:
                    prob = jsondict['label_prob']
                elif 'label_probs' in jsondict:
                    prob = max(jsondict['label_probs'])
                probs.append(prob)
            data.append(probs)
    if args.media == 'terminal':
        probs = data[0]  # terminal mode supports only 1 distribution
        import plotille
        if args.hist_type == 'histogram':
            print(plotille.histogram(probs, lc='cyan'))
        else:
            print(plotille.hist(probs, lc='cyan'))
    else:
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set()
        colors = sns.color_palette("muted", len(data))
        for i, probs in enumerate(data):
            ax = sns.distplot(probs, bins=40, kde=False,
                              color=colors[i], label=args.input[i])
        if args.title:
            ax.set_title(args.title)
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Probability distribution of SNLI model predictions')
    parser.add_argument('input', nargs='+', help='input file name')
    parser.add_argument('--media', choices=['terminal', 'png'], default='terminal',
                        help='use plotille to display on terminal or save as png via seaborn')
    parser.add_argument('--title', help='title of the figure')
    parser.add_argument('--hist-type', '-t', choices=['hist', 'histogram'], default='histogram',
                        help='choose different histogram styles from plotille')
    args = parser.parse_args()
    main(args)
