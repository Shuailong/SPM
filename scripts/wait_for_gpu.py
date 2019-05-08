#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author: Shuailong
# @Email: liangshuailong@gmail.com
# @Date: 2019-05-07 20:51:34
# @Last Modified by: Shuailong
# @Last Modified time: 2019-05-07 20:51:42

from typing import List, Tuple
import argparse

from allennlp.common.util import gpu_memory_mb
from knockknock import telegram_sender
from time import sleep

CHAT_ID: int = 725875725
@telegram_sender(token="844871631:AAHkd3woFQ2i5_WXvlbl1pmGNCdXtJ577Gs", chat_id=CHAT_ID)
def monitor(min_memory: int, check_interval: int) -> List[Tuple[int]]:
    available_gpu = []
    while(not available_gpu):
        for gpu, memory in gpu_memory_mb().items():
            if memory < min_memory:
                available_gpu.append((gpu, memory))
        sleep(check_interval)
    return available_gpu


def main(args):
    print('Start monitering...')
    _ = monitor(args.min_memory, args.check_interval)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Monitor GPU usage. Once available, send message.')
    parser.add_argument('-m', '--min-memory', type=int, default=1000,
                        help='if the memory is less than this, consider available')
    parser.add_argument('-i', '--check-interval', type=int, default=60,
                        help='interval to query gpu usage')
    args = parser.parse_args()
    main(args)
