#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author: Shuailong
# @Email: liangshuailong@gmail.com
# @Date:   2019-02-27 22:08:29
# @Last Modified by:  Shuailong
# @Last Modified time: 2019-03-24 17:09:22

# pylint: disable=wildcard-import
import os
from pathlib import PosixPath

DATA_DIR = (
    os.getenv('SPM_DATA') or
    os.path.join(PosixPath(__file__).absolute().parents[1].as_posix(), 'data')
)
