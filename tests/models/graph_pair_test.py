#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author: Shuailong
# @Email: liangshuailong@gmail.com
# @Date:   2019-02-28 15:48:23
# @Last Modified by:  Shuailong
# @Last Modified time: 2019-02-28 17:00:23

# pylint: disable=invalid-name,protected-access
from allennlp.common.testing import ModelTestCase


class GraphPairTest(ModelTestCase):
    def setUp(self):
        super(GraphPairTest, self).setUp()
        self.set_up_model('tests/fixtures/graph_pair.json',
                          'tests/fixtures/snli_1.0_sample.jsonl')

    def test_model_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)