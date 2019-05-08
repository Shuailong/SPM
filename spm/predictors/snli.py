#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author: Shuailong
# @Email: liangshuailong@gmail.com
# @Date: 2019-05-07 17:35:17
# @Last Modified by: Shuailong
# @Last Modified time: 2019-05-07 17:35:31

from typing import List
import json
from overrides import overrides

from allennlp.common.util import JsonDict
from allennlp.data import DatasetReader, Instance
from allennlp.predictors.predictor import Predictor
from allennlp.models import Model


@Predictor.register('snli')
class SNLIPredictor(Predictor):
    def __init__(self, model: Model, dataset_reader: DatasetReader) -> None:
        super().__init__(model, dataset_reader)

    def predict(self, jsonline: str) -> JsonDict:
        """
        Predict function for SNLI dataset
        Parameters
        ----------
        jsonline: ``str``
            A json line that has the same format as the snli data file.

        Returns
        ----------
        A dictionary that represents the prediction made by the system.
        """
        return self.predict_json(json.loads(jsonline))

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """
        Expects json that looks like the original snli data file.
        """
        label = json_dict["gold_label"]
        premise = json_dict["sentence1"]
        hypothesis = json_dict["sentence2"]

        instance = self._dataset_reader.text_to_instance(
            premise, hypothesis, label)

        return instance
