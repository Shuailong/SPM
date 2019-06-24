#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author: Shuailong
# @Email: liangshuailong@gmail.com
# @Date: 2019-06-23 12:08:38
# @Last Modified by: Shuailong
# @Last Modified time: 2019-06-23 12:08:46

from typing import List
import json
from overrides import overrides

from allennlp.common.util import JsonDict
from allennlp.data import DatasetReader, Instance
from allennlp.predictors.predictor import Predictor
from allennlp.models import Model


@Predictor.register('snli-weight')
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

        annotator_labels = json_dict["annotator_labels"]
        if len(annotator_labels) == 1:
            label_confidence = 1
        else:
            label_confidence = annotator_labels.count(label)\
                / len(annotator_labels)

        instance = self._dataset_reader.text_to_instance(
            premise, hypothesis, label, label_confidence)

        return instance

    @overrides
    def _batch_json_to_instances(self, json_dicts: List[JsonDict]) -> List[Instance]:
        """
        Converts a list of JSON objects into a list of :class:`~allennlp.data.instance.Instance`s.
        By default, this expects that a "batch" consists of a list of JSON blobs which would
        individually be predicted by :func:`predict_json`. In order to use this method for
        batch prediction, :func:`_json_to_instance` should be implemented by the subclass, or
        if the instances have some dependency on each other, this method should be overridden
        directly.
        Remove samples with gold_label "-" from evaluation.
        """
        instances = []
        for json_dict in json_dicts:
            if json_dict['gold_label'] != '-':
                instances.append(self._json_to_instance(json_dict))
        return instances
