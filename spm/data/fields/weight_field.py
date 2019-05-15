#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author: Shuailong
# @Email: liangshuailong@gmail.com
# @Date: 2019-05-14 14:52:20
# @Last Modified by: Shuailong
# @Last Modified time: 2019-05-14 14:52:28

from typing import Dict, Union, Set
import logging

from overrides import overrides
import torch

from allennlp.data.fields.field import Field
from allennlp.data.vocabulary import Vocabulary
from allennlp.common.checks import ConfigurationError

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class WeightField(Field[torch.Tensor]):
    """
    A ``WeightField`` is a weight containing confidence of a label.

    This field will get converted into a float number.

    Parameters
    ----------
    weight : ``float``
    """

    def __init__(self,
                 weight: float) -> None:
        self.weight = weight

    @overrides
    def as_tensor(self, padding_lengths: Dict[str, int]) -> torch.Tensor:
        # pylint: disable=unused-argument,not-callable
        tensor = torch.tensor(self.weight, dtype=torch.float)
        return tensor

    @overrides
    def get_padding_lengths(self) -> Dict[str, int]:  # pylint: disable=no-self-use
        return {}

    @overrides
    def empty_field(self):
        return WeightField(0)

    def __str__(self) -> str:
        return f"WeightField with weight: {self.weight}."
