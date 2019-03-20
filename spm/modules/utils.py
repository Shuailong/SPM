#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author: Shuailong
# @Email: liangshuailong@gmail.com
# @Date:   2019-03-17 12:35:09
# @Last Modified by:  Shuailong
# @Last Modified time: 2019-03-18 21:31:10

import itertools
from typing import List

import torch
from allennlp.common.checks import ConfigurationError
from allennlp.nn.util import replace_masked_values


def max_with_mask(hiddens, mask):
    """
    Parameters
    ----------
    hiddens : ``torch.FloatTensor``
        A tensor of shape (batch_size, seq_len, hidden_size)
    mask : ``torch.FloatTensor``
        A tensor of shape (batch_size, seq_len)
    Returns
    -------
    m: ``torch.FloatTensor``
        A tensor of shape (batch_size, hidden_size)
    """
    m, _ = replace_masked_values(
        hiddens, mask.unsqueeze(-1), -1e7
    ).max(dim=1)
    return m


def mean_with_mask(hiddens, mask):
    """
    Parameters
    ----------
    hiddens : ``torch.FloatTensor``
        A tensor of shape (batch_size, seq_len, hidden_size)
    mask : ``torch.FloatTensor``
        A tensor of shape (batch_size, seq_len)
    Returns
    -------
    m: ``torch.FloatTensor``
        A tensor of shape (batch_size, hidden_size)
    """

    m = torch.sum(hiddens * mask.unsqueeze(-1), dim=1) / \
        torch.sum(mask, 1, keepdim=True)
    return m


def block_orthogonal(tensor: torch.Tensor,
                     split_sizes: List[int],
                     gain: float = 1.0) -> None:
    """
    An initializer which allows initializing model parameters in "blocks". This is helpful
    in the case of recurrent models which use multiple gates applied to linear projections,
    which can be computed efficiently if they are concatenated together. However, they are
    separate parameters which should be initialized independently.

    Parameters
    ----------
    tensor : ``torch.Tensor``, required.
        A tensor to initialize.
    split_sizes : List[int], required.
        A list of length ``tensor.ndim()`` specifying the size of the
        blocks along that particular dimension. E.g. ``[10, 20]`` would
        result in the tensor being split into chunks of size 10 along the
        first dimension and 20 along the second.
    gain : float, optional (default = 1.0)
        The gain (scaling) applied to the orthogonal initialization.
    """
    data = tensor.data
    sizes = list(tensor.size())
    if any([a % b != 0 for a, b in zip(sizes, split_sizes)]):
        raise ConfigurationError("tensor dimensions must be divisible by their respective "
                                 "split_sizes. Found size: {} and split_sizes: {}".format(sizes, split_sizes))
    indexes = [list(range(0, max_size, split))
               for max_size, split in zip(sizes, split_sizes)]
    # Iterate over all possible blocks within the tensor.
    for block_start_indices in itertools.product(*indexes):
        # A list of tuples containing the index to start at for this block
        # and the appropriate step size (i.e split_size[i] for dimension i).
        index_and_step_tuples = zip(block_start_indices, split_sizes)
        # This is a tuple of slices corresponding to:
        # tensor[index: index + step_size, ...]. This is
        # required because we could have an arbitrary number
        # of dimensions. The actual slices we need are the
        # start_index: start_index + step for each dimension in the tensor.
        block_slice = tuple([slice(start_index, start_index + step)
                             for start_index, step in index_and_step_tuples])
        data[block_slice] = torch.nn.init.orthogonal_(
            tensor[block_slice].contiguous(), gain=gain)


def block_normal(tensor: torch.Tensor,
                 split_sizes: List[int],
                 mean: float = 0.0,
                 std: float = 1.0) -> None:
    """
    An initializer which allows initializing model parameters in "blocks". This is helpful
    in the case of recurrent models which use multiple gates applied to linear projections,
    which can be computed efficiently if they are concatenated together. However, they are
    separate parameters which should be initialized independently.

    Parameters
    ----------
    tensor : ``torch.Tensor``, required.
        A tensor to initialize.
    split_sizes : List[int], required.
        A list of length ``tensor.ndim()`` specifying the size of the
        blocks along that particular dimension. E.g. ``[10, 20]`` would
        result in the tensor being split into chunks of size 10 along the
        first dimension and 20 along the second.
    mean : float, optional (default = 0.0)
        The mean applied to the normal initialization.
    std : float, optional (default = 0.1)
    """
    data = tensor.data
    sizes = list(tensor.size())
    if any([a % b != 0 for a, b in zip(sizes, split_sizes)]):
        raise ConfigurationError("tensor dimensions must be divisible by their respective "
                                 "split_sizes. Found size: {} and split_sizes: {}".format(sizes, split_sizes))
    indexes = [list(range(0, max_size, split))
               for max_size, split in zip(sizes, split_sizes)]
    # Iterate over all possible blocks within the tensor.
    for block_start_indices in itertools.product(*indexes):
        # A list of tuples containing the index to start at for this block
        # and the appropriate step size (i.e split_size[i] for dimension i).
        index_and_step_tuples = zip(block_start_indices, split_sizes)
        # This is a tuple of slices corresponding to:
        # tensor[index: index + step_size, ...]. This is
        # required because we could have an arbitrary number
        # of dimensions. The actual slices we need are the
        # start_index: start_index + step for each dimension in the tensor.
        block_slice = tuple([slice(start_index, start_index + step)
                             for start_index, step in index_and_step_tuples])
        data[block_slice] = torch.nn.init.normal_(
            tensor[block_slice].contiguous(), mean=mean, std=std)
