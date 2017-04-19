"""
Module with a hook capable of computing accumulated variable statistics such as mean or median.
"""
from collections import OrderedDict
from typing import Iterable, Mapping

import numpy as np

from .abstract_hook import AbstractHook
from .accumulating_hook import AccumulatingHook


class StatsHook(AccumulatingHook):
    """
    Accumulate the specified variables, compute the specified aggregation values and save them to the epoch data.

    -------------------------------------------------------
    Example usage in config
    -------------------------------------------------------
    # accumulate the accuracy variable (either net output or stream source); compute and store its mean value
    hooks:
      - class: StatsHook
        variables:
            accuracy: [mean]
    -------------------------------------------------------
    """

    AGGREGATIONS = {'mean', 'std', 'min', 'max', 'median'}

    def __init__(self, variables: Mapping[str, Iterable[str]], **kwargs):
        """
        Create new stats hook.

        Raises:
            ValueError: if the specified aggregation function is not supported
        """
        for variable, aggregations in variables.items():
            for aggregation in aggregations:
                if aggregation not in StatsHook.AGGREGATIONS:
                    raise ValueError('Aggregation `{}` for variable `{}` is not supported.'
                                     .format(aggregation, variable))

        super().__init__(variables=list(variables.keys()), **kwargs)

        self._variables = variables

    @staticmethod
    def _compute_aggregation(aggregation: str, data: Iterable):
        """
        Compute the specified aggregation on the given data.

        :param aggregation: on of {mean, std, min, max, median}.
        :param data: data to be aggregated

        Raises:
            Value Error if the specified aggregation is not supported
        """
        if aggregation == 'mean':
            return np.mean(data)
        elif aggregation == 'std':
            return np.std(data)
        elif aggregation == 'min':
            return np.min(data)
        elif aggregation == 'max':
            return np.max(data)
        elif aggregation == 'median':
            return np.median(data)
        raise ValueError('Aggregation `{}` is not supported.'.format(aggregation))

    def _save_stats(self, epoch_data: AbstractHook.EpochData) -> None:
        """Extend `epoch_data` by stream:variable:aggreagation data."""

        for stream_name in epoch_data.keys():
            for variable_name, variable_aggrs in self._variables.items():
                # variables are already checked in the AccumulatingHook; hence, we do not check them here
                epoch_data[stream_name][variable_name] = OrderedDict(
                    {aggr: StatsHook._compute_aggregation(aggr, self._accumulator[stream_name][variable_name])
                     for aggr in variable_aggrs})

    def after_epoch(self, epoch_data: AbstractHook.EpochData, **kwargs) -> None:
        """Compute the specified aggregations and save them to the given epoch data."""
        self._save_stats(epoch_data)
        super().after_epoch(epoch_data=epoch_data, **kwargs)
