"""
Module with a hook capable of computing accumulated variable statistics such as mean or median.
"""
from collections import OrderedDict
from typing import Iterable, Mapping, Any

import numpy as np

from . import AbstractHook, AccumulateVariables


class ComputeStats(AccumulateVariables):
    """
    Accumulate the specified variables, compute the specified aggregation values and save them to the epoch data.

    .. code-block:: yaml
        :caption: compute loss and accuracy means after each epoch

        hooks:
        - ComputeStats:
            variables: [loss, accuracy]

    .. code-block:: yaml
        :caption: compute min and max loss after each epoch

        hooks:
          - ComputeStats:
              variables:
                - loss : [min, max]

    """

    AGGREGATIONS = {'mean', 'std', 'min', 'max', 'median'}
    """Supported numpy-like aggregation methods."""

    def __init__(self, variables, **kwargs):
        """
        Create new stats hook.

        :param variables: list of variables mapping: ``variable_name`` -> ``List`` [``aggregations``...] wherein
            ``aggregations`` are one of :py:attr:`AGGREGATIONS` or ``variable_name`` only to compute its mean
        :param kwargs: Ignored
        :raise ValueError: if the specified aggregation function is not supported
        """
        # list of mappings variable -> [aggregations..]
        variable_aggregations = [{variable: ['mean']} if isinstance(variable, str) else variable for variable in
                                 variables]

        # single mapping variable -> aggregations
        self._variable_aggregations = {}
        for variable_aggregation in variable_aggregations:
            self._variable_aggregations.update(variable_aggregation)

        for variable, aggregations in self._variable_aggregations.items():
            for aggregation in aggregations:
                if aggregation not in ComputeStats.AGGREGATIONS:
                    raise ValueError('Aggregation `{}` for variable `{}` is not supported.'
                                     .format(aggregation, variable))

        super().__init__(variables=list(self._variable_aggregations.keys()), **kwargs)

    @staticmethod
    def _compute_aggregation(aggregation: str, data: Iterable[Any]):
        """
        Compute the specified aggregation on the given data.

        :param aggregation: on of {mean, std, min, max, median}.
        :param data: data to be aggregated
        :raise ValueError: if the specified aggregation is not supported
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
        """
        Extend ``epoch_data`` by stream:variable:aggreagation data.

        :param epoch_data: data source from which the statistics are computed
        """

        for stream_name in epoch_data.keys():
            for variable, aggregations in self._variable_aggregations.items():
                # variables are already checked in the AccumulatingHook; hence, we do not check them here
                epoch_data[stream_name][variable] = OrderedDict(
                    {aggr: ComputeStats._compute_aggregation(aggr, self._accumulator[stream_name][variable])
                     for aggr in aggregations})

    def after_epoch(self, epoch_data: AbstractHook.EpochData, **kwargs) -> None:
        """
        Compute the specified aggregations and save them to the given epoch data.

        :param epoch_data: epoch data to be processed
        """
        self._save_stats(epoch_data)
        super().after_epoch(epoch_data=epoch_data, **kwargs)
