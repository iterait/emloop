"""
Module with a hook capable of computing accumulated variable statistics such as mean or median.
"""
from collections import OrderedDict
from typing import Iterable, Any

import numpy as np

from . import AccumulateVariables
from ..types import EpochData


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

    EXTRA_AGGREGATIONS = {'nanfraction', 'nancount'}
    """Extra aggregation methods extending the set of all NumPy functions."""

    def __init__(self, variables, **kwargs):
        """
        Create new stats hook.

        :param variables: list of variables mapping: ``variable_name`` -> ``List`` [``aggregations``...] wherein
            ``aggregations`` are the names of arbitrary NumPy functions returning a scalar (e.g., ``'mean'``,
            ``'nanmean'``, ``'max'``, etc.) or one of :py:attr:`EXTRA_AGGREGATIONS`. Passing just the
            ``variable name`` instead of a mapping is the same as passing {variable_name: ['mean']}.
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
                ComputeStats._raise_check_aggregation(aggregation)

        super().__init__(variables=list(self._variable_aggregations.keys()), **kwargs)

    @staticmethod
    def _raise_check_aggregation(aggregation: str):
        """
        Check whether the given aggregation is present in NumPy or it is one of EXTRA_AGGREGATIONS.

        :param aggregation: the aggregation name
        :raise ValueError: if the specified aggregation is not supported or found in NumPy
        """
        if aggregation not in ComputeStats.EXTRA_AGGREGATIONS and not hasattr(np, aggregation):
            raise ValueError('Aggregation `{}` is not a NumPy function or a member '
                             'of EXTRA_AGGREGATIONS.'.format(aggregation))

    @staticmethod
    def _compute_aggregation(aggregation: str, data: Iterable[Any]):
        """
        Compute the specified aggregation on the given data.

        :param aggregation: the name of an arbitrary NumPy function (e.g., mean, max, median, nanmean, ...)
                            or one of :py:attr:`EXTRA_AGGREGATIONS`.
        :param data: data to be aggregated
        :raise ValueError: if the specified aggregation is not supported or found in NumPy
        """
        ComputeStats._raise_check_aggregation(aggregation)
        if aggregation == 'nanfraction':
            return np.sum(np.isnan(data)) / len(data)
        if aggregation == 'nancount':
            return int(np.sum(np.isnan(data)))
        return getattr(np, aggregation)(data)

    def _save_stats(self, epoch_data: EpochData) -> None:
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

    def after_epoch(self, epoch_data: EpochData, **kwargs) -> None:
        """
        Compute the specified aggregations and save them to the given epoch data.

        :param epoch_data: epoch data to be processed
        """
        self._save_stats(epoch_data)
        super().after_epoch(epoch_data=epoch_data, **kwargs)
