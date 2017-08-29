"""
Module with official **cxflow** hooks.


.. tip::

    Hooks listed here may be configured without specifying their fully qualified names. E.g.:

    .. code-block:: yaml

        hooks:
          - SaveBest

"""
from .abstract_hook import AbstractHook, TrainingTerminated
from .accumulate_variables import AccumulateVariables
from .write_csv import WriteCSV
from .stop_after import StopAfter
from .log_variables import LogVariables
from .log_profile import LogProfile
from .save import SaveEvery, SaveBest
from .catch_sigint import CatchSigint
from .compute_stats import ComputeStats
from .check import Check

AbstractHook.__module__ = '.hooks'

__all__ = ['AbstractHook', 'TrainingTerminated', 'AccumulateVariables', 'WriteCSV', 'StopAfter', 'LogVariables',
           'LogProfile', 'SaveEvery', 'SaveBest', 'CatchSigint', 'ComputeStats', 'Check']
