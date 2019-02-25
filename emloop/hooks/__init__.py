"""
Module with official **emloop** hooks.


.. tip::

    Hooks listed here may be configured without specifying their fully qualified names. E.g.:

    .. code-block:: yaml

        hooks:
          - SaveBest

"""
from .abstract_hook import AbstractHook, TrainingTerminated
from .accumulate_variables import AccumulateVariables
from .benchmark import Benchmark
from .check import Check
from .classification_metrics import ClassificationMetrics
from .compute_stats import ComputeStats
from .every_n_epoch import EveryNEpoch
from .flatten import Flatten
from .log_dir import LogDir
from .log_profile import LogProfile
from .log_variables import LogVariables
from .logits_to_csv import LogitsToCsv
from .on_plateau import OnPlateau
from .plot_lines import PlotLines
from .save import SaveEvery, SaveBest, SaveLatest
from .save_cm import SaveConfusionMatrix
from .save_file import SaveFile
from .sequence_to_csv import SequenceToCsv
from .show_progress import ShowProgress
from .stop_after import StopAfter
from .stop_on_nan import StopOnNaN
from .stop_on_plateau import StopOnPlateau
from .training_trace import TrainingTrace
from .write_csv import WriteCSV

AbstractHook.__module__ = '.hooks'

__all__ = ['AbstractHook', 'TrainingTerminated', 'AccumulateVariables', 'WriteCSV', 'StopAfter', 'LogVariables',
           'LogProfile', 'LogDir', 'SaveEvery', 'SaveBest', 'SaveLatest', 'ComputeStats', 'Check', 'ShowProgress',
           'EveryNEpoch', 'OnPlateau', 'StopOnPlateau', 'StopOnNaN', 'SaveConfusionMatrix', 'Flatten', 'PlotLines',
           'LogitsToCsv', 'SequenceToCsv', 'SaveFile', 'Benchmark', 'ClassificationMetrics', 'TrainingTrace']

