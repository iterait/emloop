"""
This module contains util functions and classes that do not fit any other utils module.
"""

import os
import sys
import logging
import signal
from ..hooks.abstract_hook import TrainingTerminated


class DisabledPrint:
    """
    Disable printing to stdout by redirecting it to ``/dev/null`` in between ``__enter__`` and ``__exit__``.

    .. code-block:: python
        :caption: Usage

        with DisabledPrint():
            # any print commands here will be redirected to /dev/null
            pass

    """

    def __init__(self):
        self._orig = None
        self._devnull = None

    def __enter__(self):
        """Redirect stdout to /dev/null."""
        self._orig = sys.stdout
        self._devnull = open(os.devnull, 'w')
        sys.stdout = self._devnull

    def __exit__(self, *args):
        """Redirect stdout back to the original stdout."""
        sys.stdout = self._orig
        self._devnull.close()


class DisabledLogger:
    """
    Entirely disable the specified logger in between ``__enter__`` and ``__exit__``.

    .. code-block:: python
        :caption: Usage

        with DisabledLogger('my_logger_name'):
            # any logging actions with the my_logger_name will be ignored
            pass

    """

    def __init__(self, name=None):
        self._name = name
        self._orig = None

    def __enter__(self):
        """Entirely disable logging."""
        logger = logging.getLogger(self._name)
        self._orig = logger.disabled
        logger.disabled = True

    def __exit__(self, *args):
        """Restore logging ."""
        logging.getLogger(self._name).disabled = self._orig


class CatchSigint:
    """
    Catch SIGINT signals allowing to stop the training with grace in between ``__enter__`` and ``__exit__``.

    On first sigint raise :py:class:`TrainingTerminated` in :py:meth:`raise_check_sigint`.
    On second sigint quit immediately with exit code 1.

    .. code-block:: python
        :caption: Usage

        with CatchSigint() as catch:
            # sigint signals are captured here
            # do staff
            catch.raise_check_sigint()  # raise TrainingTerminated if at least one sigint was caught

    """

    def __init__(self):
        """Create new CatchSigint."""
        self._num_sigints = 0
        self._original_handler = None

    def _sigint_handler(self, signum, _) -> None:
        """
        On the first signal, increase the ``self._num_sigints`` counter.
        Quit on any subsequent signal.

        :param signum: SIGINT signal number
        """
        if self._num_sigints > 0:  # not the first sigint
            logging.error('Another interrupt signal caught - terminating program immediately')
            quit(1)
        else:  # first sigint
            logging.warning('Interrupt signal caught - training will be terminated')
            logging.warning('Another interrupt signal will terminate the program immediately')
            self._num_sigints += 1

    def raise_check_sigint(self) -> None:
        """
        Stop the training if SIGINT signal was caught.

        :raise TrainingTerminated: if a SIGINT signal was caught
        """
        if self._num_sigints > 0:
            raise TrainingTerminated('Interrupt signal caught')

    def __enter__(self):
        """Register the SIGINT signal handler."""
        self._original_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, self._sigint_handler)
        self._num_sigints = 0
        return self

    def __exit__(self, *args) -> None:
        """Switch to the original signal handler."""
        signal.signal(signal.SIGINT, self._original_handler)


__all__ = []
