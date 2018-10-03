"""
This module contains util functions and classes that do not fit any other utils module.
"""

import os
import sys
import logging
import signal
import threading
from ..types import TrainingTerminated

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


class CaughtInterrupts:
    """
    Catch SIGINT and SIGTERM interrupt signals allowing to stop the training with grace in between
    ``__enter__`` and ``__exit__``.

    On first signal raise :py:class:`TrainingTerminated` in :py:meth:`raise_check_interrupt`.
    On second signal call ``sys.exit`` with exit status 1.

    .. code-block:: python
        :caption: Usage

        with CaughtInterrupts() as catch:
            # interrupt signals are captured here
            # do staff
            catch.raise_check_interrupt()  # raise TrainingTerminated if at least one signal was caught

    """

    INTERRUPT_SIGNALS = [signal.SIGINT, signal.SIGTERM]
    """List of interrupt signals being handled by :py:class:`CaughtInterrupts`."""

    def __init__(self):
        """Create new CaughtInterrupts instance."""
        self._num_signals = 0
        self._origin_handlers = {}

    def _signal_handler(self, *_) -> None:
        """
        On the first signal, increase the ``self._num_signals`` counter.
        Call ``sys.exit`` on any subsequent signal.
        """
        if self._num_signals == 0:
            logging.warning('Interrupt signal caught - training will be terminated')
            logging.warning('Another interrupt signal will terminate the program immediately')
            self._num_signals += 1
        else:
            logging.error('Another interrupt signal caught - terminating program immediately')
            sys.exit(2)

    def raise_check_interrupt(self) -> None:
        """
        Stop the training if any interrupt signal was caught.

        :raise TrainingTerminated: if an interrupt signal was caught
        """
        if self._num_signals > 0:
            raise TrainingTerminated('Interrupt signal caught')

    def __enter__(self):
        """Register the interrupt signal handlers."""
        for sig in CaughtInterrupts.INTERRUPT_SIGNALS:
            self._origin_handlers[sig] = signal.getsignal(sig)
            signal.signal(sig, self._signal_handler)
        self._num_signals = 0
        return self

    def __exit__(self, *args) -> None:
        """Switch to the original signal handlers."""
        for sig in CaughtInterrupts.INTERRUPT_SIGNALS:
            signal.signal(sig, self._origin_handlers[sig])
        self._origin_handlers = {}


class ReleasedSemaphore:
    """
    Releases a :py:class:`threading.Semaphore` in between ``__enter__`` and ``__exit__``.

    .. code-block:: python
        :caption: Usage

        semaphore = Semaphore()
        semaphore.acquire()

        Thread 1:
            while True:
                # each item needs an aquired semaphore
                with semaphore:
                    process_item()

        Thread 2:
            # Thread 1 processing is disallowed
            do_something()
            # Thread 1 processing is allowed
            with ReleasedSemaphore(semaphore):
                do_something()

    """

    def __init__(self, semaphore: threading.Semaphore):
        self._semaphore = semaphore

    def __enter__(self) -> None:
        self._semaphore.release()

    def __exit__(self, *args) -> None:
        self._semaphore.acquire()

__all__ = []
