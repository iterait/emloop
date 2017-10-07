"""
Module with time profiling utils.

So far, it contains Timer object allowing to easily measure code execution time.
"""
import timeit

from ..types import TimeProfile


class Timer:
    """
    Simple helper which is able to measure execution time of python code.


    .. code-block:: python
        :caption: Usage

        profile = {}
        with Timer('my_work', profile):
            # my commands here
            pass

    """

    def __init__(self, name: str, profile: TimeProfile):
        """
        Create new Timer instance.

        :param name: event name under which the measured time should be saved
        :param profile: dict of lists of timings
        """
        self._name = name
        self._profile = profile
        self._start = None

    def __enter__(self):
        """Start measuring time."""
        self._start = timeit.default_timer()

    def __exit__(self, *args):
        """
        Stop measuring time and append the time span from ``_start`` to ``end`` to the ``_profile`` under the
        ``_name`` entry.
        """
        if self._start is None:
            raise ValueError('Timer was ended but not started.')
        end = timeit.default_timer()
        if self._name not in self._profile:
            self._profile[self._name] = []
        self._profile[self._name].append(end - self._start)
        self._start = None


__all__ = []
