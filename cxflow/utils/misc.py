"""
This module contains util functions and classes that do not fit any other utils module.
"""

import os
import sys
import logging


class DisabledPrint:
    """
    Disable printing to stdout by redirecting it to /dev/null in between __enter__ and __exit__.

    -------------------------------------------------------
    Usage
    -------------------------------------------------------
    with DisabledPrint():
        # any print commands here will be redirected to /dev/null
        pass
    -------------------------------------------------------
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
    Entirely disable the specified logger in between __enter__ and __exit__.

    -------------------------------------------------------
    Usage
    -------------------------------------------------------
    with DisabledLogger('my_logger_name'):
        # any logging actions with the my_logger_name will be ignored
        pass
    -------------------------------------------------------
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
