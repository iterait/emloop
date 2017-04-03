import logging
import signal

from .abstract_hook import AbstractHook, TrainingTerminated


class SigintHook(AbstractHook):
    """
    SIGINT catcher.

    On first sigint finish the current batch and terminate training politely, i.e. trigger all `after_training` hooks.
    On second sigint quit immediately with exit code 1
    """

    def __init__(self, **kwargs):
        self._num_sigints = 0
        self._original_handler = None
        super().__init__(**kwargs)

    def _sigint_handler(self, signum, frame):
        if self._num_sigints > 0:  # not the first sigint
            logging.error('Another SIGINT caught - terminating program immediately')
            quit(1)
        else:  # first sigint
            logging.warning('SIGINT caught - terminating after the following batch.')
            self._num_sigints += 1

    def before_training(self, **kwargs):
        self._original_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, self._sigint_handler)
        self._num_sigints = 0

    def after_batch(self, **kwargs):
        if self._num_sigints > 0:
            logging.warning('Terminating because SIGINT was caught during last batch processing.')
            raise TrainingTerminated('SIGINT caught')

    def after_training(self, **kwargs):
        signal.signal(signal.SIGINT, self._original_handler)
