"""
Module with simple hook logging the output dir at various occasions just for convenience.
"""
import logging


from . import AbstractHook


class LogDir(AbstractHook):
    """
    Log the output dir before training, after each epoch and after training.

    .. code-block:: yaml
        :caption: log the training dir

        hooks:
          - LogDir

    """

    def __init__(self, output_dir: str, **kwargs):
        """
        Create new LogDir hook.

        :param output_dir: training output directory
        """

        self._output_dir = output_dir
        super().__init__(**kwargs)

    def before_training(self) -> None:
        """ Log the output directory."""
        logging.info('Output dir: %s', self._output_dir)

    def after_epoch(self, **_) -> None:
        """Log the output directory."""
        logging.info('Output dir: %s', self._output_dir)

    def after_training(self) -> None:
        """Log the output directory."""
        logging.info('Output dir: %s', self._output_dir)
