from .abstract_hook import AbstractHook
from ..nets.abstract_net import AbstractNet
from ..datasets.abstract_dataset import AbstractDataset

import logging


class SaverHook(AbstractHook):
    """Save the model every `n` epochs."""

    def __init__(self, net: AbstractNet, save_every_n_epochs: int=1, **kwargs):
        """
        :param net: trained net
        :param save_every_n_epochs: how often is the model saved
        """
        super().__init__(net=net, **kwargs)
        self._net = net
        self._save_every_n_epochs = save_every_n_epochs

    def after_epoch(self, epoch_id: int, **kwargs) -> None:
        if epoch_id % self._save_every_n_epochs == 0:
            try:
                logging.debug('Creating checkpoint')
                save_path = self._net.save_checkpoint(name=str(epoch_id))
                logging.info('Created checkpoint: %s', save_path)
            except Exception as e:
                logging.error('Checkpoint at epoch=%d not created because an unexpected exception occurred: %s',
                              epoch_id, e)


class BestSaverHook(AbstractHook):
    """Save the model when it outperforms itself."""

    def __init__(self, net: AbstractNet, metric: str, condition: str, output_name: str='best', **kwargs):
        """
        Example: metric=loss, condition=min -> saved the model when the loss is best so far.
        :param net: trained network
        :param metric: metric to be evaluated (usually loss)
        :param condition: {min,max}
        :param output_name: suffix of the dumped checkpoint
        """
        super().__init__(net=net, **kwargs)
        self._net = net
        self._metric = metric
        self._condition = condition
        self._output_name = output_name

        self.best_metric = None

    def _save_checkpoint(self):
        try:
            logging.debug('Creating checkpoint')
            save_path = self._net.save_checkpoint(name=self._output_name)
            logging.info('Created checkpoint: %s', save_path)
        except Exception as e:
            logging.error('Checkpoint not created because an unexpected exception occurred: %s', e)

    def before_first_epoch(self, valid_results: AbstractDataset.Batch, **kwargs) -> None:
        self.best_metric = valid_results[self._metric]
        self._save_checkpoint()

    def after_epoch(self, valid_results: AbstractDataset.Batch, **kwargs) -> None:
        if self._condition == 'min':
            if self.best_metric is None or valid_results[self._metric] < self.best_metric:
                self.best_metric = valid_results[self._metric]
                self._save_checkpoint()
        elif self._condition == 'max':
            if self.best_metric is None or valid_results[self._metric] > self.best_metric:
                self.best_metric = valid_results[self._metric]
                self._save_checkpoint()
        else:
            logging.error('BestSaverHook support only {min,max} as a condition')
            raise ValueError('BestSaverHook support only {min,max} as a condition')
