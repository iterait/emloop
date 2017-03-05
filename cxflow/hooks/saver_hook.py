from .abstract_hook import AbstractHook
from ..nets.abstract_net import AbstractNet

import logging


class SaverHook(AbstractHook):
    def __init__(self, net: AbstractNet, save_every_n_epochs: int=1, **kwargs):
        super().__init__(**kwargs)
        self.net = net
        self.save_every_n_epochs = save_every_n_epochs

    def after_epoch(self, epoch_id: int, **kwargs) -> None:
        if epoch_id % self.save_every_n_epochs == 0:
            try:
                logging.debug('Creating checkpoint')
                save_path = self.net.save_checkpoint(name=str(epoch_id))
                logging.info('Created checkpoint: %s', save_path)
            except Exception as e:
                logging.error('Checkpoint at epoch=%d not created because an unexpected exception occurred: %s',
                              epoch_id, e)


class BestSaverHook(AbstractHook):
    def __init__(self, net: AbstractNet, metric: str, condition: str, output_name: str='best', **kwargs):
        super().__init__(net=net, **kwargs)
        self.net = net
        self.metric = metric
        self.condition = condition
        self.output_name = output_name

        self.best_metric = None

    def _save_checkpoint(self):
        try:
            logging.debug('Creating checkpoint')
            save_path = self.net.save_checkpoint(name=self.output_name)
            logging.info('Created checkpoint: %s', save_path)
        except Exception as e:
            logging.error('Checkpoint not created because an unexpected exception occurred: %s', e)

    def before_first_epoch(self, valid_results: dict, **kwargs) -> None:
        self.best_metric = valid_results[self.metric]
        self._save_checkpoint()

    def after_epoch(self, valid_results: dict, **kwargs) -> None:
        if self.condition == 'min':
            if valid_results[self.metric] < self.best_metric:
                self.best_metric = valid_results[self.metric]
                self._save_checkpoint()
        elif self.condition == 'max':
            if valid_results[self.metric] > self.best_metric:
                self.best_metric = valid_results[self.metric]
                self._save_checkpoint()
        else:
            logging.error('BestSaverHook support only {min,max} as a condition')
            raise ValueError('BestSaverHook support only {min,max} as a condition')
