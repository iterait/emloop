from .abstract_hook import AbstractHook
from ..nets.abstract_net import AbstractNet

import logging


class SaverHook(AbstractHook):
    def __init__(self, save_every_n_epochs: int=1, **kwargs):
        super().__init__(**kwargs)
        self.save_every_n_epochs = save_every_n_epochs

    def after_epoch(self, net: AbstractNet, epoch_id: int, **kwargs) -> None:
        if epoch_id % self.save_every_n_epochs == 0:
            try:
                logging.debug('Creating checkpoint')
                save_path = net.save_checkpoint(epoch_id=epoch_id)
                logging.info('Created checkpoint: %s', save_path)
            except Exception as e:
                logging.error('Checkpoint at epoch=%d not created because an unexpected exception occurred: %s',
                              epoch_id, e)
