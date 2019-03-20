import logging
import os
import os.path as path
from typing import Iterable, Optional, Sequence, Hashable
from collections import Counter

import numpy as np

import emloop.api   # avoids circular dependency

from .abstract_model import AbstractModel
from ..utils import load_config
from ..datasets import AbstractDataset, StreamWrapper
from ..types import Batch
from ..constants import EL_CONFIG_FILE


def major_vote(all_votes: Iterable[Iterable[Hashable]]) -> Iterable[Hashable]:
    """
    For the given iterable of object iterations, return an iterable of the most common object at each position of the
    inner iterations.

    E.g.: for [[1, 2], [1, 3], [2, 3]] the return value would be [1, 3] as 1 and 3 are the most common objects
    at the first and second positions respectively.

    :param all_votes: an iterable of object iterations
    :return: the most common objects in the iterations (the major vote)
    """
    return [Counter(votes).most_common()[0][0] for votes in zip(*all_votes)]


class Ensemble(AbstractModel):
    """
    Ensemble model facilitates assembling multiple models into one for more accurate predictions.

    .. warning::
        Ensemble model can be used for inference only (i.e. no training is supported).

    The typical usage is to train multiple (possibly different) models and assemble them with this class.

    .. code-block:: yaml
        :caption: usage from config

        model:
          name: MyEnsemble
          class: emloop.models.Ensemble

          inputs: [images]
          outputs: [predictions]

          models_root: /var/project/models  # will load all the models under this directory

    .. code-block:: python
        :caption: usage from python

        import emloop as el

        model = el.models.Ensemble(inputs=['images'], outputs=['predictions'], aggregation='mean',
                                   models_root='/my/directory/with/models')
        # model.run(...)

    """

    AGGREGATION_METHODS = ['mean', 'major_vote']
    """Possible ensemble aggregation methods."""

    def __init__(self,
                 inputs: Sequence[str],
                 outputs: Sequence[str],
                 aggregation: str='major_vote',
                 models_root: Optional[str]=None,
                 model_paths: Optional[Sequence[str]]=None,
                 dataset: Optional[AbstractDataset]=None,
                 eager_loading: bool=False,
                 **kwargs):
        """
        Create new Ensemble.

        If no ``models_paths`` are specified, all the sub-directories of the ``models_root`` will be taken as the models
        to be assembled together.

        :param inputs: model input names
        :param outputs: model output names
        :param aggregation: aggregation method, one of :py:attr:`Ensemble.AGGREGATION_METHODS`
        :param models_root: optional root directory of the models to be assembled together
        :param model_paths: optional list of model directory names/paths
        :param dataset: optional **emloop** dataset (will be passed to the assembled models)
        :param eager_loading: load all the models in the constructor
        :param kwargs: additional kwargs (unused)
        :raise AssertionError: if neither one of ``models_root`` and ``model_paths`` is specified
        :raise AssertionError: if the specified ``aggregation`` is not one of :py:attr:`Ensemble.AGGREGATION_METHODS`
        """
        assert aggregation in Ensemble.AGGREGATION_METHODS, 'Unsupported aggregation {} (supported: {}).'.format(
                                                                        aggregation, Ensemble.AGGREGATION_METHODS)
        assert models_root is not None or model_paths is not None, 'Either `models_root` or `model_paths` ' \
                                                                   'must be specified.'

        if model_paths is None:
            model_paths = next(os.walk(models_root))[1]
        if models_root is not None:
            model_paths = [path.join(models_root, model_path) for model_path in model_paths]
        self._model_paths = model_paths

        self._dataset = dataset
        self._inputs = inputs
        self._outputs = outputs
        self._aggregation = aggregation
        self._kwargs = kwargs

        self._models = None
        if eager_loading:
            self._load_models()

    def _load_models(self) -> None:
        """Maybe load all the models to be assembled together and save them to the ``self._models`` attribute."""
        if self._models is None:
            logging.info('Loading %d models', len(self._model_paths))

            def load_model(model_path: str):
                logging.debug('\tloading %s', model_path)
                if path.isdir(model_path):
                    model_path = path.join(model_path, EL_CONFIG_FILE)
                config = load_config(model_path)
                config['model']['inputs'] = self._inputs
                config['model']['outputs'] = self._outputs

                return emloop.api.create_model(config, output_dir=None, dataset=self._dataset,
                                               restore_from=path.dirname(model_path))

            self._models = list(map(load_model, self._model_paths))

    @property
    def input_names(self) -> Iterable[str]:
        """List of model input names."""
        return self._inputs

    @property
    def output_names(self) -> Iterable[str]:
        """List of model output names."""
        return self._outputs

    def run(self, batch: Batch, train: bool=False, stream: StreamWrapper=None) -> Batch:
        """
        Run feed-forward pass with the given batch using all the models, aggregate and return the results.

        .. warning::
            :py:class:`Ensemble` can not be trained.

        :param batch: batch to be processed
        :param train: ``True`` if this batch should be used for model update, ``False`` otherwise
        :param stream: stream wrapper (useful for precise buffer management)
        :return: aggregated results dict
        :raise ValueError: if the ``train`` flag is set to ``True``
        """
        if train:
            raise ValueError('Ensemble model cannot be trained.')
        self._load_models()

        # run all the models
        batch_outputs = [model.run(batch, False, stream) for model in self._models]

        # aggregate the outputs
        aggregated = {}
        for output_name in self._outputs:
            output_values = [batch_output[output_name] for batch_output in batch_outputs]
            if self._aggregation == 'mean':
                aggregated[output_name] = np.mean(output_values, axis=0)
            elif self._aggregation == 'major_vote':
                output_values_arr = np.array(output_values)
                output = major_vote(output_values_arr.reshape((output_values_arr.shape[0], -1)))
                aggregated[output_name] = np.array(output).reshape(output_values_arr[0].shape)

        return aggregated

    def save(self, *args, **kwargs) -> None:
        """
        Ensemble model can not be saved.

        :raise NotImplementedError: when called
        """
        raise NotImplementedError('Ensemble model cannot be saved.')
