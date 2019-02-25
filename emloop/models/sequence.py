import copy
import typing
import logging
import os.path as path
from itertools import chain
from typing import Iterable, Optional

import emloop.api   # avoids circular dependency

from .abstract_model import AbstractModel
from ..utils import load_config
from ..datasets import AbstractDataset, StreamWrapper
from ..types import Batch
from ..constants import EL_CONFIG_FILE


class Sequence(AbstractModel):
    """
    Model sequence provides simple abstraction for sequential application of multiple models to input batches.
    All the models are fed with the original inputs as well as the outputs of the preceding models.
    Ultimately, all the model outputs are returned.

    .. warning::
        Model sequence can be used for inference only (i.e. no training is supported).

    .. code-block:: yaml
        :caption: usage from config

        model:
          name: MyPipeline
          class: emloop.models.Sequence

          models_root: /var/project/models
          model_paths: [step1, step2, step3]

    .. code-block:: python
        :caption: usage from python

        import emloop as el

        model = el.models.Sequence(model_paths=['/path/to/step/step1', '/path/to/step/step2'])
        # model.run(...)

    """

    def __init__(self,
                 model_paths: typing.Sequence[str],
                 models_root: Optional[str]=None,
                 dataset: Optional[AbstractDataset]=None,
                 eager_loading: bool=False,
                 **_):
        """
        Create new model ``Sequence``.

        :param models_root: optional root directory of the models to be assembled together
        :param model_paths: list of model directory names/paths
        :param dataset: optional **emloop** dataset (will be passed to the underlying models)
        :param eager_loading: load all the models in the constructor
        """
        if models_root is not None:
            model_paths = [path.join(models_root, model_path) for model_path in model_paths]
        self._model_paths = model_paths
        self._dataset = dataset

        self._models = None
        if eager_loading:
            self._load_models()

    def _load_models(self) -> None:
        """Maybe load all the models to be applied and save them to the ``self._models`` attribute."""
        if self._models is None:
            logging.info('Loading %d models', len(self._model_paths))

            def load_model(model_path: str):
                logging.debug('\tloading %s', model_path)
                if path.isdir(model_path):
                    model_path = path.join(model_path, EL_CONFIG_FILE)
                return emloop.api.create_model(load_config(model_path), output_dir=None, dataset=self._dataset,
                                               restore_from=path.dirname(model_path))

            self._models = list(map(load_model, self._model_paths))

    @property
    def input_names(self) -> Iterable[str]:
        """List of model input names."""
        self._load_models()
        return self._models[0].input_names

    @property
    def output_names(self) -> Iterable[str]:
        """List of model output names."""
        self._load_models()
        return chain.from_iterable(map(lambda m: m.output_names, self._models))

    def run(self, batch: Batch, train: bool=False, stream: StreamWrapper=None) -> Batch:
        """
        Run all the models in-order and return accumulated outputs.

        N-th model is fed with the original inputs and outputs of all the models that were run before it.

        .. warning::
            :py:class:`Sequence` model can not be trained.

        :param batch: batch to be processed
        :param train: ``True`` if this batch should be used for model update, ``False`` otherwise
        :param stream: stream wrapper (useful for precise buffer management)
        :return: accumulated model outputs
        :raise ValueError: if the ``train`` flag is set to ``True``
        """
        if train:
            raise ValueError('Ensemble model cannot be trained.')
        self._load_models()

        # run all the models in-order
        current_batch = dict(copy.deepcopy(batch))
        for model in self._models:
            batch_inputs = set(list(current_batch.keys()))
            missing_inputs = set(model.input_names)-batch_inputs
            if missing_inputs != set():
                raise ValueError(f'Model `{model.__class__.__name__}` expects inputs `{missing_inputs}` which are '
                                 f'missing among currently passed in inputs `{batch_inputs}`.')
            current_batch.update(model.run(current_batch, False, None))

        return {key: current_batch[key] for key in self.output_names}

    def save(self, *args, **kwargs) -> None:
        """
        Sequence model can not be saved.

        :raise NotImplementedError: when called
        """
        raise NotImplementedError('Ensemble model cannot be saved.')
