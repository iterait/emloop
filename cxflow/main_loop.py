"""
**cxflow** main loop for training models.

The MainLoop requires AbstractModel, AbstractDataset and a list of AbstractHooks.
Having all that, it manages iterating through streams, training and hooks execution.
"""
import logging
from typing import Iterable, Callable, List, Dict, Optional
from collections import OrderedDict

from .datasets import AbstractDataset
from .models.abstract_model import AbstractModel
from .hooks.abstract_hook import AbstractHook, TrainingTerminated
from .utils import Timer
from .utils.misc import CaughtInterrupts
from .datasets.stream_wrapper import StreamWrapper
from .constants import CXF_TRAIN_STREAM, CXF_PREDICT_STREAM
from .types import EpochData


class MainLoop(CaughtInterrupts):   # pylint: disable=too-many-instance-attributes
    """**cxflow** main loop for training and model inference."""

    UNUSED_SOURCE_ACTIONS = ['ignore', 'warn', 'error']
    """Possible actions to be taken when a stream source is unused by the trained model."""

    def __init__(self,   # pylint: disable=too-many-arguments
                 model: AbstractModel, dataset: AbstractDataset,
                 hooks: Iterable[AbstractHook]=(),
                 extra_streams: List[str]=(),  # pylint: disable=invalid-sequence-index
                 buffer: int=0,
                 on_unused_sources: str='warn',
                 fixed_batch_size: Optional[int]=None,
                 fixed_epoch_size: Optional[int]=None,
                 skip_zeroth_epoch: bool=False):
        """
        :param model: trained model
        :param dataset: loaded dataset
        :param hooks: training hooks
        :param extra_streams: additional stream names to be evaluated between epochs
        :param buffer: size of the batch buffer, 0 means no buffer
        :param on_unused_sources: action to take when stream provides an unused sources; one of
            :py:attr:`UNUSED_SOURCE_ACTIONS`
        :param fixed_batch_size: if specified, main_loop removes all batches that do not have the specified size
        :param fixed_epoch_size: if specified, cut the train stream to epochs of at most ``fixed_epoch_size`` batches
        :param skip_zeroth_epoch: if specified, main loop skips the 0th epoch
        """
        assert on_unused_sources in MainLoop.UNUSED_SOURCE_ACTIONS

        self._model = model
        self._dataset = dataset
        self._hooks = hooks
        self._buffer = buffer
        self._on_unused_sources = on_unused_sources
        self._fixed_batch_size = fixed_batch_size
        self._fixed_epoch_size = fixed_epoch_size
        self._extra_sources_warned = False
        self._epoch_profile = {}
        self._extra_streams = list(extra_streams)
        self._skip_zeroth_epoch = skip_zeroth_epoch
        self._streams = {}
        self._epochs_done = None

        super().__init__()

    @property
    def epochs_done(self) -> Optional[int]:
        """Number of training epochs done in the last call of :py:meth:`self._run_training`."""
        return self._epochs_done

    @property
    def fixed_epoch_size(self) -> Optional[int]:
        """Fixed epoch size parameter as specified in :py:meth:`self.__init__`."""
        return self._fixed_epoch_size

    @property
    def extra_streams(self) -> List[str]:
        """List of extra stream names as specified in :py:meth:`self.__init__`."""
        return self._extra_streams

    def _create_epoch_data(self) -> EpochData:
        """Create empty epoch data double dict."""
        return OrderedDict([(stream_name, OrderedDict())
                            for stream_name in [CXF_TRAIN_STREAM] + self._extra_streams])

    def _check_sources(self, batch: Dict[str, object]) -> None:
        """
        Check for unused and missing sources.

        :param batch: batch to be checked
        :raise ValueError: if a source is missing or unused and ``self._on_unused_sources`` is set to ``error``
        """
        unused_sources = [source for source in batch.keys() if source not in self._model.input_names]
        missing_sources = [source for source in self._model.input_names if source not in batch.keys()]
        # check stream sources
        if unused_sources:
            if self._on_unused_sources == 'warn' and not self._extra_sources_warned:
                logging.warning('Some sources provided by the stream do not match model placeholders. Set '
                                '`main_loop.on_unused_sources` to `ignore` in order to suppress this warning. '
                                'Extra sources: %s', unused_sources)
                self._extra_sources_warned = True
            elif self._on_unused_sources == 'error':
                raise ValueError('Some sources provided by the stream do not match model placeholders. Set'
                                 '`main_loop.on_unused_sources` to `warn` in order to suppress this error.\n'
                                 'Extra sources: {}'.format(unused_sources))

        if missing_sources:
            raise ValueError('Stream does not provide all required sources. Missing sources: {}'
                             .format(missing_sources))

    def _run_epoch(self, stream: StreamWrapper, train: bool) -> None:
        """
        Iterate through the given stream and evaluate/train the model with the received batches.

        Calls :py:meth:`cxflow.hooks.AbstractHook.after_batch` events.

        :param stream: stream to iterate
        :param train: if set to ``True``, the model will be trained
        :param stream_name: stream name
        """
        for batch_input in stream:
            self.raise_check_interrupt()

            if self._fixed_batch_size:
                if len(batch_input[list(batch_input.keys())[0]]) != self._fixed_batch_size:
                    logging.debug('Incomplete batch skipped')
                    continue

            self._check_sources(batch_input)

            with Timer('eval_batch_{}'.format(stream.name), self._epoch_profile):
                batch_output = self._model.run(batch=batch_input, train=train)
            assert set(batch_input.keys()).isdisjoint(set(batch_output)), 'Batch inputs and outputs must not overlap.'

            with Timer('after_batch_hooks_{}'.format(stream.name), self._epoch_profile):
                for hook in self._hooks:
                    hook.after_batch(stream_name=stream.name, batch_data={**batch_input, **batch_output})

    def train_by_stream(self, stream: StreamWrapper) -> None:
        """
        Train the model with the given stream.

        :param stream: stream to train with
        """

        self._run_epoch(stream=stream, train=True)

    def evaluate_stream(self, stream: StreamWrapper) -> None:
        """
        Evaluate the given stream.

        :param stream: stream to be evaluated
        :param stream_name: stream name
        """
        self._run_epoch(stream=stream, train=False)

    def get_stream(self, stream_name: str) -> StreamWrapper:
        """
        Get a :py:class:`StreamWrapper` with the given name.

        :param stream_name: stream name
        :return: dataset function name providing the respective stream
        :raise AttributeError: if the dataset does not provide the function creating the stream
        """
        if stream_name not in self._streams:
            stream_fn_name = '{}_stream'.format(stream_name)
            try:
                stream_fn = getattr(self._dataset, stream_fn_name)
                stream_epoch_limit = -1
                if self._fixed_epoch_size is not None and stream_name == CXF_TRAIN_STREAM:
                    stream_epoch_limit = self._fixed_epoch_size
                self._streams[stream_name] = StreamWrapper(stream_fn, buffer_size=self._buffer,
                                                           epoch_size=stream_epoch_limit, name=stream_name,
                                                           profile=self._epoch_profile)
            except AttributeError as ex:
                raise AttributeError('The dataset does not have a function for creating a stream named `{}`. '
                                     'The function has to be named `{}`.'.format(stream_name, stream_fn_name)) from ex
        return self._streams[stream_name]

    def _run_zeroth_epoch(self, streams: Iterable[str]) -> None:
        """
        Run zeroth epoch on the specified streams.

        Calls
            - :py:meth:`cxflow.hooks.AbstractHook.after_epoch`

        :param streams: stream names to be evaluated
        """
        for stream_name in streams:
            with self.get_stream(stream_name) as stream:
                self.evaluate_stream(stream)

        epoch_data = self._create_epoch_data()
        for hook in self._hooks:
            hook.after_epoch(epoch_id=0, epoch_data=epoch_data)

    def _try_run(self, run_func: Callable[[], None]) -> None:
        """
        Try running the given function (training/prediction).

        Calls
            - :py:meth:`cxflow.hooks.AbstractHook.before_training`
            - :py:meth:`cxflow.hooks.AbstractHook.after_training`

        :param run_func: function to be run
        """
        # Initialization: before_training
        for hook in self._hooks:
            hook.before_training()

        try:
            run_func()
        except TrainingTerminated as ex:
            logging.info('Training terminated: %s', ex)

        # After training: after_training
        for hook in self._hooks:
            hook.after_training()

    def run_training(self) -> None:
        """
        Run the main loop in the training mode.

        Calls
            - :py:meth:`cxflow.hooks.AbstractHook.after_epoch`
            - :py:meth:`cxflow.hooks.AbstractHook.after_epoch_profile`
        """
        for stream_name in [CXF_TRAIN_STREAM] + self._extra_streams:
            self.get_stream(stream_name)

        def training():
            logging.debug('Training started')
            self._epochs_done = 0

            # Zeroth epoch: after_epoch
            if not self._skip_zeroth_epoch:
                self._run_zeroth_epoch([CXF_TRAIN_STREAM] + self._extra_streams)
                logging.info('Zero epoch done\n\n')

            # Training loop: after_epoch, after_epoch_profile
            while True:
                epoch_id = self._epochs_done + 1
                self._epoch_profile.clear()
                epoch_data = self._create_epoch_data()

                with self.get_stream(CXF_TRAIN_STREAM) as stream:
                    self.train_by_stream(stream)

                for stream_name in self._extra_streams:
                    with self.get_stream(stream_name) as stream:
                        self.evaluate_stream(stream)

                logging.info('After epoch %s', epoch_id)

                with Timer('after_epoch_hooks', self._epoch_profile):
                    for hook in self._hooks:
                        hook.after_epoch(epoch_id=epoch_id, epoch_data=epoch_data)

                for hook in self._hooks:
                    hook.after_epoch_profile(epoch_id=epoch_id, profile=self._epoch_profile,
                                             extra_streams=self._extra_streams)
                self._epochs_done = epoch_id
                logging.info('Epochs done: %s\n\n', epoch_id)

        self._try_run(training)

    def run_prediction(self) -> None:
        """Run the main loop for in the prediction mode."""
        def prediction():
            logging.debug('Prediction started')
            self._run_zeroth_epoch([CXF_PREDICT_STREAM])
        self._try_run(prediction)
