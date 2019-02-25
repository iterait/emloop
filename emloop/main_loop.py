"""
**emloop** main loop for training models.

The MainLoop requires AbstractModel, AbstractDataset and a list of AbstractHooks.
Having all that, it manages iterating through streams, training and hooks execution.
"""
import logging
from typing import Iterable, Callable, List, Dict, Optional, Union
from collections import OrderedDict

from .datasets import AbstractDataset
from .models.abstract_model import AbstractModel
from .hooks.abstract_hook import AbstractHook, TrainingTerminated
from .hooks.training_trace import TrainingTrace
from .utils import Timer
from .utils.misc import CaughtInterrupts
from .datasets.stream_wrapper import StreamWrapper
from .constants import EL_DEFAULT_TRAIN_STREAM, EL_PREDICT_STREAM
from .types import EpochData


class MainLoop(CaughtInterrupts):   # pylint: disable=too-many-instance-attributes
    """**emloop** main loop for training and model inference."""

    EMPTY_ACTIONS = ['ignore', 'warn', 'error']
    """Possible actions to be taken when a batch/stream is empty."""
    UNUSED_SOURCE_ACTIONS = ['ignore', 'warn', 'error']
    """Possible actions to be taken when a stream source is unused by the trained model."""
    INCORRECT_CONFIG_ACTIONS = ['ignore', 'warn', 'error']
    """Possible actions to be taken when a mainloop config contains some unexpected arguments."""

    def __init__(self,   # pylint: disable=too-many-arguments
                 model: AbstractModel, dataset: AbstractDataset,
                 hooks: Iterable[AbstractHook]=(),
                 train_stream_name: str=EL_DEFAULT_TRAIN_STREAM,
                 extra_streams: Iterable[str]=(),  # pylint: disable=invalid-sequence-index
                 buffer: int=0,
                 on_empty_batch: str='error',
                 on_empty_stream: str='error',
                 on_unused_sources: str='warn',
                 on_incorrect_config: str= 'error',
                 fixed_batch_size: Optional[int]=None,
                 fixed_epoch_size: Optional[int]=None,
                 skip_zeroth_epoch: bool=False,
                 **kwargs):
        """
        :param model: trained model
        :param dataset: loaded dataset
        :param hooks: training hooks
        :param train_stream_name: name of the training stream
        :param extra_streams: additional stream names to be evaluated between epochs
        :param buffer: size of the batch buffer, 0 means no buffer
        :param on_empty_batch: action to take when batch is empty; one of :py:attr:`MainLoop.EMPTY_ACTIONS`
        :param on_empty_stream: action to take when stream is empty; one of :py:attr:`MainLoop.EMPTY_ACTIONS`
        :param on_unused_sources: action to take when stream provides an unused sources; one of
            :py:attr:`UNUSED_SOURCE_ACTIONS`
        :param on_incorrect_config: action to take when mainloop config contains unexpected arguments; one of
            :py:attr:`MainLoop.INCORRECT_CONFIG_ACTIONS`
        :param fixed_batch_size: if specified, main_loop removes all batches that do not have the specified size
        :param fixed_epoch_size: if specified, cut the train stream to epochs of at most ``fixed_epoch_size`` batches
        :param skip_zeroth_epoch: if specified, main loop skips the 0th epoch
        :raise AssertionError: in case of unsupported value of ``on_empty_batch``, ``on_empty_stream`` or \
        ``on_unused_sources``
        """
        assert on_empty_batch in MainLoop.EMPTY_ACTIONS
        assert on_empty_stream in MainLoop.EMPTY_ACTIONS
        assert on_unused_sources in MainLoop.UNUSED_SOURCE_ACTIONS
        assert on_incorrect_config in MainLoop.INCORRECT_CONFIG_ACTIONS

        if kwargs:
            if on_incorrect_config == 'error':
                raise ValueError('Config yaml contains some unexpected arguments in mainloop section. '
                                 'Set `main_loop.on_incorrect_config` to `warn` in order to suppress this error.\n'
                                 'Extra arguments: {}'.format(kwargs))
            elif on_incorrect_config == 'warn':
                logging.warning('Config yaml contains some unexpected arguments in mainloop section. '
                                'Set `main_loop.on_incorrect_config` to `ignore` in order to suppress this warning. '
                                'Extra arguments: %s', kwargs)

        self._model = model
        self._dataset = dataset
        self._hooks = hooks
        self._buffer = buffer
        self._on_empty_batch = on_empty_batch
        self._on_empty_stream = on_empty_stream
        self._on_unused_sources = on_unused_sources
        self._fixed_batch_size = fixed_batch_size
        self._fixed_epoch_size = fixed_epoch_size
        self._extra_sources_warned = False
        self._epoch_profile = {}
        self._train_stream_name = train_stream_name
        self._extra_streams = list(extra_streams)
        self._skip_zeroth_epoch = skip_zeroth_epoch
        self._streams = {}
        self._training_epochs_done = 0

        for hook in self._hooks:
            hook.register_mainloop(self)

        super().__init__()

    def __enter__(self):
        """Calls before_training() for all hooks."""
        CaughtInterrupts.__enter__(self)
        for hook in self._hooks:
            hook.before_training()

    def __exit__(self, exc_type, exc_value, traceback):
        """Calls after_training() for all hooks."""
        CaughtInterrupts.__exit__(self)
        for hook in self._hooks:
            success = exc_type == None
            hook.after_training(success)

    @property
    def training_epochs_done(self) -> Optional[int]:
        """Number of training epochs done."""
        return self._training_epochs_done

    @property
    def fixed_epoch_size(self) -> Optional[int]:
        """Fixed epoch size parameter as specified in :py:meth:`self.__init__`."""
        return self._fixed_epoch_size

    @property
    def extra_streams(self) -> List[str]:
        """List of extra stream names as specified in :py:meth:`self.__init__`."""
        return self._extra_streams

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

        Calls :py:meth:`emloop.hooks.AbstractHook.after_batch` events.

        :param stream: stream to iterate
        :param train: if set to ``True``, the model will be trained
        :raise ValueError: in case of empty batch when ``on_empty_batch`` is set to ``error``
        :raise ValueError: in case of empty stream when ``on_empty_stream`` is set to ``error``
        :raise ValueError: in case of two batch variables having different lengths
        """
        nonempty_batch_count = 0
        for i, batch_input in enumerate(stream):
            self.raise_check_interrupt()

            batch_sizes = {len(source) for source in batch_input.values()}
            if len(batch_sizes) == 0 or batch_sizes == {0}:
                if self._on_empty_batch == 'warn':
                    logging.warning('%i-th batch in stream `%s` appears to be empty (%i-th empty batch in total). Set '
                                    '`main_loop.on_empty_batch` to `ignore` in order to suppress this warning.',
                                    i, stream.name, nonempty_batch_count)
                elif self._on_empty_batch == 'error':
                    raise ValueError('{}-th batch in stream `{}` appears to be empty ({}-th empty batch in total). Set '
                                     '`main_loop.on_empty_batch` to `warn` in order to change this error into warning; '
                                     'set to `ignore` to remove it.'.format(i, stream.name, nonempty_batch_count))
                continue
            elif self._fixed_batch_size:
                if batch_sizes != {self._fixed_batch_size}:
                    var, len_ = [(k, len(v)) for k, v in batch_input.items() if len(v) != self._fixed_batch_size][0]
                    logging.debug('%i-th batch in stream `%s` has variable `%s` of length %i inconsistent with '
                                  '`main_loop.fixed_size` = %i', i, stream.name, var, len_, self._fixed_batch_size)
                    continue
            nonempty_batch_count += 1

            self._check_sources(batch_input)

            with Timer('eval_batch_{}'.format(stream.name), self._epoch_profile):
                batch_output = self._model.run(batch=batch_input, train=train, stream=stream)
            assert set(batch_input.keys()).isdisjoint(set(batch_output)), 'Batch inputs and outputs must not overlap.'

            with Timer('after_batch_hooks_{}'.format(stream.name), self._epoch_profile):
                batch_data = {**batch_input, **batch_output}
                for hook in self._hooks:
                    hook.after_batch(stream_name=stream.name, batch_data=batch_data)
        if nonempty_batch_count == 0:
            if self._on_empty_stream == 'warn':
                logging.warning('Stream `%s` appears to be empty. Set `main_loop.on_empty_stream` to `ignore` in order '
                                'to suppress this warning.', stream.name)
            elif self._on_empty_stream == 'error':
                raise ValueError('Stream `{}` appears to be empty. Set '
                                 '`main_loop.on_empty_stream` to `warn` in order to change this error into warning; '
                                 'set to `ignore` to remove it.'.format(stream.name))

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
                if self._fixed_epoch_size is not None and stream_name == self._train_stream_name:
                    stream_epoch_limit = self._fixed_epoch_size
                self._streams[stream_name] = StreamWrapper(stream_fn, buffer_size=self._buffer,
                                                           epoch_size=stream_epoch_limit, name=stream_name,
                                                           profile=self._epoch_profile)
            except AttributeError as ex:
                raise AttributeError('The dataset does not have a function for creating a stream named `{}`. '
                                     'The function has to be named `{}`.'.format(stream_name, stream_fn_name)) from ex
        return self._streams[stream_name]
    
    def prepare_streams(self, stream_list: Iterable[Union[Iterable, StreamWrapper, str]],
                        base_name: str) -> Iterable[str]:
        """
        Converts streams to StreamWrappers, saves them to `self._streams` and returns their names as strings.

        :param stream_list: list of training streams, each either string (e.g. 'train'), StreamWrapper or iterator
        :param base_name: default base name for unnamed streams
        """
        stream_names = []
        unnamed_count = 0
        for stream_object in stream_list:
            stream_name = None
            if isinstance(stream_object, str):
                stream_name = stream_object
                streamwrapper = self.get_stream(stream_object)
            
            elif isinstance(stream_object, StreamWrapper):
                stream_name = stream_object.name
                streamwrapper = stream_object

            else:
                streamwrapper = StreamWrapper(lambda stream_object=stream_object: stream_object,
                                              buffer_size=self._buffer, profile=self._epoch_profile)

            if stream_name is None:
                stream_name = f"unnamed_{base_name}_{unnamed_count}"
                unnamed_count += 1

            if stream_name in self._streams:
                ValueError(f"Multiple streams with name `{stream_name}`")

            self._streams[stream_name] = streamwrapper
            stream_names.append(stream_name)
        
        return stream_names
    
    def epoch(self, train_streams: Iterable[Union[Iterable, StreamWrapper, str]],
              eval_streams: Iterable[Union[Iterable, StreamWrapper, str]]) -> None:
        """
        Runs single epoch with given streams.

        :param train_streams: list of training streams, each either string (e.g. 'train'), StreamWrapper or iterator
        :param eval_streams: list of eval streams, each either string (e.g. 'valid'), StreamWrapper or iterator
        """
        self._streams = {}
        train_streams = self.prepare_streams(train_streams, "train")
        eval_streams = self.prepare_streams(eval_streams, "eval")

        self._epoch_impl(train_streams, eval_streams)
        self._streams = {}

    def _epoch_impl(self, train_streams: Iterable[str], eval_streams: Iterable[str]) -> None:
        """
        Runs single epoch with given streams.

        :param train_streams: list of training streams
        :param eval_streams: list of eval streams
        """
        self._epoch_profile.clear()
        for stream_name in train_streams:
            with self.get_stream(stream_name) as stream:
                self._run_epoch(stream=stream, train=True)

        for stream_name in eval_streams:
            with self.get_stream(stream_name) as stream:
                self._run_epoch(stream=stream, train=False)

        if len(train_streams) > 0:
            self._training_epochs_done += 1

        epoch_data = OrderedDict([(stream_name, OrderedDict()) for stream_name in train_streams + eval_streams])

        end_training_exception = None
        with Timer('after_epoch_hooks', self._epoch_profile):
            for hook in self._hooks:
                try:
                    hook.after_epoch(epoch_id=self._training_epochs_done, epoch_data=epoch_data)
                except TrainingTerminated as ex:
                    end_training_exception = ex

        for hook in self._hooks:
            hook.after_epoch_profile(epoch_id=self._training_epochs_done, profile=self._epoch_profile,
                                     streams=train_streams + eval_streams)

        if end_training_exception:
            raise end_training_exception

    def run_training(self) -> None:
        """
        Trains until TrainingTerminated exception is raised.
        """
        if len(list(filter(lambda x: isinstance(x, TrainingTrace), self._hooks))) == 0:
            logging.warning("TrainingTrace hook missing - trace.yaml will not be generated")

        with self:
            try:
                logging.debug('Training started')

                # Zeroth epoch
                if not self._skip_zeroth_epoch:
                    logging.info('Evaluating 0th epoch')
                    self._epoch_impl([], [self._train_stream_name] + self._extra_streams)
                    logging.info('0th epoch done\n\n')

                while True:
                    logging.info('Training epoch %s', self._training_epochs_done + 1)
                    self._epoch_impl([self._train_stream_name], self._extra_streams)
                    logging.info('Epoch %s done\n\n', self._training_epochs_done)

            except TrainingTerminated as ex:
                logging.info('Training terminated: %s', ex)

    def run_evaluation(self, stream_name) -> None:
        """
        Evaluates given stream.

        :param stream_name: Name of stream to evaluate (e.g. `valid`).
        """
        with self:
            try:
                logging.info('Running the evaluation of stream `%s`', stream_name)
                self._epoch_impl(train_streams=[], eval_streams=[stream_name])
                logging.info('Evaluation done\n\n')
            except TrainingTerminated as ex:
                logging.info('Evaluation terminated: %s', ex)
