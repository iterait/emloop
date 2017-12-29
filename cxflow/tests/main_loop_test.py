"""
Test module for the main loop (cxflow.main_loop).
"""
import time
from collections import defaultdict
from typing import Mapping, List, Iterable

import numpy as np
from testfixtures import LogCapture

import cxflow as cx
from cxflow.constants import CXF_BUFFER_SLEEP
from cxflow.datasets import StreamWrapper
from cxflow.hooks import StopAfter
from cxflow.utils.profile import Timer
from cxflow.types import EpochData, Batch, Stream, TimeProfile

from .test_core import CXTestCaseWithDir

_READ_DATA_SLEEP_S = 0.1
_AFTER_BATCH_SLEEP_S = 0.2
_AFTER_EPOCH_SLEEP_S = 0.3
_MODEL_RUN_SLEEP_S = 0.5

_DATASET_ITERS = 13
_DATASET_SHAPE = (11, 10)

_EPOCH_DATA_VAR_VALUE = 11


class SimpleDataset(cx.AbstractDataset):
    """Simple dataset with train, valid and test streams."""

    def __init__(self):
        super().__init__(config_str='')
        self.iters = _DATASET_ITERS
        self.shape = _DATASET_SHAPE
        self.train_used = self.valid_used = self.test_used = self.predict_used = False
        self.batches = defaultdict(lambda: [])
        self.source_names = ['input', 'target']
        self._iter = 1

    def stream(self, stream_name: str)-> Stream:
        """Generate a datastream with increasing inputs and constant target."""
        for _ in range(self.iters):
            batch = {'input': self._iter * np.ones(self.shape), 'target': np.zeros(self.shape)}
            self.batches[stream_name].append(batch)
            self._iter += 1
            yield batch

    def train_stream(self) -> Stream:
        self.train_used = True
        for batch in self.stream('train'):
            yield batch

    def valid_stream(self) -> Stream:
        self.valid_used = True
        for batch in self.stream('valid'):
            yield batch

    def test_stream(self) -> Stream:
        self.test_used = True
        for batch in self.stream('test'):
            yield batch

    def predict_stream(self) -> Stream:
        self.predict_used = True
        for batch in self.stream('predict'):
            yield batch


class ExtendedDataset(SimpleDataset):
    """SimpleDataset extension with additional 'unused' source in the train stream."""

    def train_stream(self) -> Stream:
        self.train_used = True
        for _ in range(self.iters):
            yield {'input': np.ones(self.shape), 'target': np.zeros(self.shape), 'unused': np.zeros(self.shape)}


class DelayedDataset(SimpleDataset):
    """SimpleDataset extension which sleeps briefly before each train batch allowing to measure the data read time."""

    def train_stream(self) -> Stream:
        for _ in range(self.iters):
            time.sleep(_READ_DATA_SLEEP_S)
            yield {'input': np.ones(self.shape), 'target': np.zeros(self.shape)}

class ShortSimpleDataset(SimpleDataset):
    """SimpleDataset extension with one batch and one variable"""

    def stream(self, stream_name: str) -> Stream:
        for _ in range(1):
            yield {'input': self._iter * np.ones(self.shape)}

class EmptyStreamDataset(SimpleDataset):
    """SimpleDataset extension providing empty streams."""

    def stream(self, stream_name: str):
        return iter([])

class SomeEmptyBatchDataset(SimpleDataset):
    """SimpleDataset extension with empty first batch."""

    def __init__(self):
        super().__init__()

    def stream(self, stream_name: str):
        for i in range(self.iters):
            if i == 0:
                batch =  {'input': [], 'target': []}
            else:
                batch = {'input': self._iter * np.ones(self.shape), 'target': np.zeros(self.shape)}
            yield batch

class AllEmptyBatchDataset(SimpleDataset):
    """SimpleDataset extension with all batches empty."""

    def stream(self, stream_name: str):
        for _ in range(10):
            yield {'input': [], 'target': []}

class EventRecordingHook(cx.AbstractHook):
    """EventRecordingHook records all the events and store their count and order."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._event_id = 1
        self.before_training_events = []
        self.after_batch_events = []
        self.after_epoch_events = []
        self.after_epoch_profile_events = []
        self.after_training_events = []

    def before_training(self) -> None:
        self.before_training_events.append(self._event_id)
        self._event_id += 1

    def after_batch(self, stream_name: str, batch_data: Batch) -> None:
        self.after_batch_events.append(self._event_id)
        self._event_id += 1

    def after_epoch(self, epoch_id: int, epoch_data: EpochData) -> None:
        self.after_epoch_events.append(self._event_id)
        self._event_id += 1

    def after_epoch_profile(self, epoch_id: int, profile: TimeProfile, extra_streams: Iterable[str]) -> None:
        self.after_epoch_profile_events.append(self._event_id)
        self._event_id += 1

    def after_training(self) -> None:
        self.after_training_events.append(self._event_id)
        self._event_id += 1


class DataRecordingHook(cx.AbstractHook):
    """DataRecordingHook records epoch_ids and all the batch_data."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.epoch_ids = []
        self.batch_data = defaultdict(lambda: [])

    def after_batch(self, stream_name: str, batch_data: Batch) -> None:
        self.batch_data[stream_name].append(batch_data)

    def after_epoch(self, epoch_id: int, epoch_data: EpochData) -> None:
        self.epoch_ids.append(epoch_id)


class DelayedHook(cx.AbstractHook):
    """DelayedHook sleeps briefly in after_batch and after_epoch events allowing to measure hook processing times."""

    def after_batch(self, stream_name: str, batch_data: Batch) -> None:
        time.sleep(_AFTER_BATCH_SLEEP_S)

    def after_epoch(self, epoch_id: int, epoch_data: EpochData) -> None:
        time.sleep(_AFTER_EPOCH_SLEEP_S)


class SaveProfileHook(cx.AbstractHook):
    """SaveProfileHook saves the epoch profile dict to self.profile."""

    def __init__(self):
        super().__init__()
        self.profile = None

    def after_epoch_profile(self, epoch_id: int, profile: TimeProfile, extra_streams: Iterable[str]) -> None:
        """Save the profile to self.profile."""
        self.profile = profile


class EpochDataProducer(cx.AbstractHook):
    """Simple hook that adds my_variable to the train entry in the epoch_data."""

    def after_epoch(self, epoch_id: int, epoch_data: EpochData) -> None:
        epoch_data['train']['my_variable'] = _EPOCH_DATA_VAR_VALUE


class EpochDataConsumer(cx.AbstractHook):
    """Simple hook that asserts presence of my_variable in the train entry of the epoch_data."""

    def after_epoch(self, epoch_id: int, epoch_data: EpochData) -> None:
        assert 'train' in epoch_data
        assert 'my_variable' in epoch_data['train']
        assert epoch_data['train']['my_variable'] == _EPOCH_DATA_VAR_VALUE


class TrainableModel(cx.AbstractModel):
    """Simple trainable model"""
    def __init__(self, io: dict, **kwargs):  # pylint: disable=unused-argument
        self._input_names = io['in']
        self._output_names = io['out']

    def run(self, batch: Mapping[str, object], train: bool, stream: StreamWrapper) -> Mapping[str, object]:
        return {o: i for i, o in enumerate(self._output_names)}

    def save(self, name_suffix: str) -> str:
        pass

    @property
    def input_names(self) -> List[str]:   # pylint: disable=invalid-sequence-index
        """List of tf tensor names listed as model inputs."""
        return self._input_names

    @property
    def output_names(self) -> List[str]:   # pylint: disable=invalid-sequence-index
        """List of tf tensor names listed as model outputs."""
        return self._output_names

    @property
    def restore_fallback(self) -> str:
        return ''


class DelayedModel(TrainableModel):
    """Trainable model which sleeps briefly when processing a batch allowing to measure the model eval time."""

    def run(self, batch: Mapping[str, object], train: bool, stream: StreamWrapper):
        with stream.allow_buffering:
            time.sleep(_MODEL_RUN_SLEEP_S)
        return super().run(batch, train, stream)


class RecordingModel(TrainableModel):
    """Model which records its outputs from all the run method calls."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.output_data = []
        self.input_data = []
        self.is_train_data = []

    def run(self, batch: Mapping[str, object], train: bool, stream: StreamWrapper):
        outputs = super().run(batch, train, stream)
        self.output_data.append(outputs)
        self.input_data.append(batch)
        self.is_train_data.append(train)
        return outputs


class MainLoopTest(CXTestCaseWithDir):
    """MainLoop test case."""

    def create_main_loop(self,  # pylint: disable=too-many-arguments
                         epochs=1, extra_hooks=(), dataset=None, model_class=None, skip_zeroth_epoch=True,
                         **main_loop_kwargs):
        """
        Create and return a model, dataset and mainloop.

        :param epochs: the number of epochs to be run in the main_loop
        :param extra_hooks: additional hooks to be passed to the main loop
        :param main_loop_kwargs: additional kwargs to be passed to the main loop
        :param dataset: dataset to be passed to the main loop, SimpleDataset() is created if None
        :param model_class: model class to be created and passed to the main loop, TrainableModel if None
        :param skip_zeroth_epoch: skip zeroth epoch flag passed to the main loop
        :return: a tuple of the created model, dataset and mainloop
        """
        hooks = list(extra_hooks) + [StopAfter(epochs=epochs)]
        if dataset is None:
            dataset = SimpleDataset()
        if model_class is None:
            model_class = TrainableModel
        model = model_class(dataset=dataset, log_dir=self.tmpdir,  # pylint: disable=redefined-variable-type
                            io={'in': ['input', 'target'], 'out': ['output']})
        mainloop = cx.MainLoop(model=model, dataset=dataset, hooks=hooks,
                               skip_zeroth_epoch=skip_zeroth_epoch, **main_loop_kwargs)
        return model, dataset, mainloop

    def test_events(self):
        """Test event counts and order."""
        recording_hook = EventRecordingHook()
        _, _, mainloop = self.create_main_loop(epochs=3, extra_hooks=[recording_hook])
        mainloop.run_training()

        before_training = [1]
        first_epoch_batches = list(range(2, 2+_DATASET_ITERS))
        first_epoch = [2+_DATASET_ITERS]
        first_epoch_profile = [3+_DATASET_ITERS]
        second_epoch_batches = list(range(4+_DATASET_ITERS, 4+2*_DATASET_ITERS))
        second_epoch = [4+2*_DATASET_ITERS]
        second_epoch_profile = [5+2*_DATASET_ITERS]
        third_epoch_batches = list(range(6+2*_DATASET_ITERS, 6+3*_DATASET_ITERS))
        third_epoch = [6+3*_DATASET_ITERS]
        # at this point, the training is interrupted by the EpochStopperHook,
        # hence the last after_epoch_profile event is not recorded
        third_epoch_profile = []
        after_training = [7+3*_DATASET_ITERS]

        self.assertListEqual(recording_hook.before_training_events, before_training)
        self.assertListEqual(recording_hook.after_batch_events,
                             first_epoch_batches + second_epoch_batches + third_epoch_batches)
        self.assertListEqual(recording_hook.after_epoch_events, first_epoch + second_epoch + third_epoch)
        self.assertListEqual(recording_hook.after_epoch_profile_events,
                             first_epoch_profile + second_epoch_profile + third_epoch_profile)
        self.assertListEqual(recording_hook.after_training_events, after_training)

    def test_event_data(self):
        """Test after_epoch and after_batch event args match the expectation."""
        recording_hook = DataRecordingHook()
        model, dataset, mainloop = self.create_main_loop(epochs=3, model_class=RecordingModel,
                                                         extra_hooks=[recording_hook], extra_streams=['valid'])
        mainloop.run_training()

        # check the epoch ids
        self.assertListEqual(recording_hook.epoch_ids, [1, 2, 3])
        self.assertListEqual(model.is_train_data,
                             [True]*_DATASET_ITERS+[False]*_DATASET_ITERS+[True]*_DATASET_ITERS +
                             [False]*_DATASET_ITERS+[True]*_DATASET_ITERS+[False]*_DATASET_ITERS)

        # now the model recorded its outputs as a list of all the batches regardless the stream and epoch, i.e.:
        # [train_e1_b1, train_e1_b2, ..., valid_e1_b1, ... train_e2,b1, ...]
        # while the DataRecordingHook has the following structure:
        # {'train': [train_e1_b1, train_e1,b2, ..., train_e2,b1, ...], 'valid': [...]}
        # we will convert the 'model structure' to the 'hook structure' so that they are comparable
        def chunks(list_, size):
            """Split the given list_ into chunks of size consecutive elements."""
            for i in range(0, len(list_), size):
                yield list_[i:i + size]

        output_data = model.output_data  # pylint: disable=no-member
        input_data = model.input_data  # pylint: disable=no-member
        model_outputs_by_stream_list = list(zip(*[(epoch[:len(epoch)//2], epoch[len(epoch)//2:])
                                                  for epoch in chunks(output_data, _DATASET_ITERS*2)]))
        model_inputs_by_stream_list = list(zip(*[(epoch[:len(epoch)//2], epoch[len(epoch)//2:])
                                                 for epoch in chunks(input_data, _DATASET_ITERS*2)]))

        model_outpus_by_stream = {'train': sum(model_outputs_by_stream_list[0], []),
                                  'valid': sum(model_outputs_by_stream_list[1], [])}

        model_inputs_by_stream = {'train': sum(model_inputs_by_stream_list[0], []),
                                  'valid': sum(model_inputs_by_stream_list[1], [])}

        # for all the streams
        for stream_name in ['valid', 'train']:
            self.assertIn(stream_name, recording_hook.batch_data)
            io_data = zip(recording_hook.batch_data[stream_name],
                          model_outpus_by_stream[stream_name],
                          model_inputs_by_stream[stream_name],
                          dataset.batches[stream_name])
            for hook_data, model_outputs, model_inputs, batches in io_data:
                # check if the hook_data and model_inputs contain correct stream sources
                for source_name in dataset.source_names:
                    self.assertIn(source_name, hook_data)
                    self.assertIn(source_name, model_inputs)
                    self.assertTrue(np.alltrue(hook_data[source_name] == batches[source_name]))
                    self.assertTrue(np.alltrue(model_inputs[source_name] == batches[source_name]))
                # check if the hook_data contains correct model outputs
                for output_name in model.output_names:
                    self.assertIn(output_name, hook_data)
                    self.assertTrue(np.alltrue(hook_data[output_name] == model_outputs[output_name]))

    def test_stream_usage(self):
        """Test if the streams are used only when specified."""
        # test if the train stream is used by default
        _, dataset, mainloop = self.create_main_loop()
        mainloop.run_training()
        self.assertTrue(dataset.train_used)
        self.assertFalse(dataset.valid_used)
        self.assertFalse(dataset.test_used)

        # test if the valid stream is used when specified
        _, dataset2, mainloop2 = self.create_main_loop(extra_streams=['valid'])
        mainloop2.run_training()
        self.assertTrue(dataset2.train_used)
        self.assertTrue(dataset2.valid_used)
        self.assertFalse(dataset2.test_used)

        # test an exception is raised when a stream that is not available is specified
        _, _, mainloop3 = self.create_main_loop(extra_streams=['another'])
        self.assertRaises(AttributeError, mainloop3.run_training)

    def test_profiling(self):
        """Test if the mainloop is profiled correctly."""

        # data read profiling
        profile_hook = SaveProfileHook()
        _, _, mainloop = self.create_main_loop(epochs=2, extra_hooks=[profile_hook], dataset=DelayedDataset())
        mainloop.run_training()
        profile = profile_hook.profile

        self.assertIn('read_batch_train', profile)
        self.assertTrue(np.allclose(profile['read_batch_train'], [_READ_DATA_SLEEP_S]*_DATASET_ITERS, atol=0.01))

        # hook profiling
        profile_hook2 = SaveProfileHook()
        _, _, mainloop2 = self.create_main_loop(epochs=2, extra_hooks=[profile_hook2, DelayedHook()])
        mainloop2.run_training()
        profile2 = profile_hook2.profile

        self.assertIn('after_batch_hooks_train', profile2)
        self.assertTrue(np.allclose(profile2['after_batch_hooks_train'],
                                    [_AFTER_BATCH_SLEEP_S]*_DATASET_ITERS, atol=0.01))

        self.assertIn('after_epoch_hooks', profile2)
        self.assertTrue(np.allclose(profile2['after_epoch_hooks'], [_AFTER_EPOCH_SLEEP_S], atol=0.01))

        # model eval profiling
        profile_hook3 = SaveProfileHook()
        _, _, mainloop3 = self.create_main_loop(epochs=2, extra_hooks=[profile_hook3], model_class=DelayedModel)

        mainloop3.run_training()
        profile3 = profile_hook3.profile

        self.assertIn('eval_batch_train', profile3)
        self.assertTrue(np.allclose(profile3['eval_batch_train'], [_MODEL_RUN_SLEEP_S]*_DATASET_ITERS, atol=0.1))

        # multiple streams profiling
        profile_hook4 = SaveProfileHook()
        _, _, mainloop4 = self.create_main_loop(epochs=2, extra_hooks=[profile_hook4], extra_streams=['valid', 'test'])
        mainloop4.run_training()
        profile4 = profile_hook4.profile
        for prefix in ['eval_batch_', 'read_batch_', 'after_batch_hooks_']:
            for stream_name in ['train', 'valid', 'test']:
                self.assertIn(prefix+stream_name, profile4)

    def test_zeroth_epoch(self):
        """Test the model is not trained in the zeroth epoch."""
        data_recording_hook = DataRecordingHook()
        _, _, mainloop = self.create_main_loop(epochs=0, extra_hooks=[data_recording_hook], skip_zeroth_epoch=False)
        mainloop.run_training()

        # check if we actually iterated through the train stream
        self.assertListEqual(data_recording_hook.epoch_ids, [0])
        self.assertIn('train', data_recording_hook.batch_data)
        self.assertEqual(len(data_recording_hook.batch_data['train']), _DATASET_ITERS)

    def test_on_unused_sources(self):
        """Test error is raised when on_unused_inputs='error'."""
        _, _, mainloop = self.create_main_loop(epochs=3, on_unused_sources='error')
        # this should not raise an error
        mainloop.run_training()

        _, _, mainloop2 = self.create_main_loop(epochs=3, dataset=ExtendedDataset(), on_unused_sources='error')
        self.assertRaises(ValueError, mainloop2.run_training)

        _, _, mainloop3 = self.create_main_loop(epochs=3, dataset=ExtendedDataset(), on_unused_sources='ignore')
        mainloop3.run_training()

    def test_epoch_data(self):
        """Test correct epoch_data handling."""

        # test if the epoch data is passed between hooks
        _, _, mainloop = self.create_main_loop(epochs=3, extra_hooks=[EpochDataProducer(), EpochDataConsumer()])
        mainloop.run_training()

        # test wrong hook order
        _, _, mainloop2 = self.create_main_loop(epochs=3, extra_hooks=[EpochDataConsumer(), EpochDataProducer()])
        self.assertRaises(AssertionError, mainloop2.run_training)

    def test_predict(self):
        """Test if predict iterates only the predict stream."""
        _, dataset, mainloop = self.create_main_loop(extra_streams=['valid'])
        mainloop.run_prediction()
        self.assertTrue(dataset.predict_used)
        self.assertFalse(dataset.valid_used)
        self.assertFalse(dataset.train_used)

    def test_buffer(self):
        """Check if buffer speeds up reading data."""
        self.assertGreater(_MODEL_RUN_SLEEP_S, _READ_DATA_SLEEP_S)  # we need to hide read under run

        profile_hook = SaveProfileHook()
        _, _, mainloop = self.create_main_loop(epochs=2, extra_hooks=[profile_hook], model_class=DelayedModel,
                                               dataset=DelayedDataset(), buffer=4)
        mainloop.run_training()
        profile = profile_hook.profile['read_batch_train']
        expected_profile = [_READ_DATA_SLEEP_S + CXF_BUFFER_SLEEP] + [0]*(_DATASET_ITERS-1)

        self.assertTrue(np.allclose(profile, expected_profile, atol=0.01))

    def test_stream_check(self):
        """Test handling of empty batches, streams and checking batch variable lengths."""

        with LogCapture() as log_capture:
            _, _, mainloop = self.create_main_loop(dataset=SomeEmptyBatchDataset(), on_empty_batch='warn')
            mainloop.run_training()

            log_capture.check(
                ('root', 'DEBUG', 'Training started'),
                ('root', 'INFO', 'Training epoch 1'),
                ('root', 'WARNING', '0-th batch in stream `train` appears to be empty (0-th empty batch in '
                                    'total). Set `main_loop.on_empty_batch` to `ignore` in order to suppress '
                                    'this warning.'),
                ('root', 'INFO', 'EpochStopperHook triggered'),
                ('root', 'INFO', 'Training terminated: Training terminated after epoch 1'))

        with LogCapture() as log_capture:
            _, _, mainloop = self.create_main_loop(dataset=EmptyStreamDataset(), on_empty_stream='warn')
            mainloop.run_training()

            self.check = log_capture.check(
                ('root', 'DEBUG', 'Training started'), ('root', 'INFO', 'Training epoch 1'),
                ('root', 'WARNING', 'Stream `train` appears to be empty. Set `main_loop.on_empty_stream` to '
                                    '`ignore` in order to suppress this warning.'),
                 ('root', 'INFO', 'EpochStopperHook triggered'),
                 ('root', 'INFO', 'Training terminated: Training terminated after epoch 1'))

        with self.assertRaises(ValueError):
            _, _, mainloop = self.create_main_loop(dataset=EmptyStreamDataset())
            mainloop.run_training()

        with self.assertRaises(ValueError):
            _, _, mainloop = self.create_main_loop(dataset=AllEmptyBatchDataset(), on_empty_batch='ignore')
            mainloop.run_training()

        with self.assertRaises(ValueError):
            _, _, mainloop = self.create_main_loop(dataset=SomeEmptyBatchDataset())
            mainloop.run_training()

        _, _, mainloop = self.create_main_loop(dataset=AllEmptyBatchDataset(), on_empty_batch='ignore', on_empty_stream='ignore')
        mainloop.run_training()

        with LogCapture() as log_capture:
            _, dataset, mainloop = self.create_main_loop(dataset=ShortSimpleDataset(), fixed_batch_size=47, on_empty_stream='ignore')
            mainloop.run_prediction()

            log_capture.check(
                ('root', 'INFO', 'Running prediction'),
                ('root', 'DEBUG', '0-th batch in stream `predict` has variable `input` of length 11 inconsistent with '
                                  '`main_loop.fixed_size` = 47'),
                ('root', 'INFO', 'Prediction done\n\n'))
