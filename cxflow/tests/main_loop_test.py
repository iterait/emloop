"""
Test module for the main loop (cxflow.main_loop).
"""
import time
from collections import defaultdict
from typing import Mapping, List, Iterable

import numpy as np

from cxflow import AbstractNet, MainLoop, AbstractDataset
from cxflow.hooks.abstract_hook import AbstractHook
from cxflow.hooks.epoch_stopper_hook import EpochStopperHook
from cxflow.utils.profile import Timer

from .test_core import CXTestCaseWithDir

_READ_DATA_SLEEP_S = 0.1
_AFTER_BATCH_SLEEP_S = 0.2
_AFTER_EPOCH_SLEEP_S = 0.3
_NET_RUN_SLEEP_S = 0.5

_DATASET_ITERS = 13
_DATASET_SHAPE = (11, 10)

_EPOCH_DATA_VAR_VALUE = 11


class SimpleDataset(AbstractDataset):
    """Simple dataset with train, valid and test streams."""

    def __init__(self):
        super().__init__(config_str='')
        self.iters = _DATASET_ITERS
        self.shape = _DATASET_SHAPE
        self.train_used = self.valid_used = self.test_used = False
        self.batches = defaultdict(lambda: [])
        self.source_names = ['input', 'target']
        self._iter = 1

    def stream(self, stream_name: str)-> AbstractDataset.Stream:
        """Generate a datastream with increasing inputs and constant target."""
        for _ in range(self.iters):
            batch = {'input': self._iter * np.ones(self.shape), 'target': np.zeros(self.shape)}
            self.batches[stream_name].append(batch)
            self._iter += 1
            yield batch

    def create_train_stream(self) -> AbstractDataset.Stream:
        self.train_used = True
        for batch in self.stream('train'):
            yield batch

    def create_valid_stream(self) -> AbstractDataset.Stream:
        self.valid_used = True
        for batch in self.stream('valid'):
            yield batch

    def create_test_stream(self) -> AbstractDataset.Stream:
        self.test_used = True
        for batch in self.stream('test'):
            yield batch


class ExtendedDataset(SimpleDataset):
    """SimpleDataset extension with additional 'unused' source in the train stream."""

    def create_train_stream(self) -> AbstractDataset.Stream:
        self.train_used = True
        for _ in range(self.iters):
            yield {'input': np.ones(self.shape), 'target': np.zeros(self.shape), 'unused': np.zeros(self.shape)}


class DelayedDataset(SimpleDataset):
    """SimpleDataset extension which sleeps briefly before each train batch allowing to measure the data read time."""

    def create_train_stream(self) -> AbstractDataset.Stream:
        for _ in range(self.iters):
            time.sleep(_READ_DATA_SLEEP_S)
            yield {'input': np.ones(self.shape), 'target': np.zeros(self.shape)}


class EventRecordingHook(AbstractHook):
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

    def after_batch(self, stream_name: str, batch_data: AbstractDataset.Batch) -> None:
        self.after_batch_events.append(self._event_id)
        self._event_id += 1

    def after_epoch(self, epoch_id: int, epoch_data: AbstractHook.EpochData) -> None:
        self.after_epoch_events.append(self._event_id)
        self._event_id += 1

    def after_epoch_profile(self, epoch_id: int, profile: Timer.TimeProfile, extra_streams: Iterable[str]) -> None:
        self.after_epoch_profile_events.append(self._event_id)
        self._event_id += 1

    def after_training(self) -> None:
        self.after_training_events.append(self._event_id)
        self._event_id += 1


class DataRecordingHook(AbstractHook):
    """DataRecordingHook records epoch_ids and all the batch_data."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.epoch_ids = []
        self.batch_data = defaultdict(lambda: [])

    def after_batch(self, stream_name: str, batch_data: AbstractDataset.Batch) -> None:
        self.batch_data[stream_name].append(batch_data)

    def after_epoch(self, epoch_id: int, epoch_data: AbstractHook.EpochData) -> None:
        self.epoch_ids.append(epoch_id)


class DelayedHook(AbstractHook):
    """DelayedHook sleeps briefly in after_batch and after_epoch events allowing to measure hook processing times."""

    def after_batch(self, stream_name: str, batch_data: AbstractDataset.Batch) -> None:
        time.sleep(_AFTER_BATCH_SLEEP_S)

    def after_epoch(self, epoch_id: int, epoch_data: AbstractHook.EpochData) -> None:
        time.sleep(_AFTER_EPOCH_SLEEP_S)


class SaveProfileHook(AbstractHook):
    """SaveProfileHook saves the epoch profile dict to self.profile."""

    def __init__(self):
        super().__init__()
        self.profile = None

    def after_epoch_profile(self, epoch_id: int, profile: Timer.TimeProfile, extra_streams: Iterable[str]) -> None:
        """Save the profile to self.profile."""
        self.profile = profile


class EpochDataProducer(AbstractHook):
    """Simple hook that adds my_variable to the train entry in the epoch_data."""

    def after_epoch(self, epoch_id: int, epoch_data: AbstractHook.EpochData) -> None:
        epoch_data['train']['my_variable'] = _EPOCH_DATA_VAR_VALUE


class EpochDataConsumer(AbstractHook):
    """Simple hook that asserts presence of my_variable in the train entry of the epoch_data."""

    def after_epoch(self, epoch_id: int, epoch_data: AbstractHook.EpochData) -> None:
        assert 'train' in epoch_data
        assert 'my_variable' in epoch_data['train']
        assert epoch_data['train']['my_variable'] == _EPOCH_DATA_VAR_VALUE


class TrainableNet(AbstractNet):
    """Simple trainable net"""
    def __init__(self, io: dict, **kwargs):  #pylint: disable=unused-argument
        self._input_names = io['in']
        self._output_names = io['out']

    def run(self, batch: Mapping[str, object], train: bool) -> Mapping[str, object]:
        return {o: i for i, o in enumerate(self._output_names)}

    def save(self, name_suffix: str) -> str:
        pass

    @property
    def input_names(self) -> List[str]:   # pylint: disable=invalid-sequence-index
        """List of tf tensor names listed as net inputs."""
        return self._input_names

    @property
    def output_names(self) -> List[str]:   # pylint: disable=invalid-sequence-index
        """List of tf tensor names listed as net outputs."""
        return self._output_names


class DelayedNet(TrainableNet):
    """Trainable net which sleeps briefly when processing a batch allowing to measure the net eval time."""

    def run(self, batch: Mapping[str, object], train: bool):
        time.sleep(_NET_RUN_SLEEP_S)
        return super().run(batch, train)


class RecordingNet(TrainableNet):
    """Net which records its outputs from all the run method calls."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.output_data = []
        self.input_data = []

    def run(self, batch: Mapping[str, object], train: bool):
        outputs = super().run(batch, train)
        self.output_data.append(outputs)
        self.input_data.append(batch)
        return outputs


class MainLoopTest(CXTestCaseWithDir):
    """MainLoop test case."""

    def create_main_loop(self,  # pylint: disable=too-many-arguments
                         epochs=1, extra_hooks=(), dataset=None, net_class=None, skip_zeroth_epoch=True,
                         **main_loop_kwargs):
        """
        Create and return a net, dataset and mainloop.

        :param epochs: the number of epochs to be run in the main_loop
        :param extra_hooks: additional hooks to be passed to the main loop
        :param main_loop_kwargs: additional kwargs to be passed to the main loop
        :param dataset: dataset to be passed to the main loop, SimpleDataset() is created if None
        :param net_class: net class to be created and passed to the main loop, TrainableNet if None
        :param skip_zeroth_epoch: skip zeroth epoch flag passed to the main loop
        :return: a tuple of the created net, dataset and mainloop
        """
        hooks = list(extra_hooks) + [EpochStopperHook(epoch_limit=epochs)]
        if dataset is None:
            dataset = SimpleDataset()
        if net_class is None:
            net_class = TrainableNet
        net = net_class(dataset=dataset, log_dir=self.tmpdir,  # pylint: disable=redefined-variable-type
                        io={'in': ['input', 'target'], 'out': ['output']})
        mainloop = MainLoop(net=net, dataset=dataset, hooks=hooks,
                            skip_zeroth_epoch=skip_zeroth_epoch, **main_loop_kwargs)
        return net, dataset, mainloop

    def test_events(self):
        """Test event counts and order."""
        recording_hook = EventRecordingHook()
        _, _, mainloop = self.create_main_loop(epochs=3, extra_hooks=[recording_hook])
        mainloop.run()

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
        net, dataset, mainloop = self.create_main_loop(epochs=3, net_class=RecordingNet,
                                                       extra_hooks=[recording_hook], extra_streams=['valid'])
        mainloop.run()

        # check the epoch ids
        self.assertListEqual(recording_hook.epoch_ids, [1, 2, 3])

        # now the net recorded its outputs as a list of all the batches regardless the stream and epoch, i.e.:
        # [train_e1_b1, train_e1_b2, ..., valid_e1_b1, ... train_e2,b1, ...]
        # while the DataRecordingHook has the following structure:
        # {'train': [train_e1_b1, train_e1,b2, ..., train_e2,b1, ...], 'valid': [...]}
        # we will convert the 'net structure' to the 'hook structure' so that they are comparable
        def chunks(list_, size):
            """Split the given list_ into chunks of size consecutive elements."""
            for i in range(0, len(list_), size):
                yield list_[i:i + size]

        output_data = net.output_data  # pylint: disable=no-member
        input_data = net.input_data  # pylint: disable=no-member
        net_outputs_by_stream_list = list(zip(*[(epoch[:len(epoch)//2], epoch[len(epoch)//2:])
                                                for epoch in chunks(output_data, _DATASET_ITERS*2)]))
        net_inputs_by_stream_list = list(zip(*[(epoch[:len(epoch)//2], epoch[len(epoch)//2:])
                                               for epoch in chunks(input_data, _DATASET_ITERS*2)]))

        net_outpus_by_stream = {'train': sum(net_outputs_by_stream_list[0], []),
                                'valid': sum(net_outputs_by_stream_list[1], [])}

        net_inputs_by_stream = {'train': sum(net_inputs_by_stream_list[0], []),
                                'valid': sum(net_inputs_by_stream_list[1], [])}

        # for all the streams
        for stream_name in ['valid', 'train']:
            self.assertIn(stream_name, recording_hook.batch_data)
            io_data = zip(recording_hook.batch_data[stream_name],
                          net_outpus_by_stream[stream_name],
                          net_inputs_by_stream[stream_name],
                          dataset.batches[stream_name])
            for hook_data, net_outputs, net_inputs, batches in io_data:
                # check if the hook_data and net_inputs contain correct stream sources
                for source_name in dataset.source_names:
                    self.assertIn(source_name, hook_data)
                    self.assertIn(source_name, net_inputs)
                    self.assertTrue(np.alltrue(hook_data[source_name] == batches[source_name]))
                    self.assertTrue(np.alltrue(net_inputs[source_name] == batches[source_name]))
                # check if the hook_data contains correct net outputs
                for output_name in net.output_names:
                    self.assertIn(output_name, hook_data)
                    self.assertTrue(np.alltrue(hook_data[output_name] == net_outputs[output_name]))

    def test_stream_usage(self):
        """Test if the streams are used only when specified."""
        # test if the train stream is used by default
        _, dataset, mainloop = self.create_main_loop()
        mainloop.run()
        self.assertTrue(dataset.train_used)
        self.assertFalse(dataset.valid_used)
        self.assertFalse(dataset.test_used)

        # test if the valid stream is used when specified
        _, dataset2, mainloop2 = self.create_main_loop(extra_streams=['valid'])
        mainloop2.run()
        self.assertTrue(dataset2.train_used)
        self.assertTrue(dataset2.valid_used)
        self.assertFalse(dataset2.test_used)

        # test an exception is raised when a stream that is not available is specified
        _, _, mainloop3 = self.create_main_loop(extra_streams=['another'])
        self.assertRaises(AttributeError, mainloop3.run)

    def test_profiling(self):
        """Test if the mainloop is profiled correctly."""

        # data read profiling
        profile_hook = SaveProfileHook()
        _, _, mainloop = self.create_main_loop(epochs=2, extra_hooks=[profile_hook], dataset=DelayedDataset())
        mainloop.run()
        profile = profile_hook.profile

        self.assertIn('read_batch_train', profile)
        self.assertTrue(np.allclose(profile['read_batch_train'], [_READ_DATA_SLEEP_S]*_DATASET_ITERS, atol=0.01))

        # hook profiling
        profile_hook2 = SaveProfileHook()
        _, _, mainloop2 = self.create_main_loop(epochs=2, extra_hooks=[profile_hook2, DelayedHook()])
        mainloop2.run()
        profile2 = profile_hook2.profile

        self.assertIn('after_batch_hooks_train', profile2)
        self.assertTrue(np.allclose(profile2['after_batch_hooks_train'],
                                    [_AFTER_BATCH_SLEEP_S]*_DATASET_ITERS, atol=0.01))

        self.assertIn('after_epoch_hooks', profile2)
        self.assertTrue(np.allclose(profile2['after_epoch_hooks'], [_AFTER_EPOCH_SLEEP_S], atol=0.01))

        # net eval profiling
        profile_hook3 = SaveProfileHook()
        _, _, mainloop3 = self.create_main_loop(epochs=2, extra_hooks=[profile_hook3], net_class=DelayedNet)

        mainloop3.run()
        profile3 = profile_hook3.profile

        self.assertIn('eval_batch_train', profile3)
        self.assertTrue(np.allclose(profile3['eval_batch_train'], [_NET_RUN_SLEEP_S]*_DATASET_ITERS, atol=0.1))

        # multiple streams profiling
        profile_hook4 = SaveProfileHook()
        _, _, mainloop4 = self.create_main_loop(epochs=2, extra_hooks=[profile_hook4], extra_streams=['valid', 'test'])
        mainloop4.run()
        profile4 = profile_hook4.profile
        for prefix in ['eval_batch_', 'read_batch_', 'after_batch_hooks_']:
            for stream_name in ['train', 'valid', 'test']:
                self.assertIn(prefix+stream_name, profile4)

    def test_zeroth_epoch(self):
        """Test the net is not trained in the zeroth epoch."""
        data_recording_hook = DataRecordingHook()
        _, _, mainloop = self.create_main_loop(epochs=0, extra_hooks=[data_recording_hook], skip_zeroth_epoch=False)
        mainloop.run()

        # check if we actually iterated through the train stream
        self.assertListEqual(data_recording_hook.epoch_ids, [0])
        self.assertIn('train', data_recording_hook.batch_data)
        self.assertEqual(len(data_recording_hook.batch_data['train']), _DATASET_ITERS)

    def test_on_unused_sources(self):
        """Test error is raised when on_unused_inputs='error'."""
        _, _, mainloop = self.create_main_loop(epochs=3, on_unused_sources='error')
        # this should not raise an error
        mainloop.run()

        _, _, mainloop2 = self.create_main_loop(epochs=3, dataset=ExtendedDataset(), on_unused_sources='error')
        self.assertRaises(ValueError, mainloop2.run)

        _, _, mainloop3 = self.create_main_loop(epochs=3, dataset=ExtendedDataset(), on_unused_sources='ignore')
        mainloop3.run()

    def test_epoch_data(self):
        """Test correct epoch_data handling."""

        # test if the epoch data is passed between hooks
        _, _, mainloop = self.create_main_loop(epochs=3, extra_hooks=[EpochDataProducer(), EpochDataConsumer()])
        mainloop.run()

        # test wrong hook order
        _, _, mainloop2 = self.create_main_loop(epochs=3, extra_hooks=[EpochDataConsumer(), EpochDataProducer()])
        self.assertRaises(AssertionError, mainloop2.run)
