from cxflow import AbstractDataset
from cxflow.datasets.stream_wrapper import StreamWrapper

from ..test_core import CXTestCase
from ..main_loop_test import SimpleDataset, _DATASET_ITERS


class FailingDataset(SimpleDataset):
    """SimpleDataset extension which raises exception after first train batch."""

    def train_stream(self) -> AbstractDataset.Stream:
        for batch in super().train_stream():
            yield batch
            raise RuntimeError('Explosion.')


class StreamWrapperTest(CXTestCase):
    """StreamWrapper test case."""

    def test_iteration(self):
        """Test both buffered and non-buffered iterations."""
        dataset = SimpleDataset()
        stream = StreamWrapper(dataset.train_stream)

        epoch = list(stream)
        self.assertEqual(len(epoch), _DATASET_ITERS)  # 1st epoch
        self.assertListEqual(epoch, dataset.batches['train'])
        epoch2 = []
        for batch in stream:  # 2nd epoch
            epoch2.append(batch)
        self.assertEqual(len(epoch2), _DATASET_ITERS)
        self.assertListEqual(epoch + epoch2, dataset.batches['train'])

        dataset2 = SimpleDataset()
        buferred_stream = StreamWrapper(dataset2.train_stream, buffer_size=4)
        self.assertRaises(ValueError, next, buferred_stream)  # used outside with-resource

        with buferred_stream:
            buffered_epoch = list(buferred_stream)
        with buferred_stream:
            buffered_epoch2 = []
            for batch in buferred_stream:
                buffered_epoch2.append(batch)

        self.assertListEqual(buffered_epoch + buffered_epoch2, dataset2.batches['train'])

    def test_epoch_size(self):
        """Test both buffered and non-buffered iterations with fixed epoch size."""
        epoch_size = 10
        dataset = SimpleDataset()
        with StreamWrapper(dataset.train_stream, epoch_size=epoch_size) as stream:
            epoch = list(stream)
            self.assertEqual(len(epoch), epoch_size)
            epoch2 = list(stream)
            self.assertListEqual(epoch+epoch2, dataset.batches['train'])

        dataset2 = SimpleDataset()
        with StreamWrapper(dataset2.train_stream, buffer_size=4, epoch_size=epoch_size) as stream2:
            buffered_epoch = list(stream2)
            self.assertEqual(len(buffered_epoch), epoch_size)
            buffered_epoch2 = list(stream2)

        self.assertListEqual(buffered_epoch + buffered_epoch2, dataset2.batches['train'])

    def test_breaking(self):
        """Test buffered stream works correctly with stream cut into multiple with-resource directives."""
        dataset = SimpleDataset()
        buffered_stream = StreamWrapper(dataset.train_stream, buffer_size=4)
        epoch = []
        for i in range(5):
            with buffered_stream:
                epoch.append(next(buffered_stream))
        epoch += list(buffered_stream)
        self.assertListEqual(epoch, dataset.batches['train'])

    def test_enqueueing_thread_exception(self):
        dataset = FailingDataset()
        buffered_stream = StreamWrapper(dataset.train_stream, buffer_size=4)
        with buffered_stream:
            self.assertRaises(ChildProcessError, list, buffered_stream)

    def test_empty_stream(self):
        stream = StreamWrapper(list)
        self.assertListEqual(list(stream), [])
        with StreamWrapper(list, buffer_size=4) as buffered_stream:
            self.assertListEqual(list(buffered_stream), [])

    def test_simple_iterator(self):
        stream = StreamWrapper(lambda: [1, 2])
        next(stream)
        self.assertEqual(next(stream), 2)
        self.assertRaises(StopIteration, next, stream)
