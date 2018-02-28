import os
import os.path as path

import numpy as np
import cxflow as cx
from cxflow.constants import CXF_CONFIG_FILE
from cxflow.datasets import StreamWrapper
from cxflow.models.sequence import Sequence

from ..test_core import CXTestCaseWithDir


_STEP1_CONFIG = """
model:
  name: step1
  class: cxflow.tests.models.sequence_test.Step1
  inputs: [images]
  outputs: [masks]

"""


_STEP2_CONFIG = """
model:
  name: step2
  class: cxflow.tests.models.sequence_test.Step2
  inputs: [images, masks]
  outputs: [classes]

"""


class Step1:
    def __init__(self, dataset, **_):
        self.dataset = dataset

    @property
    def input_names(self):
        return ['images']

    @property
    def output_names(self):
        return ['masks']

    def run(self, batch: cx.Batch, train: bool, stream: StreamWrapper) -> cx.Batch:
        return {'masks': np.zeros_like(batch['images'])}


class Step2:
    def __init__(self, **_):
        pass

    @property
    def input_names(self):
        return ['images', 'masks']

    @property
    def output_names(self):
        return ['classes']

    def run(self, batch: cx.Batch, train: bool, stream: StreamWrapper) -> cx.Batch:
        assert 'images' in batch
        assert 'masks' in batch
        return {'classes': np.ones(np.array(batch['images']).shape[0])}


class SequenceTest(CXTestCaseWithDir):
    """Sequence model function test case."""

    def _create_models(self):
        """Create two step models in the tmp dir."""
        for name, config in zip(['step1', 'step2'], [_STEP1_CONFIG, _STEP2_CONFIG]):
            model_dir = path.join(self.tmpdir, name)
            os.mkdir(model_dir)
            with open(path.join(model_dir, CXF_CONFIG_FILE), 'w') as config_file:
                config_file.write(config)

    def test_init(self):
        """Test if Sequence model ``__init__`` works properly"""
        self._create_models()
        sequence = Sequence(models_root=self.tmpdir, model_paths=['step1', 'step2'])
        self.assertIsNone(sequence._models)

        # test eager loading
        sequence2 = Sequence(models_root=self.tmpdir, model_paths=['step1', 'step2'], eager_loading=True)
        self.assertIsNotNone(sequence2._models)

        self.assertListEqual(sequence2.input_names, ['images'])
        self.assertListEqual(list(sequence2.output_names), ['masks', 'classes'])

    def test_run(self):
        """Test if Sequence model accumulates the outputs properly."""
        self._create_models()
        sequence = Sequence(models_root=self.tmpdir, model_paths=['step1', 'step2'], dataset='my_dataset')

        # outputs accumulating
        images = [[2.], [2.], [2.]]
        output = sequence.run({'images': images}, False, None)
        self.assertIn('masks', output)
        self.assertIn('classes', output)
        np.testing.assert_array_equal(np.zeros_like(images), output['masks'])
        np.testing.assert_array_equal(np.ones((3,)), output['classes'])

        # test dataset is propagated
        self.assertEqual('my_dataset', sequence._models[0].dataset)

    def test_raising(self):
        """Test if Sequence model raises the exceptions as expected."""
        self._create_models()
        sequence = Sequence(models_root=self.tmpdir, model_paths=['step1', 'step2'])

        with self.assertRaises(ValueError):
            sequence.run(None, True, None)
        with self.assertRaises(NotImplementedError):
            sequence.save()
        self.assertIsNone(sequence.restore_fallback)
