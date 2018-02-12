import os
import os.path as path

import numpy as np
import cxflow as cx
from cxflow.constants import CXF_CONFIG_FILE
from cxflow.datasets import StreamWrapper
from cxflow.models.ensemble_model import major_vote, EnsembleModel

from ..test_core import CXTestCase, CXTestCaseWithDir


_DUMMY_CONFIG = """
model:
  name: test
  class: cxflow.tests.models.ensemble_model_test.DummyModel
  inputs: [images, extra_input]

"""


class DummyModel:
    """Dummy model multiplying the input with the last character of the ``restore_from`` path."""

    def __init__(self,
                 inputs: str,
                 outputs: str,
                 dataset: cx.AbstractDataset,
                 restore_from: str,
                 **_):

        self.dataset = dataset
        self.inputs = inputs
        self.outputs = outputs
        self.restore_from = restore_from

    def run(self, batch: cx.Batch, train: bool, stream: StreamWrapper) -> cx.Batch:
        return {'outputs': list(np.array(batch['inputs'])*int(self.restore_from[-1]))}


class MajorVoteTest(CXTestCase):
    """major_vote function test case."""

    def test_major_vote(self):
        """Test if major_vote works properly."""
        vote1 = (1, 3, (5, 5), 12)
        vote2 = [1, 2, 2, 2]
        vote3 = [1, 2, (5, 5), 1]

        result = major_vote([vote1, vote2, vote3])
        self.assertListEqual([1, 2, (5, 5)], list(result)[:3])
        self.assertIn(result[-1], {1, 2, 12})


class EnsembleModelTest(CXTestCaseWithDir):
    """EnsembleModel function test case."""

    def _create_models(self):
        """Create three dummy models in the tmp dir."""
        for i in [1, 2, 3]:
            model_dir = path.join(self.tmpdir, str(i))
            os.mkdir(model_dir)
            with open(path.join(model_dir, CXF_CONFIG_FILE), 'w') as config:
                config.write(_DUMMY_CONFIG)

    def test_init(self):
        """Test if EnsembleModel ``__init__`` works properly"""
        with self.assertRaises(AssertionError):
            EnsembleModel(inputs=['inputs'], outputs=['outputs'])
        with self.assertRaises(AssertionError):
            EnsembleModel(inputs=['inputs'], outputs=['outputs'], models_root=self.tmpdir, aggregation='unknown')

        ensemble = EnsembleModel(inputs=['inputs'], outputs=['outputs'], models_root=self.tmpdir, aggregation='mean')
        self.assertIsNone(ensemble._models)

        # test eager loading
        ensemble2 = EnsembleModel(inputs=['inputs'], outputs=['outputs'], models_root=self.tmpdir, eager_loading=True)
        self.assertIsNotNone(ensemble2._models)

        self.assertListEqual(ensemble2.input_names, ['inputs'])
        self.assertListEqual(ensemble2.output_names, ['outputs'])

    def test_propagation(self):
        """Test if EnsembleModel propagates all the arguments properly."""
        self._create_models()
        ensemble = EnsembleModel(inputs=['inputs'], outputs=['outputs'], models_root=self.tmpdir, aggregation='mean',
                                 dataset='my_dataset')

        # test restore_from is propagated, test mean aggregation
        batch = {'inputs': [1., 1., 1.]}
        output = ensemble.run(batch, False, None)
        self.assertListEqual([2., 2., 2.], list(output['outputs']))

        # test dataset is propagated
        self.assertEqual('my_dataset', ensemble._models[0].dataset)

        # test inputs parameter is propagated
        self.assertNotIn('extra_input', batch)

    def test_major_vote(self):
        """Test if Ensemble model aggregates the results properly."""
        self._create_models()
        ensemble = EnsembleModel(inputs=['inputs'], outputs=['outputs'], models_root=self.tmpdir,
                                 model_paths=['1', '1', '2'])

        # test major vote aggregation
        batch = {'inputs': [1., 1., 1.]}
        output = ensemble.run(batch, False, None)
        self.assertListEqual([1., 1., 1.], list(output['outputs']))

        # test multi-dim data
        batch = {'inputs': [[1., 1., 2.], [1., 1., 2.]]}
        output = ensemble.run(batch, False, None)
        self.assertTrue(np.equal([[1., 1., 2.], [1., 1., 2.]], output['outputs']).all())

    def test_raising(self):
        """Test if Ensemble model raises the exceptions as expected."""
        self._create_models()
        ensemble = EnsembleModel(inputs=['inputs'], outputs=['outputs'], models_root=self.tmpdir,
                                 model_paths=['1', '1', '2'])

        with self.assertRaises(ValueError):
            ensemble.run(None, True, None)
        with self.assertRaises(NotImplementedError):
            ensemble.save()
        self.assertIsNone(ensemble.restore_fallback)
