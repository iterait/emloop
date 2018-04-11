import os
import matplotlib

from cxflow.tests.test_core import CXTestCaseWithDir
from cxflow.hooks.save_cm import SaveConfusionMatrix
from ..main_loop_test import SimpleDataset


class TestDataset(SimpleDataset):

    @staticmethod
    def num_classes():
        return 4

    @staticmethod
    def num_classes_bad():
        return 1  # too small number

    @staticmethod
    def classes_names():
        return ['a', 'b', 'c', 'd']

    @staticmethod
    def classes_names_bad():
        return ['a']  # too few names


class SaveConfusionMatrixTest(CXTestCaseWithDir):
    """Test case for :py:class:`cxflow.hooks.SaveConfusionMatrix hook."""

    @staticmethod
    def run_hook(hook,
                 batch_data: dict={'labels': [0, 1], 'predictions': [0, 1]},
                 epoch_data: dict={'train': {'accuracy': 1}}):
        """
        Run hook's methods `after_batch` and `after_epoch`
        Returns modified epoch_data
        """
        hook.after_batch(stream_name='train', batch_data=batch_data)
        hook.after_epoch(epoch_id=0, epoch_data=epoch_data)
        return epoch_data

    def test_after_epoch(self):

        # test wrong input parameters
        with self.assertRaises(KeyError):
            hook = SaveConfusionMatrix(dataset=TestDataset(), output_dir=self.tmpdir,
                                       labels_name='fake')
            SaveConfusionMatrixTest.run_hook(hook)
        with self.assertRaises(KeyError):
            hook = SaveConfusionMatrix(dataset=TestDataset(), output_dir=self.tmpdir,
                                       predictions_name='fake')
            SaveConfusionMatrixTest.run_hook(hook)
        with self.assertRaises(ValueError):
            hook = SaveConfusionMatrix(dataset=TestDataset(), output_dir=self.tmpdir,
                                       cmap='fake')
            SaveConfusionMatrixTest.run_hook(hook)
        with self.assertRaises(ValueError):
            hook = SaveConfusionMatrix(dataset=TestDataset(), output_dir=self.tmpdir,
                                       figure_action='non_existing')
        # test wrong number of classes' names
        with self.assertRaises(AssertionError):
            hook = SaveConfusionMatrix(dataset=TestDataset(), output_dir=self.tmpdir,
                                       classes_names=['just_one'])
            SaveConfusionMatrixTest.run_hook(hook)
        with self.assertRaises(AssertionError):
            hook = SaveConfusionMatrix(dataset=TestDataset(), output_dir=self.tmpdir,
                                       classes_names_method_name='classes_names_bad')
            SaveConfusionMatrixTest.run_hook(hook)
        with self.assertRaises(AssertionError):
            hook = SaveConfusionMatrix(dataset=TestDataset(), output_dir=self.tmpdir,
                                       num_classes_method_name='num_classes_bad',
                                       classes_names_method_name='not_existing')
            SaveConfusionMatrixTest.run_hook(hook)
        # test saving .png
        hook = SaveConfusionMatrix(dataset=TestDataset(), output_dir=self.tmpdir)
        SaveConfusionMatrixTest.run_hook(hook)
        self.assertTrue(os.path.exists(os.path.join(self.tmpdir, 'confusion_matrix_epoch_0_train.png')))
        # test storing .png
        hook = SaveConfusionMatrix(dataset=TestDataset(), output_dir='',
                                   figure_action='store')
        epoch_data = SaveConfusionMatrixTest.run_hook(hook)
        self.assertTupleEqual(tuple(epoch_data['train']['confusion_heatmap'].shape), (480, 640, 3))

        # test changing figure size
        hook = SaveConfusionMatrix(dataset=TestDataset(), output_dir='', figure_action='store',
                                   figsize=(10, 15))
        epoch_data = SaveConfusionMatrixTest.run_hook(hook)
        dpi = matplotlib.rcParams['figure.dpi']
        self.assertTupleEqual(tuple(epoch_data['train']['confusion_heatmap'].shape), (15*dpi, 10*dpi, 3))

        # test hook is working if each argument is OK
        hook = SaveConfusionMatrix(dataset=TestDataset(), output_dir=self.tmpdir,
                                   classes_names=['first', 'second'])
        SaveConfusionMatrixTest.run_hook(hook)

        hook = SaveConfusionMatrix(dataset=TestDataset(), output_dir=self.tmpdir,
                                   classes_names_method_name='not_existing')
        SaveConfusionMatrixTest.run_hook(hook)

        hook = SaveConfusionMatrix(dataset=TestDataset(), output_dir=self.tmpdir,
                                   classes_names_method_name='not_existing',
                                   num_classes_method_name='not_existing')
        SaveConfusionMatrixTest.run_hook(hook)

        hook = SaveConfusionMatrix(dataset=TestDataset(), output_dir=self.tmpdir,
                                   cmap='Greens')
        SaveConfusionMatrixTest.run_hook(hook)

        hook = SaveConfusionMatrix(dataset=TestDataset(), output_dir=self.tmpdir,
                                   normalize=False)
        SaveConfusionMatrixTest.run_hook(hook)

        hook = SaveConfusionMatrix(dataset=TestDataset(), output_dir=self.tmpdir,
                                   labels_name='special_labels',
                                   predictions_name='special_predictions')
        SaveConfusionMatrixTest.run_hook(hook, batch_data={'special_labels': [0, 1],
                                                           'special_predictions': [0, 1]})
