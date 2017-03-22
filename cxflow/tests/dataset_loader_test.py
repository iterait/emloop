from cxflow.dataset_loader import DatasetLoader

import logging
from unittest import TestCase


class SampleDataset:
    def __init__(self, **kwargs):
        self.set_on_complete = True

    def create_train_stream(self):
        pass

    def create_valid_stream(self):
        pass


class SampleDatasetTest:
    def __init__(self, **kwargs):
        self.set_on_complete = True

    def create_train_stream(self):
        pass

    def create_valid_stream(self):
        pass

    def create_test_stream(self):
        pass


class SampleDatasetNoTrain:
    def __init__(self, **kwargs):
        self.set_on_complete = True

    def create_valid_stream(self):
        pass


class SampleDatasetNoValid:
    def __init__(self, **kwargs):
        self.set_on_complete = True

    def create_train_stream(self):
        pass


class SampleDatasetNoTest:
    def __init__(self, **kwargs):
        self.set_on_complete = True

    def create_train_stream(self):
        pass

    def create_valid_stream(self):
        pass


class DatasetLoaderTest(TestCase):
    def __init__(self, *args, **kwargs):
        logging.getLogger().disabled = True
        super().__init__(*args, **kwargs)

    def test_config_assert(self):
        self.assertRaises(KeyError, DatasetLoader, {'dataset': {'dataset_module': '', 'dataset_class': ''}}, '')
        self.assertRaises(KeyError, DatasetLoader, {'dataset': {'dataset_module': '', 'backend': ''}}, '')
        self.assertRaises(KeyError, DatasetLoader, {'dataset': {'backend': '', 'dataset_class': ''}}, '')

    def test_module_load(self):
        config = {'dataset': {
            'dataset_module': 'cxflow.tests.dataset_loader_test',
            'dataset_class': 'SampleDataset',
            'backend': 'fuel'
        }}

        try:
            DatasetLoader(config, '')
        except:
            self.fail()

        config = {'dataset': {
            'dataset_module': 'cxflow.tests.dataset_loader_test_XXX',
            'dataset_class': 'SampleDataset',
            'backend': 'fuel'
        }}

        self.assertRaises(ImportError, DatasetLoader, config, '')

    def test_class_load(self):
        config = {'dataset': {
            'dataset_module': 'cxflow.tests.dataset_loader_test',
            'dataset_class': 'SampleDataseXXXXX',
            'backend': 'fuel'
        }}

        self.assertRaises(AttributeError, DatasetLoader, config, '')
