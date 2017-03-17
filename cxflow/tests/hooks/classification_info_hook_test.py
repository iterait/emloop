from cxflow.hooks.classification_info_hook import ClassificationInfoHook

import pandas as pd

import logging
from os import path
from unittest import TestCase
import tempfile


class NetMocker:
    def __init__(self):
        self.log_dir = tempfile.mkdtemp(prefix='csvhooktest', dir=tempfile.gettempdir())


class ClassificationInfoHookTest(TestCase):
    def __init__(self, *args, **kwargs):
        logging.getLogger().disabled = True
        super().__init__(*args, **kwargs)

    def test_classification_info(self):
        hook = ClassificationInfoHook(predicted_variable='predicted', gold_variable='gold', f1_average='macro',
                                      net=None, config=None)

        hook.after_batch('train', {'predicted': [0,0,0,1,2], 'gold': [0,0,0,0,2]})
        hook.after_batch('train', {'predicted': [0,0,1,1,2], 'gold': [0,1,2,0,1]})

        hook.after_batch('valid', {'predicted': [1,1,0,1,2], 'gold': [1,0,0,0,2]})
        hook.after_batch('valid', {'predicted': [0,2,1,1,2], 'gold': [0,1,1,0,1]})

        hook.after_batch('test', {'predicted': [2,2,0,1,2], 'gold': [0,2,0,0,2]})
        hook.after_batch('test', {'predicted': [0,2,1,1,2], 'gold': [2,2,2,0,1]})

        self.assertListEqual([0,0,0,1,2,0,0,1,1,2], hook._train_predicted)
        self.assertListEqual([0,0,0,0,2,0,1,2,0,1], hook._train_gold)

        self.assertListEqual([1,1,0,1,2,0,2,1,1,2], hook._valid_predicted)
        self.assertListEqual([1,0,0,0,2,0,1,1,0,1], hook._valid_gold)

        self.assertListEqual([2,2,0,1,2,0,2,1,1,2], hook._test_predicted)
        self.assertListEqual([0,2,0,0,2,2,2,2,0,1], hook._test_gold)

        train_res = {'a': 'b'}
        valid_res = {'c': 'd'}
        test_res = {'e': 'f'}

        hook.after_epoch(1, train_res, valid_res, test_res)

        self.assertIn('a', train_res)
        self.assertIn('fscore', train_res)
        self.assertIn('recall', train_res)
        self.assertIn('precision', train_res)

        self.assertIn('c', valid_res)
        self.assertIn('fscore', valid_res)
        self.assertIn('recall', valid_res)
        self.assertIn('precision', valid_res)

        self.assertIn('e', test_res)
        self.assertIn('fscore', test_res)
        self.assertIn('recall', test_res)
        self.assertIn('precision', test_res)

        self.assertListEqual([], hook._train_predicted)
        self.assertListEqual([], hook._train_gold)

        self.assertListEqual([], hook._valid_predicted)
        self.assertListEqual([], hook._valid_gold)

        self.assertListEqual([], hook._test_predicted)
        self.assertListEqual([], hook._test_gold)
