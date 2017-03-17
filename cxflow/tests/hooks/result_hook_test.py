from cxflow.hooks.result_hook import ResultHook

import numpy as np

import logging
from unittest import TestCase


class ResultHookTest(TestCase):
    def __init__(self, *args, **kwargs):
        logging.getLogger().disabled = True
        super().__init__(*args, **kwargs)

    def test_result_with_train(self):
        hook = ResultHook(metrics_to_log=['a', 'b'], log_train_results=True, net=None, config=None)

        hook.after_batch('train', {'a': np.array([0,1,2]), 'b': np.array([0,10,20]), 'c': np.array([1,1,1])})
        hook.after_batch('train', {'a': np.array([3,4,5]), 'b': np.array([30,40,50]), 'c': np.array([2,3,3])})

        hook.after_batch('valid', {'a': np.array([6,7,8]), 'b': np.array([60,70,80]), 'c': np.array([2,1,1])})
        hook.after_batch('valid', {'a': np.array([9,10,11]), 'b': np.array([90,100,110]), 'c': np.array([2,8,3])})

        hook.after_batch('test', {'a': np.array([12,13,14]), 'b': np.array([120,130,140]), 'c': np.array([0,9,1])})
        hook.after_batch('test', {'a': np.array([15,16,17]), 'b': np.array([150,160,170]), 'c': np.array([2,3,1])})

        self.assertListEqual([0,1,2,3,4,5], hook._train_buffer['a'])
        self.assertListEqual([0,10,20,30,40,50], hook._train_buffer['b'])

        self.assertListEqual([6,7,8,9,10,11], hook._valid_buffer['a'])
        self.assertListEqual([60,70,80,90,100,110], hook._valid_buffer['b'])

        self.assertListEqual([12,13,14,15,16,17], hook._test_buffer['a'])
        self.assertListEqual([120,130,140,150,160,170], hook._test_buffer['b'])

        train_res = {'a': 'yy', 'x': 'y'}
        valid_res = {'a': 'yy', 'x': 'y'}
        test_res = {'a': 'yy', 'x': 'y'}

        hook.after_epoch(epoch_id=1, train_results=train_res, valid_results=valid_res, test_results=test_res)

        self.assertListEqual([0,1,2,3,4,5], train_res['results']['a'])
        self.assertListEqual([0,10,20,30,40,50], train_res['results']['b'])

        self.assertListEqual([6,7,8,9,10,11], valid_res['results']['a'])
        self.assertListEqual([60,70,80,90,100,110], valid_res['results']['b'])

        self.assertListEqual([12,13,14,15,16,17], test_res['results']['a'])
        self.assertListEqual([120,130,140,150,160,170], test_res['results']['b'])

        self.assertIn('x', train_res)
        self.assertIn('x', valid_res)
        self.assertIn('x', test_res)

        self.assertEqual(0, len(hook._train_buffer))
        self.assertEqual(0, len(hook._valid_buffer))
        self.assertEqual(0, len(hook._test_buffer))

    def test_result_without_train(self):
        hook = ResultHook(metrics_to_log=['a', 'b'], log_train_results=False, net=None, config=None)

        hook.after_batch('train', {'a': np.array([0,1,2]), 'b': np.array([0,10,20]), 'c': np.array([1,1,1])})
        hook.after_batch('train', {'a': np.array([3,4,5]), 'b': np.array([30,40,50]), 'c': np.array([2,3,3])})

        hook.after_batch('valid', {'a': np.array([6,7,8]), 'b': np.array([60,70,80]), 'c': np.array([2,1,1])})
        hook.after_batch('valid', {'a': np.array([9,10,11]), 'b': np.array([90,100,110]), 'c': np.array([2,8,3])})

        hook.after_batch('test', {'a': np.array([12,13,14]), 'b': np.array([120,130,140]), 'c': np.array([0,9,1])})
        hook.after_batch('test', {'a': np.array([15,16,17]), 'b': np.array([150,160,170]), 'c': np.array([2,3,1])})

        self.assertListEqual([], hook._train_buffer['a'])
        self.assertListEqual([], hook._train_buffer['b'])

        self.assertListEqual([6,7,8,9,10,11], hook._valid_buffer['a'])
        self.assertListEqual([60,70,80,90,100,110], hook._valid_buffer['b'])

        self.assertListEqual([12,13,14,15,16,17], hook._test_buffer['a'])
        self.assertListEqual([120,130,140,150,160,170], hook._test_buffer['b'])

        train_res = {'a': 'yy', 'x': 'y'}
        valid_res = {'a': 'yy', 'x': 'y'}
        test_res = {'a': 'yy', 'x': 'y'}

        hook.after_epoch(epoch_id=1, train_results=train_res, valid_results=valid_res, test_results=test_res)

        self.assertDictEqual({'a': 'yy', 'x': 'y'}, train_res)

        self.assertListEqual([6,7,8,9,10,11], valid_res['results']['a'])
        self.assertListEqual([60,70,80,90,100,110], valid_res['results']['b'])

        self.assertListEqual([12,13,14,15,16,17], test_res['results']['a'])
        self.assertListEqual([120,130,140,150,160,170], test_res['results']['b'])

        self.assertIn('x', valid_res)
        self.assertIn('x', test_res)

        self.assertEqual(0, len(hook._train_buffer))
        self.assertEqual(0, len(hook._valid_buffer))
        self.assertEqual(0, len(hook._test_buffer))
