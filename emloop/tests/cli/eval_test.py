import pytest
import os
import logging

from emloop.cli.eval import evaluate


@pytest.fixture
def yaml():
    yield """
          model:
            class: emloop.tests.cli.common_test.DummyModelWithKwargs
                   
            outputs: [outputs]
        
          dataset:
            class: emloop.tests.cli.common_test.DummyDataset
        
          hooks:
          - emloop.tests.cli.common_test.DummyEvalHook:
              epochs: 1
        
          - StopAfter:
              epochs: 1
        
          eval:
            valid:      
              model:
                class: emloop.tests.cli.common_test.DummyModel
                
                io: 
                  in: [a]
                  out: [dummy]
                
                outputs: [new_outputs]
                
              dataset:
                class: emloop.tests.cli.common_test.DummyConfigDataset
        
              hooks:
              - StopAfter:
                  epochs: 2
          """


config = {'dataset': {'class': 'Dataset'},
          'hooks': [{'emloop.tests.cli.common_test.DummyEvalHook': {'epochs': 1}}, {'StopAfter': {'epochs': 1}}],
          'model': {'class': 'Model'},
          'eval': {'train': {'model': {'class': 'emloop.tests.cli.common_test.DummyModel',
                                       'io': {'in': ['a'], 'out': ['dummy']}},
                             'dataset': {'class': 'emloop.tests.cli.common_test.DummyDataset'}}}}


def test_evaluate(tmpdir, yaml, caplog):
    config_path = os.path.join(tmpdir, 'test.yaml')
    with open(config_path, 'w') as file:
        file.write(yaml)

    caplog.set_level(logging.DEBUG)

    evaluate(model_path='emloop.tests.cli.common_test.DummyModel', stream_name='valid',
             config_path=config_path, cl_arguments=['dataset=class: emloop.tests.cli.common_test.DummyEvalDataset'],
             output_root=tmpdir)

    assert 'DummyModel created' in caplog.text
    assert 'DummyEvalDataset created' in caplog.text
    assert 'StopAfter created' in caplog.text

    assert 'DummyModelWithKwargs created' not in caplog.text
    assert 'DummyDataset created' not in caplog.text
    assert 'DummyConfigDataset created' not in caplog.text
    assert 'DummyEvalHook created' not in caplog.text
