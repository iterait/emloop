import pytest
import os
import logging

from emloop.cli.eval import evaluate


@pytest.fixture
def yaml():
    yield """
          model:
            class: emloop.tests.cli.common_test.DummyModelWithKwargs
            
            io: 
              in: [a]
              out: [dummy]
                   
            outputs: [outputs]
        
          dataset:
            class: emloop.tests.cli.common_test.DummyDataset
        
          hooks:
          - emloop.tests.cli.common_test.DummyEvalHook:
              epochs: 1
        
          - StopAfter:
              epochs: 1
        
          eval:
            train:      
              model:
                class: emloop.tests.cli.common_test.DummyModel
                
                io: 
                  in: [a]
                  out: [dummy]
                
                outputs: [new_outputs]
                
              dataset:
                class: emloop.tests.cli.common_test.DummyDataset
        
              hooks:
              - StopAfter:
                  epochs: 2
                  
            valid:      
              model:
                class: emloop.tests.cli.common_test.DummyModel
                
                io: 
                  in: [a]
                  out: [dummy]
                
                outputs: [new_outputs]
                
              dataset:
                class: emloop.tests.cli.common_test.DummyDataset
        
              hooks:
              - StopAfter:
                  epochs: 2
          """


def test_evaluate(tmpdir, yaml, caplog):
    """Test configuration is first overridden by eval section and CLI arguments then override everything."""
    config_path = os.path.join(tmpdir, 'test.yaml')
    with open(config_path, 'w') as file:
        file.write(yaml)

    caplog.set_level(logging.DEBUG)

    evaluate(model_path='emloop.tests.cli.common_test.DummyModel', stream_name='valid',
             config_path=config_path, cl_arguments=['dataset.class=emloop.tests.cli.common_test.DummyEvalDataset'],
             output_root=tmpdir)

    assert 'DummyModel created' in caplog.text
    assert 'DummyModelWithKwargs created' not in caplog.text

    assert 'DummyEvalDataset created' in caplog.text
    assert 'DummyDataset created' not in caplog.text

    assert 'StopAfter created' in caplog.text
    assert 'DummyEvalHook created' not in caplog.text
