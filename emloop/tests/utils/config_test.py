"""
Test module for config utils functions (emloop.utils.config).
"""
from os import path
from collections import OrderedDict
import pytest

from emloop.utils.config import parse_arg, load_config


_ARGS = [([('common.name', 'BatchSize1'), ('model.name', 'modelie'), ('stream.train.seed', 'none')], str),
         ([('common.batch_size', 12), ('model.dropout', 0), ('stream.train.seed', 123)], int),
         ([('common.some_int_number', 12.), ('model.dropout', 0.5), ('stream.train.float_seed', 123.456)], float),
         ([('common.quiet', True), ('model.dropout', False), ('stream.train.float_seed', True)], bool)]


@pytest.mark.parametrize('args, arg_type', _ARGS)
def test_arg_type(args, arg_type):
    """Test case for parse_arg function."""
    for key, val in args:
        parsed_key, parsed_val = parse_arg(key + '=' + str(val))
        assert (key, val) == (parsed_key, parsed_val)
        assert type(parsed_val) == arg_type


def test_ast_type():
    """Test ast type."""
    args = [('common.arch', [1, 2, 3.4, 5]), ('model.arch', {"a": "b"}),
                     ('stream.train.deep', {"a": {"b": ["c", "d", "e"]}}), ('model.arch', 12), ('model.arch', 12.2)]
    for key, val in args:
        parsed_key, parsed_val = parse_arg(key+'='+str(val))
        assert (key, val) == (parsed_key, parsed_val)
        assert type(parsed_val) == type(val)


_ANCHORLESS_KEYS = [([[]], {'e': {'f': 'f', 'h': ['j', 'k']}}),
                    ([['e.f=12']], {'e': {'f': 12, 'h': ['j', 'k']}}),
                    ([['e.x=12']], {'e': {'f': 'f', 'h': ['j', 'k'], 'x': 12}})]


@pytest.mark.parametrize('params, expected_output', _ANCHORLESS_KEYS)
def test_load_anchorless_config(tmpdir, anchorless_yaml, params, expected_output):
    """Test loading of a config without yaml anchors."""

    f_name = path.join(tmpdir, 'conf.yaml')

    with open(f_name, 'w') as file:
        file.write(anchorless_yaml)

    assert load_config(f_name, *params) == expected_output


_ANCHORED_KEYS = [([[]], {'b': 'c', 'd': 11}, [('f', 'f'), ('h', ['j', 'k']), ('b', 'c'), ('d', 11)]),
                   ([['a.b=12']], {'b': 12, 'd': 11}, [('f', 'f'), ('h', ['j', 'k']), ('b', 12), ('d', 11)]),
                   ([['e.b=19']], {'b': 'c', 'd': 11}, [('f', 'f'), ('h', ['j', 'k']), ('b', 19), ('d', 11)])]


@pytest.mark.parametrize('params, a_key, e_key', _ANCHORED_KEYS)
def test_load_anchored_config(tmpdir, anchored_yaml, params, a_key, e_key):
    """Test loading of a config with yaml anchors."""
    f_name = path.join(tmpdir, 'conf.yaml')

    with open(f_name, 'w') as file:
        file.write(anchored_yaml)

    assert load_config(f_name, *params)['a'] == a_key
    assert OrderedDict(load_config(f_name, *params)['e']) == OrderedDict(e_key)


@pytest.fixture
def yaml():
    yield """
          model:
            class: Model

            io: 
              in: [a]
              out: [b]

            outputs: [c]

          dataset:
            class: Dataset
            
            batch_size: 10

          hooks:
          - Hook_1:
              epochs: 1

          - Hook_2:
              epochs: 1

          eval:
            train:      
              model:
                class: EvalModel

                io: 
                  in: [x]
                  out: [y]
                  
                inputs: [z]

              dataset:
                class: EvalDataset

              hooks:
              - Hook_2:
                  epochs: 2
          """


def test_override(tmpdir, yaml):
    """Test configuration is overridden by the eval section and subsequently by the CLI arguments."""
    orig_config = path.join(tmpdir, 'test.yaml')

    with open(orig_config, 'w') as file:
        file.write(yaml)

    cl_arguments = ['dataset.class=CliDataset', 'model.io.in=[m]']

    # no cl_arguments nor override_stream specified - no override
    config_0 = {'model': {'class': 'Model', 'io': {'in': ['a'], 'out': ['b']}, 'outputs': ['c']},
                'dataset': {'class': 'Dataset', 'batch_size': 10},
                'hooks': [{'Hook_1': {'epochs': 1}}, {'Hook_2': {'epochs': 1}}],
                'eval': {'train': {'model': {'class': 'EvalModel', 'io': {'in': ['x'], 'out': ['y']}, 'inputs': ['z']},
                                   'dataset': {'class': 'EvalDataset'},
                                   'hooks': [{'Hook_2': {'epochs': 2}}]}}}
    assert OrderedDict(load_config(orig_config)) == OrderedDict(config_0)

    # no cl_arguments specified, override_stream not in eval - no override
    assert OrderedDict(load_config(config_file=orig_config, override_stream='valid')) == OrderedDict(config_0)

    # no cl_arguments specified, override_stream in eval - override
    config_1 = {'model': {'class': 'EvalModel', 'io': {'in': ['x'], 'out': ['y']}, 'outputs': ['c'], 'inputs': ['z']},
                'dataset': {'class': 'EvalDataset', 'batch_size': 10},
                'hooks': [{'Hook_2': {'epochs': 2}}],
                'eval': {'train': {'model': {'class': 'EvalModel', 'io': {'in': ['x'], 'out': ['y']}, 'inputs': ['z']},
                                   'dataset': {'class': 'EvalDataset'},
                                   'hooks': [{'Hook_2': {'epochs': 2}}]}}}
    assert OrderedDict(load_config(config_file=orig_config, override_stream='train')) == OrderedDict(config_1)

    # cl_arguments specified, override_stream not in eval - override
    config_2 = {'model': {'class': 'Model', 'io': {'in': ['m'], 'out': ['b']}, 'outputs': ['c']},
                'dataset': {'class': 'CliDataset', 'batch_size': 10},
                'hooks': [{'Hook_1': {'epochs': 1}}, {'Hook_2': {'epochs': 1}}],
                'eval': {'train': {'model': {'class': 'EvalModel', 'io': {'in': ['x'], 'out': ['y']}, 'inputs': ['z']},
                                   'dataset': {'class': 'EvalDataset'},
                                   'hooks': [{'Hook_2': {'epochs': 2}}]}}}
    assert OrderedDict(load_config(orig_config, cl_arguments, 'valid')) == OrderedDict(config_2)

    # cl_arguments specified, override_stream in eval - override
    config_3 = {'model': {'class': 'EvalModel', 'io': {'in': ['m'], 'out': ['y']}, 'outputs': ['c'], 'inputs': ['z']},
                'dataset': {'class': 'CliDataset', 'batch_size': 10},
                'hooks': [{'Hook_2': {'epochs': 2}}],
                'eval': {'train': {'model': {'class': 'EvalModel', 'io': {'in': ['x'], 'out': ['y']}, 'inputs': ['z']},
                                   'dataset': {'class': 'EvalDataset'},
                                   'hooks': [{'Hook_2': {'epochs': 2}}]}}}
    assert OrderedDict(load_config(orig_config, cl_arguments, 'train')) == OrderedDict(config_3)
