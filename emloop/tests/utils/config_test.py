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
            test:      
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

    # config not overridden
    config_model = {'class': 'Model', 'io': {'in': ['a'], 'out': ['b']}, 'outputs': ['c']}
    config_dataset = {'class': 'Dataset', 'batch_size': 10}
    config_hooks = [{'Hook_1': {'epochs': 1}}, {'Hook_2': {'epochs': 1}}]

    # config overridden by eval only
    config_model_eval = {'class': 'EvalModel', 'io': {'in': ['x'], 'out': ['y']}, 'outputs': ['c'], 'inputs': ['z']}
    config_dataset_eval = {'class': 'EvalDataset', 'batch_size': 10}
    config_hooks_eval = [{'Hook_2': {'epochs': 2}}]

    # config overridden by cli only
    config_model_cli = {'class': 'Model', 'io': {'in': ['m'], 'out': ['b']}, 'outputs': ['c']}
    config_dataset_cli = {'class': 'CliDataset', 'batch_size': 10}

    # config overridden by eval and then by cli
    config_model_eval_cli = {'class': 'EvalModel', 'io': {'in': ['m'], 'out': ['y']}, 'outputs': ['c'], 'inputs': ['z']}

    # neither cl_arguments nor override_stream specified - no override
    loaded_config_0 = load_config(config_file=orig_config)
    assert OrderedDict(loaded_config_0['model']) == OrderedDict(config_model)
    assert OrderedDict(loaded_config_0['dataset']) == OrderedDict(config_dataset)
    assert loaded_config_0['hooks'] == config_hooks

    # no cl_arguments specified, override_stream not in eval - no override
    loaded_config_1 = load_config(config_file=orig_config, override_stream='valid')
    assert OrderedDict(loaded_config_1['model']) == OrderedDict(config_model)
    assert OrderedDict(loaded_config_1['dataset']) == OrderedDict(config_dataset)
    assert loaded_config_1['hooks'] == config_hooks

    # no cl_arguments specified, override_stream in eval - override eval only
    loaded_config_2 = load_config(config_file=orig_config, override_stream='test')
    assert OrderedDict(loaded_config_2['model']) == OrderedDict(config_model_eval)
    assert OrderedDict(loaded_config_2['dataset']) == OrderedDict(config_dataset_eval)
    assert loaded_config_2['hooks'] == config_hooks_eval

    # cl_arguments specified, override_stream not in eval - override cli only
    loaded_config_3 = load_config(config_file=orig_config, additional_args=cl_arguments, override_stream='valid')
    assert OrderedDict(loaded_config_3['model']) == OrderedDict(config_model_cli)
    assert OrderedDict(loaded_config_3['dataset']) == OrderedDict(config_dataset_cli)
    assert loaded_config_3['hooks'] == config_hooks

    # cl_arguments specified, override_stream in eval - override eval, then cli
    loaded_config_3 = load_config(config_file=orig_config, additional_args=cl_arguments, override_stream='test')
    assert OrderedDict(loaded_config_3['model']) == OrderedDict(config_model_eval_cli)
    assert OrderedDict(loaded_config_3['dataset']) == OrderedDict(config_dataset_cli)
    assert loaded_config_3['hooks'] == config_hooks_eval
