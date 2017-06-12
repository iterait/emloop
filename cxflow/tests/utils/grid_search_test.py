"""
Test module for config utils functions (cxflow.utils.grid_search).
"""

from cxflow.tests.test_core import CXTestCase
from cxflow.utils.grid_search import _build_grid_search_commands


class GridSearchTest(CXTestCase):
    """Test case for grid search."""

    def test_built_params(self):
        """Test built parameter configurations."""

        script = ['echo', 'fixed_param']
        param1 = ['1', '2', '3']
        param2 = ['"hello"', '"hi"']

        params = _build_grid_search_commands(script=' '.join(script),
                                             params=['param1:int=[{}]'.format(', '.join(param1)),
                                                     'param2=[{}]'.format(', '.join(param2))])
        self.assertCountEqual(params,
                              [['echo', 'fixed_param', 'param1:int="1"', 'param2="hello"'],
                               ['echo', 'fixed_param', 'param1:int="2"', 'param2="hello"'],
                               ['echo', 'fixed_param', 'param1:int="3"', 'param2="hello"'],
                               ['echo', 'fixed_param', 'param1:int="1"', 'param2="hi"'],
                               ['echo', 'fixed_param', 'param1:int="2"', 'param2="hi"'],
                               ['echo', 'fixed_param', 'param1:int="3"', 'param2="hi"']])
