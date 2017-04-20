# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# Modifications copyright 2017 Cognexa Solutions s.r.o.
# ==============================================================================
"""Converts checkpoint variables into Const ops in a standalone GraphDef file.

This script is designed to take a GraphDef proto, and a set of
variable values stored in a checkpoint file, and output a GraphDef with all of
the variable ops converted into const ops containing the values of the
variables.

It's useful to do this when we need to load a single file in C++, especially in
environments like mobile or embedded where we may not have access to the
RestoreTensor ops and file loading calls that they rely on.

You can look at freeze_graph_test.py for an example of how to use it.

"""
from typing import Iterable

from ...utils.misc import DisabledLogger, DisabledPrint
from tensorflow.core.framework import graph_pb2  # pylint: disable=no-name-in-module
from tensorflow.python import pywrap_tensorflow  # pylint: disable=no-name-in-module
from tensorflow.python.client import session  # pylint: disable=no-name-in-module
from tensorflow.python.framework import graph_util  # pylint: disable=no-name-in-module
from tensorflow.python.framework import importer  # pylint: disable=no-name-in-module
from tensorflow.python.platform import gfile  # pylint: disable=no-name-in-module
from tensorflow.python.training import saver as saver_lib  # pylint: disable=no-name-in-module


def freeze_graph(input_graph: str, input_checkpoint: str, output_node_names: Iterable[str], output_graph: str):
    """
    Convert all variables in a graph and checkpoint into constants and save the new graph to the specified file.

    Additionally, the graph is pruned of all nodes that are not needed for the specified outputs.

    -------------------------------------------------------
    NOTE: this function creates new nodes in the default graph, you may want to wrap the call with the following code
    -------------------------------------------------------
    with tf.Graph().as_default():
        freeze_graph(...)
    -------------------------------------------------------

    :param input_graph: path to the input graph file
    :param input_checkpoint: path to the input checkpoint
    :param output_node_names: iterable collection of output node names
    :param output_graph: path to the output frozen graph file

    Raises:
        ValueError: if any of the specified files does not exist
    """

    if not gfile.Exists(input_graph):
        raise ValueError('Input graph file `{}` does not exist!'.format(input_graph))

    # 'input_checkpoint' may be a prefix if we're using Saver V2 format
    if not saver_lib.checkpoint_exists(input_checkpoint):
        raise ValueError('Input checkpoint `{}` does not exist!'.format(input_checkpoint))

    # read the graph definition
    input_graph_def = graph_pb2.GraphDef()
    with gfile.FastGFile(input_graph, 'rb') as file:
        input_graph_def.ParseFromString(file.read())

    # remove all the explicit device specifications for this node. This helps to make the graph more portable.
    for node in input_graph_def.node:
        node.device = ''

    # restore the input graph and checkpoint
    _ = importer.import_graph_def(input_graph_def, name='')
    with session.Session() as sess:
        var_list = {}
        reader = pywrap_tensorflow.NewCheckpointReader(input_checkpoint)
        var_to_shape_map = reader.get_variable_to_shape_map()
        for key in var_to_shape_map:
            try:
                tensor = sess.graph.get_tensor_by_name(key + ':0')
            except KeyError:
                # This tensor doesn't exist in the graph (for example it's
                # 'global_step' or a similar housekeeping element) so skip it.
                continue
            var_list[key] = tensor
        saver = saver_lib.Saver(var_list=var_list)
        saver.restore(sess, input_checkpoint)

        # convert all the variables to constants
        with DisabledLogger('tensorflow'), DisabledPrint():
            output_graph_def = graph_util.convert_variables_to_constants(sess, input_graph_def, output_node_names)

    with gfile.GFile(output_graph, 'wb') as file:
        file.write(output_graph_def.SerializeToString())
