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
"""Converts checkpoint variables into Const ops in a standalone GraphDef file.

This script is designed to take a GraphDef proto, a SaverDef proto, and a set of
variable values stored in a checkpoint file, and output a GraphDef with all of
the variable ops converted into const ops containing the values of the
variables.

It's useful to do this when we need to load a single file in C++, especially in
environments like mobile or embedded where we may not have access to the
RestoreTensor ops and file loading calls that they rely on.

You can look at freeze_graph_test.py for an example of how to use it.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import logging
import sys

from google.protobuf import text_format

from tensorflow.core.framework import graph_pb2
from tensorflow.core.protobuf import saver_pb2
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.client import session
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import importer
from tensorflow.python.platform import gfile
from tensorflow.python.training import saver as saver_lib


def freeze_graph(input_graph,
                 input_saver,
                 input_binary,
                 input_checkpoint,
                 output_node_names,
                 restore_op_name,
                 filename_tensor_name,
                 output_graph,
                 clear_devices,
                 initializer_nodes,
                 variable_names_blacklist=""):
  """Converts all variables in a graph and checkpoint into constants."""

  del restore_op_name, filename_tensor_name  # Unused by updated loading code.

  if not gfile.Exists(input_graph):
    raise ValueError("Input graph file '" + input_graph + "' does not exist!")

  if input_saver and not gfile.Exists(input_saver):
    raise ValueError("Input saver file '" + input_saver + "' does not exist!")

  # 'input_checkpoint' may be a prefix if we're using Saver V2 format
  if not saver_lib.checkpoint_exists(input_checkpoint):
    raise ValueError("Input checkpoint '" + input_checkpoint + "' doesn't exist!")

  if not output_node_names:
    raise ValueError("You need to supply the name of a node to --output_node_names.")

  input_graph_def = graph_pb2.GraphDef()
  mode = "rb" if input_binary else "r"
  with gfile.FastGFile(input_graph, mode) as f:
    if input_binary:
      input_graph_def.ParseFromString(f.read())
    else:
      text_format.Merge(f.read(), input_graph_def)
  # Remove all the explicit device specifications for this node. This helps to
  # make the graph more portable.
  if clear_devices:
    for node in input_graph_def.node:
      node.device = ""

  _ = importer.import_graph_def(input_graph_def, name="")

  with session.Session() as sess:
    if input_saver:
      with gfile.FastGFile(input_saver, mode) as f:
        saver_def = saver_pb2.SaverDef()
        if input_binary:
          saver_def.ParseFromString(f.read())
        else:
          text_format.Merge(f.read(), saver_def)
        saver = saver_lib.Saver(saver_def=saver_def)
        saver.restore(sess, input_checkpoint)
    else:
      var_list = {}
      reader = pywrap_tensorflow.NewCheckpointReader(input_checkpoint)
      var_to_shape_map = reader.get_variable_to_shape_map()
      for key in var_to_shape_map:
        try:
          tensor = sess.graph.get_tensor_by_name(key + ":0")
        except KeyError:
          # This tensor doesn't exist in the graph (for example it's
          # 'global_step' or a similar housekeeping element) so skip it.
          continue
        var_list[key] = tensor
      saver = saver_lib.Saver(var_list=var_list)
      saver.restore(sess, input_checkpoint)
      if initializer_nodes:
        sess.run(initializer_nodes)

    variable_names_blacklist = (variable_names_blacklist.split(",") if
                                variable_names_blacklist else None)
    output_graph_def = graph_util.convert_variables_to_constants(
        sess,
        input_graph_def,
        output_node_names.split(","),
        variable_names_blacklist=variable_names_blacklist)

  with gfile.GFile(output_graph, "wb") as f:
    f.write(output_graph_def.SerializeToString())
  logging.debug('%d ops in the final graph." % len(output_graph_def.node)')
