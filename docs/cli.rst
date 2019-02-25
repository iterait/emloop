CLI Reference
=============
Although the whole **emloop** API can be accessed programmatically, the intended way of using it is through the
command line instruments. The design goal is to focus on defining the actual models, datasets etc.
instead of the burdensome code, which just puts all the components together.

With proper installation, the |emloop| command should become available. The command comes with four basic sub-commands
explained below.

.. |emloop| raw:: html

   <kbd>emloop</kbd>

.. code-block:: yaml

   usage: emloop (train | resume | predict | dataset) [-v] [-o] ...

All the sub-commands share the following arguments:

.. raw:: html

   <table class="docutils option-list" frame="void" rules="none">
   <col class="option" />
   <col class="description" />
   <tbody valign="top">
   <tr><td class="option-group">
   <kbd>--output, -o</kbd></td>
   <td>output directory, defaults to <code class="class="docutils literal"><span class="pre">./log</span></code></td></tr>
   <tr><td class="option-group">
   <kbd>--verbose, -v</kbd></td>
   <td>increase verbosity level to <code class="class="docutils literal"><span class="pre">DEBUG</span></code></td></tr>
   </tbody>
   </table>

emloop train
------------
.. argparse::
   :ref: emloop.cli.get_emloop_arg_parser
   :prog: emloop
   :path: train

emloop resume
-------------
.. argparse::
   :ref: emloop.cli.get_emloop_arg_parser
   :prog: emloop
   :path: resume

emloop eval
--------------
.. argparse::
   :ref: emloop.cli.get_emloop_arg_parser
   :prog: emloop
   :path: eval

emloop dataset
--------------
.. argparse::
   :ref: emloop.cli.get_emloop_arg_parser
   :prog: emloop
   :path: dataset

emloop ls
--------------
.. argparse::
   :ref: emloop.cli.get_emloop_arg_parser
   :prog: emloop
   :path: ls


emloop prune
--------------
.. argparse::
   :ref: emloop.cli.get_emloop_arg_parser
   :prog: emloop
   :path: prune
