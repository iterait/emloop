Advanced
========

This section dives into the implementation details of the **emloop** components.
See the following sections for more information.

.. toctree::
   :maxdepth: 1

   main_loop
   dataset
   model
   hook
   config

.. Architecture
   ------------
   As already mentioned, emloop consists of multiple mutually orthogonal components.
   The following figure demonstrate their relationships.
   Each solid rectangle represents a single component.
   The public methods and attributes are listed below in order to provide an insight of the component API.
   The solid lines represents passing one object to the constructor of another.
   For example, Dataset is passed to Hook, Model and MainLoop.
   On the contrary, Model is passed only to Hook and MainLoop.
   The dashed lines represent the possible method invocations among the components.
   For example, MainLoop can invoke Hook's ``after_batch`` method once another batch is processed.
   .. figure:: emloop-architecture.svg
      :scale: 100%
      :align: center
      emloop overall architecture.
