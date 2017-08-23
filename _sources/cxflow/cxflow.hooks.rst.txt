================
``cxflow.hooks``
================

.. automodule:: cxflow.hooks


.. currentmodule:: cxflow.hooks


Classes
=======

- :py:class:`AbstractHook`:
  Cxflow hooks interface.

- :py:class:`AccumulateVariables`:
  This hook accumulates the specified variables allowing their aggregation after each epoch.

- :py:class:`WriteCSV`:
  Log the training results to CSV file.

- :py:class:`StopAfter`:
  Stop the training after the specified number of epochs.

- :py:class:`LogVariables`:
  Log the training results to stderr via standard logging module.

- :py:class:`LogProfile`:
  Summarize and log epoch profile.

- :py:class:`SaveEvery`:
  Save the model every ``n`` epochs.

- :py:class:`SaveBest`:
  Save the model when it outperforms itself.

- :py:class:`CatchSigint`:
  SIGINT catcher.

- :py:class:`ComputeStats`:
  Accumulate the specified variables, compute the specified aggregation values and save them to the epoch data.

- :py:class:`Check`:
  Terminate training if the given stream variable exceeds the threshold in at most specified number of epochs.



.. autoclass:: AbstractHook
   :show-inheritance:
   :members:
   :private-members:
   :special-members:

   .. inheritance-diagram:: AbstractHook
      :parts: 1

   |



.. autoclass:: AccumulateVariables
   :show-inheritance:
   :members:
   :private-members:
   :special-members:

   .. inheritance-diagram:: AccumulateVariables
      :parts: 1

   |



.. autoclass:: WriteCSV
   :show-inheritance:
   :members:
   :private-members:
   :special-members:

   .. inheritance-diagram:: WriteCSV
      :parts: 1

   |



.. autoclass:: StopAfter
   :show-inheritance:
   :members:
   :private-members:
   :special-members:

   .. inheritance-diagram:: StopAfter
      :parts: 1

   |



.. autoclass:: LogVariables
   :show-inheritance:
   :members:
   :private-members:
   :special-members:

   .. inheritance-diagram:: LogVariables
      :parts: 1

   |



.. autoclass:: LogProfile
   :show-inheritance:
   :members:
   :private-members:
   :special-members:

   .. inheritance-diagram:: LogProfile
      :parts: 1

   |



.. autoclass:: SaveEvery
   :show-inheritance:
   :members:
   :private-members:
   :special-members:

   .. inheritance-diagram:: SaveEvery
      :parts: 1

   |



.. autoclass:: SaveBest
   :show-inheritance:
   :members:
   :private-members:
   :special-members:

   .. inheritance-diagram:: SaveBest
      :parts: 1

   |



.. autoclass:: CatchSigint
   :show-inheritance:
   :members:
   :private-members:
   :special-members:

   .. inheritance-diagram:: CatchSigint
      :parts: 1

   |



.. autoclass:: ComputeStats
   :show-inheritance:
   :members:
   :private-members:
   :special-members:

   .. inheritance-diagram:: ComputeStats
      :parts: 1

   |



.. autoclass:: Check
   :show-inheritance:
   :members:
   :private-members:
   :special-members:

   .. inheritance-diagram:: Check
      :parts: 1

   |



Exceptions
==========

- :py:exc:`TrainingTerminated`:
  Exception that is raised when a hook terminates the training.


.. autoexception:: TrainingTerminated

   .. rubric:: Inheritance
   .. inheritance-diagram:: TrainingTerminated
      :parts: 1
