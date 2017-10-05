"""
Module with StopOnPlateau hook which terminates the training when the observed variable reaches its plateau.
"""

from . import TrainingTerminated, OnPlateau


class StopOnPlateau(OnPlateau):
    """
    Terminate the training when the observed variable reaches its plateau.

    .. code-block:: yaml
        :caption: stop the training when the mean of last 100 valid ``loss`` values is
                  smaller than the mean of last 30 ``loss`` values.

        hooks:
          - StopOnPlateau:
              long_term: 100
              short_term: 30


    .. code-block:: yaml
        :caption: stop the training when accuracy stops improving (raising)

        hooks:
          - StopOnPlateau:
              variable: accuracy
              objective: max

    """

    def _on_plateau_action(self, **kwargs) -> None:
        """
        Terminate the training when the observed variable reaches its plateau.

        :raise TrainingTerminated: if the model stops improving
        """
        raise TrainingTerminated('Detected plateau of variable `{}` in stream `{}`'.
                                 format(self._variable, self._stream))
