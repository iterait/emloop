"""
Module with AbstractDataset concept.

"""
class AbstractDataset:
    """
    This concept prescribes the API that is required from every **cxflow** dataset.

    Every **cxflow** dataset has to have a constructor which takes YAML string config.
    Additionally, one may implement any ``<stream_name>_stream`` method
    in order to make ``stream_name`` stream available in the **cxflow** :py:class:`cxflow.MainLoop`.

    All the defined stream methods should return a :py:attr:`Stream`.
    """

    def __init__(self, config_str: str):
        """
        Create new dataset configured with the given YAML string (obligatory).

        The configuration must contain ``dataset`` entry and may contain ``output_dir`` entry.

        :param config_str: YAML string config
        """
        pass
