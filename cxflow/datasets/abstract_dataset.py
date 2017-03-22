from abc import abstractmethod
import typing


class AbstractDataset:
    """
    cxflow dataset concept.

    This is rather documentation-purpose class and it should not be inherited.
    """

    Batch = typing.NewType('Batch', typing.Mapping[str, typing.Any])
    Stream = typing.NewType('Stream', typing.Iterable[Batch])

    @abstractmethod
    def create_train_stream(self) -> Stream:
        """Return a train iterator which provides a mapping (source name -> value)."""
        pass

    @abstractmethod
    def create_valid_stream(self) -> Stream:
        """Return a valid iterator which provides a mapping (source name -> value)."""
        pass

    def create_test_stream(self) -> Stream:
        """
        Return a test iterator which provides a mapping (source name -> value).

        This method is not mandatory.
        """
        pass

    @abstractmethod
    def split(self, num_splits: int, train: float, valid: float, test: float):
        """
        Perform cross-val split.

        This method is not mandatory.
        """
        pass
