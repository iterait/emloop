from abc import abstractmethod


class AbstractDataset:

    @abstractmethod
    def create_train_stream(self) -> None:
        pass

    @abstractmethod
    def create_valid_stream(self) -> None:
        pass

    @abstractmethod
    def create_test_stream(self) -> None:
        pass
