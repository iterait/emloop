from abc import abstractmethod


class AbstractFuelDataset:

    def __init__(self, **kwargs):
        # Save kwargs
        for name, value in kwargs.items():
            setattr(self, name, value)

    def create_train_stream(self):
        return self._create_train_stream(**self.train)

    def create_valid_stream(self):
        return self._create_valid_stream(**self.valid)

    def create_test_stream(self):
        return self._create_test_stream(**self.test)

    @abstractmethod
    def _create_train_stream(self, **kwargs):
        pass

    @abstractmethod
    def _create_valid_stream(self, **kwargs):
        pass

    @abstractmethod
    def _create_test_stream(self, **kwargs):
        pass
