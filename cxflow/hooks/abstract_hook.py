from ..nets.abstract_net import AbstractNet


class TrainingTerminated(Exception):
    pass


class AbstractHook:
    def __init__(self, net: AbstractNet, config: dict, **kwargs):
        pass

    def before_training(self, net: AbstractNet, **kwargs) -> None:
        pass

    def before_first_epoch(self, net: AbstractNet, valid_results: dict, test_results: dict=None, **kwargs) -> None:
        pass

    def after_batch(self, net: AbstractNet, stream_type: str, results: dict, **kwargs) -> None:
        pass

    def after_epoch(self, net: AbstractNet, epoch_id: int, train_results: dict, valid_results: dict,
                    test_results: dict=None, **kwargs) -> None:
        pass

    def after_training(self, net: AbstractNet, **kwargs) -> None:
        pass
