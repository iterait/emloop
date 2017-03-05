from ..nets.abstract_net import AbstractNet


class TrainingTerminated(Exception):
    pass


class AbstractHook:
    def __init__(self, net: AbstractNet, config: dict, **kwargs):
        pass

    def before_training(self, **kwargs) -> None:
        pass

    def before_first_epoch(self, valid_results: dict, test_results: dict=None, **kwargs) -> None:
        pass

    def after_batch(self, stream_type: str, results: dict, **kwargs) -> None:
        pass

    def after_epoch(self, epoch_id: int, train_results: dict, valid_results: dict, test_results: dict=None,
                    **kwargs) -> None:
        pass

    def after_training(self, **kwargs) -> None:
        pass
