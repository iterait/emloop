import logging

from .util import fallback
from .common import create_dataset
from ..utils.config import load_config


def split(config_file: str, num_splits: int, train_ratio: float, valid_ratio: float, test_ratio: float=0) -> None:
    """
    Create dataset and call the split method with the given args.
    :param config_file: path to the training yaml config
    :param num_splits: number of x-val splits to be created
    :param train_ratio: portion of data to be split to the train set
    :param valid_ratio: portion of data to be split to the valid set
    :param test_ratio: portion of data to be split to the test set
    """
    logging.info('Splitting to %d splits with ratios %f:%f:%f', num_splits, train_ratio, valid_ratio, test_ratio)

    config = dataset = None

    try:
        logging.info('Loading config')
        config = load_config(config_file=config_file, additional_args=[])
    except Exception as ex:  # pylint: disable=broad-except
        fallback('Loading config failed', ex)

    try:
        logging.info('Creating dataset')
        dataset = create_dataset(config)
    except Exception as ex:  # pylint: disable=broad-except
        fallback('Creating dataset failed', ex)

    logging.info('Splitting')
    dataset.split(num_splits, train_ratio, valid_ratio, test_ratio)
