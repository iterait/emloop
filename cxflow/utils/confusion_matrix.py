import numpy as np


def confusion_matrix(expected: np.ndarray, predicted: np.ndarray, num_classes: int) -> np.ndarray:
    """
    Calculate and return confusion matrix for the predicted and expected labels

    :param expected: array of expected classes (integers) with shape `[num_of_data]`
    :param predicted: array of predicted classes (integers) with shape `[num_of_data]`
    :param num_classes: number of classification classes
    :return: confusion matrix (cm) with absolute values
    """
    assert np.issubclass_(expected.dtype.type, np.integer), " Classes' indices must be integers"
    assert np.issubclass_(predicted.dtype.type, np.integer), " Classes' indices must be integers"
    assert expected.shape == predicted.shape, "Predicted and expected data must be the same length"
    assert num_classes > np.max([predicted, expected]), \
        "Number of classes must be at least the number of indices in predicted/expected data"
    assert np.min([predicted, expected]) >= 0, " Classes' indices must be positive integers"
    cm_abs = np.zeros((num_classes, num_classes), dtype=np.int32)
    for pred, exp in zip(predicted, expected):
        cm_abs[exp, pred] += 1
    return cm_abs
