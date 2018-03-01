"""
Test module for confusion_matrix utils (cxflow.utils.confusion_matrix).
"""
import numpy as np

from cxflow.utils.confusion_matrix import confusion_matrix
from cxflow.tests.test_core import CXTestCaseWithDir


class ConfusionMatrixTest(CXTestCaseWithDir):
    """Test case for the confusion_matrix."""

    def test_confusion_matrix(self):
        # test wrong argument types
        self.assertRaises(TypeError, confusion_matrix, expected=np.array([0]), predicted=np.array([0]), num_classes='a')
        self.assertRaises(AttributeError, confusion_matrix, expected=[0], predicted=np.array([0]), num_classes=3)
        self.assertRaises(AttributeError, confusion_matrix, expected=np.array([0]), predicted=[0], num_classes=3)

        # test wrong array shapes
        self.assertRaises(AssertionError, confusion_matrix, expected=np.array([[0]]), predicted=np.array([0]), num_classes=3)
        self.assertRaises(AssertionError, confusion_matrix, expected=np.array([0]), predicted=np.array([[0]]), num_classes=3)
        self.assertRaises(AssertionError, confusion_matrix, expected=np.array([0, 1]), predicted=np.array([0]), num_classes=3)

        # test wrong value of `num_classes` input
        self.assertRaises(AssertionError, confusion_matrix, expected=np.array([0, 2]), predicted=np.array([0, 1]), num_classes=2)

        # test wrong `np.dtype` of input arrays
        self.assertRaises(AssertionError, confusion_matrix, expected=np.array([0.]), predicted=np.array([0]), num_classes=3)
        self.assertRaises(AssertionError, confusion_matrix, expected=np.array([0]), predicted=np.array([0.]), num_classes=3)

        # test non-negativity of input array
        self.assertRaises(AssertionError, confusion_matrix, expected=np.array([-1]), predicted=np.array([0]), num_classes=3)
        self.assertRaises(AssertionError, confusion_matrix, expected=np.array([0]), predicted=np.array([-1]), num_classes=3)

        # test calculation of confusion_matrix
        expected = np.array([0, 1, 2])
        predicted = np.array([0, 1, 1])
        num_classes = 3
        calculated_cm = confusion_matrix(expected=expected, predicted=predicted, num_classes=num_classes)
        groundtruth_cm = np.array([[1, 0, 0],
                                   [0, 1, 0],
                                   [0, 1, 0]])
        np.testing.assert_equal(calculated_cm, groundtruth_cm)

        num_classes = 4
        calculated_cm = confusion_matrix(expected=expected, predicted=predicted, num_classes=num_classes)
        groundtruth_cm = np.array([[1, 0, 0, 0],
                                   [0, 1, 0, 0],
                                   [0, 1, 0, 0],
                                   [0, 0, 0, 0]])
        np.testing.assert_equal(calculated_cm, groundtruth_cm)

        expected = np.array([0, 1, 2, 1, 0, 1, 2, 2, 1, 0, 0, 0, 1, 2, 2])
        predicted = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2])
        num_classes = 3
        calculated_cm = confusion_matrix(expected=expected, predicted=predicted, num_classes=num_classes)
        groundtruth_cm = np.array([[2, 0, 3],
                                   [2, 2, 1],
                                   [1, 2, 2]])
        np.testing.assert_equal(calculated_cm, groundtruth_cm)
