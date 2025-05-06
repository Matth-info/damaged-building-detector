import numpy as np
import torch

from .confusionmatrix import ConfusionMatrix
from .metric import Metric


class IoU(Metric):
    """
    Computes the intersection over union (IoU) per class and corresponding mean (mIoU).

    Intersection over union (IoU) is a common evaluation metric for semantic
    segmentation. It uses the confusion matrix to compute IoU per class:

        IoU = true_positive / (true_positive + false_positive + false_negative).

    Keyword arguments:
    - num_classes (int): Number of classes in the classification problem.
    - normalized (boolean, optional): Determines whether the confusion matrix is normalized.
      Default: False.
    - ignore_index (int or iterable, optional): Index of the classes to ignore when computing IoU.
      Can be an int, or any iterable of ints.
    """

    def __init__(self, num_classes, normalized=False, ignore_index=None):
        super().__init__()
        self.conf_metric = ConfusionMatrix(num_classes, normalized)

        if ignore_index is None:
            self.ignore_index = None
        elif isinstance(ignore_index, int):
            self.ignore_index = (ignore_index,)
        else:
            try:
                self.ignore_index = tuple(ignore_index)
            except TypeError:
                raise ValueError("'ignore_index' must be an int or iterable")

    def reset(self):
        """Resets the confusion matrix."""
        self.conf_metric.reset()

    def add(self, predicted, target):
        """
        Adds the predicted and target pair to the IoU metric.

        Args:
        - predicted (Tensor): Can be an (N, K, H, W) tensor of predicted scores obtained from
                              the model for N examples and K classes, or (N, H, W) tensor of
                              integer values between 0 and K-1.
        - target (Tensor): Can be an (N, K, H, W) tensor of target scores for N examples and
                           K classes, or (N, H, W) tensor of integer values between 0 and K-1.
        """
        # Validate batch sizes
        assert predicted.size(0) == target.size(
            0
        ), "Number of targets and predicted outputs do not match"

        # Validate dimensions
        assert predicted.dim() in [
            3,
            4,
        ], "Predictions must be of dimension (N, H, W) or (N, K, H, W)"
        assert target.dim() in [
            3,
            4,
        ], "Targets must be of dimension (N, H, W) or (N, K, H, W)"

        # Flatten for confusion matrix processing
        self.conf_metric.add(predicted, target)

    def value(self):
        """
        Computes the IoU and mean IoU.

        The mean computation ignores NaN elements of the IoU array.

        Returns:
            Tuple: (IoU, mIoU). The first output is the per-class IoU,
                   a numpy.ndarray with K elements for K classes.
                   The second output is the mean IoU (mIoU).
        """
        conf_matrix = self.conf_metric.value()

        if self.ignore_index is not None:
            for index in self.ignore_index:
                conf_matrix[:, index] = 0
                conf_matrix[index, :] = 0

        true_positive = np.diag(conf_matrix)
        false_positive = np.sum(conf_matrix, axis=0) - true_positive
        false_negative = np.sum(conf_matrix, axis=1) - true_positive

        # Handle division by zero gracefully
        with np.errstate(divide="ignore", invalid="ignore"):
            iou = true_positive / (true_positive + false_positive + false_negative)

        # Return IoU for each class and the mean IoU
        return iou, np.nanmean(iou)
