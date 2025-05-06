import numpy as np
import torch
from .metric import Metric


class ConfusionMatrix(Metric):
    """
    Constructs a confusion matrix for multi-class classification problems.

    Does not support multi-label, multi-class problems.

    Keyword arguments:
    - num_classes (int): Number of classes in the classification problem.
    - normalized (boolean, optional): Determines whether or not the confusion
      matrix is normalized or not. Default: False.

    Modified from: https://github.com/pytorch/tnt/blob/master/torchnet/meter/confusionmeter.py
    """

    def __init__(self, num_classes, normalized=False):
        super().__init__()

        self.conf = np.ndarray((num_classes, num_classes), dtype=np.int32)
        self.normalized = normalized
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        """Resets the confusion matrix to zeros."""
        self.conf.fill(0)

    def add(self, predicted, target):
        """
        Updates the confusion matrix based on predictions and targets.

        Handles both single-instance and batch predictions.

        Args:
            predicted (Tensor or numpy.ndarray): Can be an N x K tensor/array of
            predicted scores obtained from the model for N examples and K classes,
            or an N-tensor/array of integer values between 0 and K-1.

            target (Tensor or numpy.ndarray): Can be an N x K tensor/array of
            ground-truth classes for N examples and K classes, or an N-tensor/array
            of integer values between 0 and K-1.
        """
        # Convert tensors to numpy arrays if necessary
        if torch.is_tensor(predicted):
            predicted = predicted.cpu().numpy()
        if torch.is_tensor(target):
            target = target.cpu().numpy()

        # Ensure batch dimension consistency
        assert predicted.shape[0] == target.shape[0], "Number of targets and predicted outputs do not match"

        # If predicted is not a 1D array, convert class scores to class indices
        if np.ndim(predicted) != 1:
            assert (
                predicted.shape[1] == self.num_classes
            ), "Number of predictions does not match size of confusion matrix"
            predicted = np.argmax(predicted, axis=1)
        else:
            assert (predicted.max() < self.num_classes) and (
                predicted.min() >= 0
            ), "Predicted values are not between 0 and k-1"

        # If target is not a 1D array, convert one-hot encoding to class indices
        if np.ndim(target) != 1:
            assert target.shape[1] == self.num_classes, "One-hot target does not match size of confusion matrix"
            assert (target >= 0).all() and (target <= 1).all(), "In one-hot encoding, target values should be 0 or 1"
            assert (target.sum(axis=1) == 1).all(), "Multi-label setting is not supported"
            target = np.argmax(target, axis=1)
        else:
            assert (target.max() < self.num_classes) and (target.min() >= 0), "Target values are not between 0 and k-1"

        # flatten target and predictions
        predicted = predicted.flatten()
        target = target.flatten()
        # Compute confusion matrix for the batch
        x = predicted + self.num_classes * target
        bincount_2d = np.bincount(x.astype(np.int32), minlength=self.num_classes**2)
        assert bincount_2d.size == self.num_classes**2
        conf = bincount_2d.reshape((self.num_classes, self.num_classes))

        # Accumulate the batch confusion matrix
        self.conf += conf

    def value(self):
        """
        Returns:
            Confusion matrix of shape K x K, where rows correspond to ground-truth
            targets and columns correspond to predicted targets.
        """
        if self.normalized:
            conf = self.conf.astype(np.float32)
            return conf / conf.sum(axis=1, keepdims=True).clip(min=1e-12)
        else:
            return self.conf
