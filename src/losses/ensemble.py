from typing import List, Optional

import torch
from torch.nn.modules.loss import _Loss


class Ensemble(_Loss):
    def __init__(
        self,
        list_losses: Optional[List[_Loss]] = None,
        weights: Optional[List[float]] = None,
    ):
        super().__init__()
        self.list_losses = list_losses or []
        self.weights = weights or []

        if not self.list_losses:
            raise ValueError("List of losses cannot be empty.")
        if len(self.weights) != len(self.list_losses):
            raise ValueError("Weights must match the number of losses.")

        # Normalize weights to sum to 1
        total_weight = sum(self.weights)
        if total_weight <= 0:
            raise ValueError("Weights must sum to a positive value.")
        self.weights = [w / total_weight for w in self.weights]

    def __repr__(self):
        return " / ".join(loss.__class__.__name__ for loss in self.list_losses)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Computes the weighted ensemble loss.

        :param inputs: Predictions from the model.
        :param targets: Ground truth labels.
        :return: Weighted ensemble loss.
        """
        total_loss = 0.0
        for weight, loss_fn in zip(self.weights, self.list_losses):
            loss = loss_fn(inputs, targets)
            total_loss += weight * loss

        return total_loss
