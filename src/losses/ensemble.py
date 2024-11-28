import torch
from torch.nn.modules.loss import _Loss
from typing import List 

class Ensemble(_Loss):
    def __init__(
        self,
        list_losses: List[_Loss] = [],
        weights: List[float] = []
    ):
        super().__init__()
        self.list_losses = list_losses
        self.weights = weights

        assert len(list_losses) > 0, "List of losses cannot be empty."
        assert len(weights) == len(list_losses), "Weights must match the number of losses."

        # Normalize weights to sum to 1
        total_weight = sum(self.weights)
        assert total_weight > 0, "Weights must sum to a positive value."
        self.weights = [w / total_weight for w in self.weights]

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
