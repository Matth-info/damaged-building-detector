import torch
from torch import nn
from typing import Literal


class EnsembleModel(nn.Module):
    """EnsembleModel is a PyTorch module that aggregates predictions from multiple models using mean or voting, supporting both Siamese and non-Siamese architectures."""

    def __init__(
        self,
        models: list[nn.Module],
        aggregation: Literal["vote", "mean"] = "mean",
        is_siamese=False,
    ) -> None:
        """Ensemble of models, supporting both Siamese and non-Siamese networks.

        Args:
        ----
            models (list of nn.Module): List of PyTorch models to be used in the ensemble.
            aggregation (str): Aggregation method to combine model outputs.
                               Options: "mean", "vote". Default is "mean".
            is_siamese (bool): Whether the models in the ensemble are Siamese networks.

        """
        super().__init__()
        if not all(isinstance(model, nn.Module) for model in models):
            raise TypeError("All models should inherit from nn.Module.")
        if aggregation not in ["mean", "vote"]:
            raise ValueError("Aggregation must be either 'mean' or 'vote'.")

        self.models = nn.ModuleList(models)
        self.aggregation = aggregation
        self.is_siamese = is_siamese

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass through the ensemble.

        Args:
        ----
            *inputs (torch.Tensor): Input tensor(s). Single input for non-Siamese networks,
                                    or two inputs (x1, x2) for Siamese networks.

        Returns:
        -------
            torch.Tensor: Aggregated output from the ensemble.

        """
        if self.is_siamese:
            if len(inputs) != 2:
                raise ValueError("Siamese networks require two input tensors (x1, x2).")
            x1, x2 = inputs
            outputs = torch.stack([model(x1, x2) for model in self.models], dim=0)
        else:
            if len(inputs) != 1:
                raise ValueError("Non-Siamese networks require a single input tensor.")
            x = inputs[0]
            outputs = torch.stack([model(x) for model in self.models], dim=0)

        if self.aggregation == "mean":
            return torch.mean(outputs, dim=0)  # Average predictions
        if self.aggregation == "vote":
            return torch.mode(outputs.argmax(dim=1), dim=0)[
                0
            ]  # Majority vote on class predictions
        return None

    @torch.no_grad()
    def predict(self, *inputs: torch.Tensor) -> torch.Tensor:
        """Inference mode."""
        self.forward(*inputs)
