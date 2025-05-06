import torch
import torch.nn as nn


class EnsembleModel(nn.Module):
    def __init__(self, models, aggregation="mean", is_siamese=False):
        """
        Ensemble of models, supporting both Siamese and non-Siamese networks.

        Args:
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

    def forward(self, *inputs):
        """
        Forward pass through the ensemble.

        Args:
            *inputs (torch.Tensor): Input tensor(s). Single input for non-Siamese networks,
                                    or two inputs (x1, x2) for Siamese networks.

        Returns:
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
        elif self.aggregation == "vote":
            return torch.mode(outputs.argmax(dim=-1), dim=0)[
                0
            ]  # Majority vote on class predictions

    @torch.no_grad()
    def predict(self, *inputs):
        self.forward(*inputs)

    def test_compatibility(self, input_shape, siamese_input_shape=None):
        """
        Test compatibility of all models in the ensemble.

        Args:
            input_shape (tuple): Expected input shape for non-Siamese networks.
            siamese_input_shape (tuple): Expected input shape for Siamese networks.

        Returns:
            bool: True if all models are compatible, False otherwise.
        """
        try:
            if self.is_siamese:
                if siamese_input_shape is None:
                    raise ValueError("Siamese networks require a siamese_input_shape.")
                dummy_input1 = torch.rand(siamese_input_shape)
                dummy_input2 = torch.rand(siamese_input_shape)
                outputs = [model(dummy_input1, dummy_input2) for model in self.models]
            else:
                dummy_input = torch.rand(input_shape)
                outputs = [model(dummy_input) for model in self.models]

            output_shapes = [output.shape for output in outputs]

            # Check that all outputs have the same shape
            if len(set(output_shapes)) != 1:
                raise ValueError(f"Inconsistent output shapes: {output_shapes}")

        except Exception as e:
            print(f"Compatibility test failed: {e}")
            return False

        print("All models are compatible!")
        return True
