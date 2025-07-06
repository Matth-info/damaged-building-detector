from typing import Any

import torch
import torch.distributions as td
from torch import nn


class ReshapeTransform(td.Transform):
    """A torch.distributions.Transform that reshapes tensors to a specified shape.

    This transform is useful for reshaping the output of distributions to match
    a desired event shape, especially in calibration models.
    """

    domain = td.constraints.real
    codomain = td.constraints.real
    bijective = True
    sign = +1

    def __init__(self, shape) -> None:  # noqa: D107
        super().__init__()
        self.shape = shape

    def __eq__(self, other):  # noqa: D105
        return isinstance(other, ReshapeTransform) and self.shape == other.shape

    def _call(self, x):
        return x.view(x.shape[:-1] + self.shape)

    def _inverse(self, y):
        return y.view(y.shape[: -len(self.shape)] + (-1,))

    def _log_abs_det_jacobian(self, x, y):
        return torch.zeros_like(x[..., 0])


# Temperature Scaling : post-training optimization of softmax scaling parameter.
class TemperatureScaling(nn.Module):
    """Apply temperature scaling to logits for calibration purposes.

    This class implements post-training optimization of a softmax scaling parameter
    to calibrate the output probabilities.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the TemperatureScaling module with a learnable temperature parameter.

        Parameters
        ----------
        *args : Any
            Additional positional arguments (unused).
        **kwargs : Any
            Additional keyword arguments (unused).

        """
        super().__init__()
        self.temperature_single = nn.Parameter(torch.ones(1))

    def weights_init(self):
        """Weight initialization."""
        self.temperature_single.data.fill_(1)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:  # noqa: D102
        temperature = self.temperature_single.expand(logits.size())
        return logits / temperature


# Vector Scaling : post-training optimization of class wise weights and bias optimization (before softmax)
class VectorScaling(nn.Module):
    """Apply vector scaling to logits for calibration purposes.

    This class implements post-training optimization of class-wise weights and bias
    (before softmax) to calibrate the output probabilities.

    Attributes:
    ----------
    vector_parameters : torch.nn.Parameter
        Learnable scaling parameters for each class.
    vector_offset : torch.nn.Parameter
        Learnable bias parameters for each class.

    Methods:
    -------
    weights_init()
        Initializes the scaling and bias parameters.
    forward(logits: torch.Tensor) -> torch.Tensor
        Applies vector scaling to the input logits.

    """

    def __init__(self, num_classes: int) -> None:
        """Initialize Vector Scaling layer."""
        super().__init__()
        self.vector_parameters = nn.Parameter(torch.ones(1, num_classes, 1, 1))
        self.vector_offset = nn.Parameter(torch.zeros(1, num_classes, 1, 1))

    def weights_init(self) -> None:
        """Weight initialization."""
        #        pass
        self.vector_offset.data.fill_(0)
        self.vector_parameters.data.fill_(1)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return logits * self.vector_parameters + self.vector_offset


class StochasticSpatialScaling(nn.Module):
    """Applies stochastic spatial scaling to logits for calibration purposes.

    This class introduces stochasticity in the scaling of logits using a learned convolution
    and samples from a transformed normal distribution for spatial calibration.

    Args:
        num_classes (int): Number of classes for the classification task.

    """

    def __init__(self, num_classes: int) -> None:  # noqa: D107
        super().__init__()
        self.num_classes = num_classes
        self.epsilon = 1e-5
        self.conv_logits = nn.Conv2d(num_classes, num_classes, kernel_size=1)
        self.rank = 10  # unused here

    @staticmethod
    def fixed_re_parametrization_trick(dist: td.Distribution, num_samples: int) -> torch.Tensor:
        """Generate symmetric samples from a distribution using the reparameterization trick.

        Parameters
        ----------
        dist : td.Distribution
            The distribution to sample from.
        num_samples : int
            The total number of samples to generate (must be even).

        Returns:
        -------
        torch.Tensor
            A tensor containing symmetric samples around the mean.

        Raises:
        ------
        ValueError
            If num_samples is not even.

        """
        if num_samples % 2 == 0:
            raise ValueError("num_samples must be even")
        samples = dist.rsample((num_samples // 2,))
        mean = dist.mean.unsqueeze(0)
        samples = samples - mean
        return torch.cat([samples, -samples]) + mean

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        batch_size = logits.shape[0]
        event_shape = (self.num_classes,) + logits.shape[2:]

        # Compute mean and covariance
        mean = self.conv_logits(logits)  # [B, C, H, W]
        cov_diag = (mean * 1e-5).exp() + self.epsilon

        # Flatten spatial and channel dims for the base distribution
        mean_flat = mean.view(batch_size, -1)
        cov_diag_flat = cov_diag.view(batch_size, -1)

        base_distribution = td.Independent(
            td.Normal(loc=mean_flat, scale=torch.sqrt(cov_diag_flat)),
            1,
        )

        # Transform distribution to have desired event shape
        transform = ReshapeTransform(event_shape)
        distribution = td.TransformedDistribution(base_distribution, [transform])

        # Sample using symmetric reparameterization trick
        num_samples = 2
        samples = distribution.rsample((num_samples // 2,))
        mean_sample = base_distribution.mean.view(batch_size, *event_shape).unsqueeze(0)
        samples = samples - mean_sample
        logit_samples = torch.cat([samples, -samples]) + mean_sample
        return logit_samples.mean(dim=0)


class DirichletScaling(nn.Module):
    """Applies Dirichlet scaling to logits for calibration purposes.

    This class implements a linear transformation on the log-probabilities of the input logits
    to calibrate the output probabilities using a Dirichlet-based approach.

    Args:
        num_classes (int): Number of classes for the classification task.
        kwargs (object, optional): Additional keyword arguments (unused).

    """

    def __init__(self, num_classes: int) -> None:
        """Initialize Dirichlet scaling.

        Args:
            num_classes (int): number of classes.
        """
        super().__init__()
        self.dirichlet_linear = nn.Linear(num_classes, num_classes)

    def weights_init(self) -> None:
        """Weight initialization where A is full of one and B is a zero vector."""
        self.dirichlet_linear.weight.data.copy_(torch.eye(self.dirichlet_linear.weight.shape[0]))
        self.dirichlet_linear.bias.data.copy_(torch.zeros(*self.dirichlet_linear.bias.shape))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        logits = logits.permute(0, 2, 3, 1)
        softmax = torch.nn.Softmax(dim=-1)
        probs = softmax(logits)
        ln_probs = torch.log(probs + 1e-10)

        return self.dirichlet_linear(ln_probs).permute(0, 3, 1, 2)


CALIBRATION_MAP = {
    "DirichletScaling": DirichletScaling,
    "TemperatureScaling": TemperatureScaling,
    "VectorScaling": VectorScaling,
    "StochasticSpatialScaling": StochasticSpatialScaling,
}
