from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Literal

from src.utils.visualization import DEFAULT_MAPPING

if TYPE_CHECKING:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader


class BaseLogger(ABC):
    """Base Class for Logger experiment."""

    @abstractmethod
    def log_hyperparams(self, *args: Any, **kwargs: Any):  # noqa: ANN401
        """Logging hyperparameters."""

    @abstractmethod
    def log_metrics(self, *args: Any, **kwargs: Any) -> None:  # noqa: ANN401
        """Logging metric values."""

    @abstractmethod
    def log_loss(self, *args: Any, **kwargs: Any) -> None:  # noqa: ANN401
        """Logging Loss value."""

    @abstractmethod
    def log_lr(self, *args: Any, **kwargs: Any) -> None:  # noqa: ANN401
        """Logging Learning Rate."""

    @abstractmethod
    def log_epoch_time(self, *args: Any, **kwargs: Any) -> None:  # noqa: ANN401
        """Logging Epoch Time."""

    @abstractmethod
    def log_model(
        self,
        *args: Any,  # noqa: ANN401
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        """Logging model."""

    @abstractmethod
    def log_model_architecture(
        self,
        *args: Any,  # noqa: ANN401
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        """Logging model architecture."""

    @abstractmethod
    def log_images(self, *args: Any, **kwargs: Any) -> None:  # noqa: ANN401
        """Logging images, predictions and ground truth mask."""
