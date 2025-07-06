"""Overall utils functions."""

from typing import Literal

from .logger import BaseLogger
from .mlflow_utils import MLflowLogger
from .tensorboard_utils import TensorboardLogger
from .visualization import (
    display_instance_predictions_batch,
    display_semantic_is_siamese_predictions_batch,
    display_semantic_predictions_batch,
)


def get_logger(logger_type: Literal["mlflow", "tensorboard"]) -> BaseLogger:
    """Get logger between mlflow and tensorboard."""
    match logger_type:
        case "mlflow":
            return MLflowLogger
        case "tensorboard":
            return TensorboardLogger
        case _:
            raise ValueError("Logger type %s is not supported.", logger_type)
