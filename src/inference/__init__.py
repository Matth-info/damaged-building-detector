"""Focusing on Inference Only."""

from .batch_inference import (
    batch_inference,
    custom_infer_collate,
    custom_infer_collate_is_siamese,
)
from .functional import Inference

__all_ = [
    "Inference",
    "batch_inference",
    "custom_infer_collate_is_siamese",
    "custom_infer_collate",
]
