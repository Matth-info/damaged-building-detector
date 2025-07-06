"""Conformal Prediction for Semantic Segmentation.

This pytorch version implementation of Conformal Prediction for Semantic Segmentation is highly inspired by the deel-ai implementation available at https://github.com/deel-ai-papers/conformal-segmentation.
"""

from .conformal import Conformalizer, compute_losses_on_test, lambda_optimization
