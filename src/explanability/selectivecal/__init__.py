"""A Calibration module for semantic segmentation.

Highly inspired by selectivecal implementation from https://github.com/dwang181/selectivecal
"""
# Highly inspired by selectivecal implementation
# https://github.com/dwang181/selectivecal

from .calibration_models import (
    CALIBRATION_MAP,
    DirichletScaling,
    StochasticSpatialScaling,
    TemperatureScaling,
    VectorScaling,
)
from .calibration_train import CalibrationTrainer
