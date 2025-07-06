"""Metric implementations from https://github.com/qubvel-org/segmentation_models.pytorch."""

from .confusionmatrix import ConfusionMatrix
from .functional import (
    accuracy,
    balanced_accuracy,
    f1_score,
    false_discovery_rate,
    false_negative_rate,
    false_omission_rate,
    false_positive_rate,
    fbeta_score,
    get_stats,
    iou_score,
    negative_likelihood_ratio,
    negative_predictive_value,
    positive_likelihood_ratio,
    positive_predictive_value,
    precision,
    recall,
    sensitivity,
    specificity,
)
from .iou import IoU

METRICS_MAP = {
    "accuracy": accuracy,
    "balanced_accuracy": balanced_accuracy,
    "fbeta_score": fbeta_score,
    "f1_score": f1_score,
    "iou_score": iou_score,
    "precision": precision,
    "recall": recall,
    "sensitivity": sensitivity,
    "specificity": specificity,
    "positive_predictive_value": positive_predictive_value,
    "negative_predictive_value": negative_predictive_value,
    "false_negative_rate": false_negative_rate,
    "false_positive_rate": false_positive_rate,
    "false_discovery_rate": false_discovery_rate,
    "false_omission_rate": false_omission_rate,
    "positive_likelihood_ratio": positive_likelihood_ratio,
    "negative_likelihood_ratio": negative_likelihood_ratio,
}
