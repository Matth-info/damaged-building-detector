# pytest ./test_metrics.py -v
import pytest
import torch

from src.losses import BINARY_MODE, MULTICLASS_MODE, MULTILABEL_MODE
from src.metrics import (
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

# Sample test parameters
BATCH_SIZE = 4
NUM_CLASSES = 3
IMG_SIZE = (224, 224)


@pytest.mark.parametrize(
    "mode, num_classes",
    [
        (BINARY_MODE, 1),
        (MULTICLASS_MODE, NUM_CLASSES),
        (MULTILABEL_MODE, NUM_CLASSES),
    ],
)
@pytest.mark.parametrize(
    "metric",
    [
        fbeta_score,
        f1_score,
        iou_score,
        accuracy,
        precision,
        recall,
        sensitivity,
        specificity,
        balanced_accuracy,
        positive_predictive_value,
        negative_predictive_value,
        false_negative_rate,
        false_positive_rate,
        false_discovery_rate,
        false_omission_rate,
        positive_likelihood_ratio,
        negative_likelihood_ratio,
    ],
)
def test_metrics_computation(metric, mode, num_classes):
    """Ensure all metrics compute valid values based on confusion matrix components."""

    # Create dummy logits and targets based on mode
    if mode == BINARY_MODE:
        preds = torch.randint(0, 2, size=(BATCH_SIZE, 1, *IMG_SIZE)).long()  # (N, 1, H, W)
        targets = torch.randint(0, 2, size=(BATCH_SIZE, 1, *IMG_SIZE)).long()  # (N, 1, H, W)
    elif mode == MULTICLASS_MODE:
        logits = torch.randn(size=(BATCH_SIZE, NUM_CLASSES, *IMG_SIZE))  # (N, C, H, W)
        preds = logits.argmax(dim=1).long()  # (N, H, W)
        targets = torch.randint(0, NUM_CLASSES, size=(BATCH_SIZE, *IMG_SIZE)).long()  # (N, H, W)
    else:  # MULTILABEL_MODE
        preds = (
            torch.randint(0, NUM_CLASSES, size=(BATCH_SIZE, NUM_CLASSES, *IMG_SIZE)).round().long()
        )  # (N, C, H, W)
        targets = (
            torch.randint(0, NUM_CLASSES, size=(BATCH_SIZE, NUM_CLASSES, *IMG_SIZE)).round().long()
        )  # (N, C, H, W)

    # Compute confusion matrix components
    tp, fp, fn, tn = get_stats(preds, targets, mode=mode, num_classes=num_classes)

    # Compute metric
    metric_value = metric(tp, fp, fn, tn, class_weights=None, reduction="macro")

    # Assertions
    assert torch.isfinite(metric_value).all(), f"Metric {metric.__name__} returned NaN or Inf"
    assert metric_value.numel() > 0, f"Metric {metric.__name__} returned an empty tensor"
