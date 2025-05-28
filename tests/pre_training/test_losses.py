# pytest ./test_losses.py -v
import pytest
import torch

from src.losses import (
    BINARY_MODE,
    MULTICLASS_MODE,
    MULTILABEL_MODE,
    DiceLoss,
    Ensemble,
    FocalLoss,
    JaccardLoss,
    LovaszLoss,
    MCCLoss,
    SoftBCEWithLogitsLoss,
    SoftCrossEntropyLoss,
    TverskyLoss,
)
from src.losses.ordinal import (
    O2_Loss,
    Ordinal_CrossEntropy,
    Weighted_Categorical_CrossEntropy,
)

# Sample batch size, classes, and image size
BATCH_SIZE = 4
NUM_CLASSES = 3
IMG_SIZE = (224, 224)


@pytest.mark.parametrize(
    "loss_fn, mode",
    [
        (JaccardLoss(mode=BINARY_MODE), BINARY_MODE),
        (JaccardLoss(mode=MULTICLASS_MODE, classes=NUM_CLASSES), MULTICLASS_MODE),
        (DiceLoss(mode=BINARY_MODE), BINARY_MODE),
        (DiceLoss(mode=MULTICLASS_MODE, classes=NUM_CLASSES), MULTICLASS_MODE),
        (FocalLoss(mode=BINARY_MODE), BINARY_MODE),
        (FocalLoss(mode=MULTICLASS_MODE), MULTICLASS_MODE),
        (LovaszLoss(mode=BINARY_MODE), BINARY_MODE),
        (LovaszLoss(mode=MULTICLASS_MODE), MULTICLASS_MODE),
        (SoftBCEWithLogitsLoss(), BINARY_MODE),
        (SoftBCEWithLogitsLoss(), MULTILABEL_MODE),
        (SoftCrossEntropyLoss(smooth_factor=0.1), MULTICLASS_MODE),
        (TverskyLoss(mode=BINARY_MODE), BINARY_MODE),
        (TverskyLoss(mode=MULTICLASS_MODE, classes=NUM_CLASSES), MULTICLASS_MODE),
        (MCCLoss(), BINARY_MODE),
    ],
)
def test_loss_computation(loss_fn, mode):
    """Check if loss functions return a valid positive loss value."""

    # Create dummy logits and targets based on mode
    if mode == BINARY_MODE:
        logits = torch.randn(size=(BATCH_SIZE, 1, *IMG_SIZE), requires_grad=True)  # (N, 1, H, W)
        targets = torch.randint(0, 2, size=(BATCH_SIZE, 1, *IMG_SIZE)).float()  # (N, 1, H, W)
    elif mode == MULTICLASS_MODE:
        logits = torch.randn(
            size=(BATCH_SIZE, NUM_CLASSES, *IMG_SIZE), requires_grad=True
        )  # (N, C, H, W)
        targets = torch.randint(0, NUM_CLASSES, size=(BATCH_SIZE, *IMG_SIZE))  # (N, H, W)
    else:  # MULTILABEL_MODE
        logits = torch.randn(
            size=(BATCH_SIZE, NUM_CLASSES, *IMG_SIZE), requires_grad=True
        )  # (N, C, H, W)
        targets = torch.randint(
            0, NUM_CLASSES, size=(BATCH_SIZE, NUM_CLASSES, *IMG_SIZE)
        ).float()  # (N, C, H, W)

    # Compute loss
    loss = loss_fn(logits, targets)

    # Assertions
    assert torch.isfinite(loss).all(), f"Loss is not finite: {loss}"
    assert loss.item() >= 0, f"Loss should be positive but got {loss}"

    # Check if gradients can be computed
    loss.backward()
    assert logits.grad is not None, "Gradients are not computed!"


@pytest.mark.parametrize(
    "loss_fn",
    [
        Ensemble(
            [JaccardLoss(mode=MULTICLASS_MODE), DiceLoss(mode=MULTICLASS_MODE)],
            weights=[0.5, 0.5],
        )
    ],
)
def test_ensemble_loss(loss_fn):
    """Ensure that Ensemble loss properly combines multiple loss functions."""
    logits = torch.randn(
        size=(BATCH_SIZE, NUM_CLASSES, *IMG_SIZE), requires_grad=True
    )  # (N, C, H, W)
    targets = torch.randint(0, NUM_CLASSES, size=(BATCH_SIZE, *IMG_SIZE))  # (N, H, W)

    loss = loss_fn(logits, targets)
    assert loss.item() >= 0, "Ensemble loss should be positive"

    loss.backward()
    assert logits.grad is not None, "Gradients are not computed for Ensemble loss!"


def test_weighted_categorical_crossentropy_forward():
    loss_fn = Weighted_Categorical_CrossEntropy()
    y_pred = torch.randn(2, 4, 8, 8, requires_grad=True)  # (B, C, H, W)
    y_true = torch.randint(0, 4, (2, 8, 8))  # (B, H, W)
    loss = loss_fn(y_pred, y_true)
    assert loss.dim() == 0
    loss.backward()  # Should not raise


def test_ordinal_crossentropy_forward():
    num_classes = 4
    loss_fn = Ordinal_CrossEntropy(reduction="sum", num_classes=num_classes)
    y_pred = torch.randn(2, num_classes, 8, 8, requires_grad=True)  # (B, C, H, W)
    y_true = torch.randint(0, num_classes, (2, 8, 8))  # (B, H, W)
    # Patch convert_to_ordinal_labels to use correct num_classes
    loss = loss_fn(y_pred, y_true)
    assert loss.shape == torch.Size([])  # should be a scalar
    loss.backward()  # Should not raise


def test_o2_loss_forward():
    loss_fn = O2_Loss()
    y_pred = torch.randn(2, 4, 8, 8, requires_grad=True)  # (B, C, H, W)
    y_true = torch.randint(0, 4, (2, 8, 8))  # (B, H, W)
    loss = loss_fn(y_pred, y_true)
    assert loss.dim() == 0
    loss.backward()  # Should not raise
