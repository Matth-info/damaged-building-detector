from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal, Optional

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from src.metrics import get_stats, iou_score

from .calibration_models import CALIBRATION_MAP
from .utils import make_model_diagrams

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
_logger = logging.getLogger(__name__)

rng = np.random.default_rng(42)


class CalibrationTrainer:
    """Trainer class for calibrating neural network outputs using various calibration layers.

    Attributes:
    ----------
    model : nn.Module or None
        The neural network model to calibrate.
    calibration_layer : nn.Module
        The calibration layer applied to the model outputs.
    num_classes : int
        Number of output classes.
    train_dataloader : DataLoader
        DataLoader for training data.
    val_dataloader : DataLoader or None
        DataLoader for validation data.
    optimizer : torch.optim.Optimizer
        Optimizer for training the calibration layer.
    device : str
        Device to run computations on ('cuda' or 'cpu').
    max_epoch : int
        Number of training epochs.
    start_epoch : int
        Epoch to start training from.
    is_siamese : bool
        Whether the model is a siamese network.

    Methods:
    -------
    train()
        Trains the calibration layer.
    validate(epoch: int)
        Validates the calibration layer.
    assessing_calibration(n_samples: int = 10, n_bins: int = 10)
        Assesses the calibration performance.
    check_coherence()
        Checks dataloader and batch structure coherence.

    """

    def __init__(
        self,
        model: nn.Module | None = None,
        calibration_layer: Literal[
            "TemperatureScaling",
            "DirichletScaling",
            "VectorScaling",
            "StochasticSpatialScaling",
        ] = "TemperatureScaling",
        num_classes: int = 2,
        train_dataloader: DataLoader = None,
        val_dataloader: DataLoader | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        n_epoch: int = 10,
        start_epoch: int = 0,
        *,
        is_siamese: bool = False,
    ) -> None:
        """Initialize the CalibrationTrainer.

        Parameters
        ----------
        model : nn.Module or None
            The neural network model to calibrate.
        calibration_layer : str
            The calibration layer to use.
        num_classes : int
            Number of output classes.
        train_dataloader : DataLoader
            DataLoader for training data.
        val_dataloader : DataLoader or None
            DataLoader for validation data.
        optimizer : torch.optim.Optimizer or None
            Optimizer for training the calibration layer.
        device : str
            Device to run computations on ('cuda' or 'cpu').
        n_epoch : int
            Number of training epochs.
        start_epoch : int
            Epoch to start training from.
        is_siamese : bool
            Whether the model is a siamese network.

        """
        self.device = device
        if model is not None:
            self.model = model.to(device)
            self.model.eval()
        else:
            self.model = None

        self.calibration_layer = self._choose_calibration_layer(calibration_layer)(
            num_classes=num_classes,
        ).to(device)
        self.calibration_layer.weights_init()
        self.num_classes = num_classes

        self.loss_fn = nn.CrossEntropyLoss(ignore_index=255)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer or torch.optim.AdamW(
            self.calibration_layer.parameters(),
            lr=1e-3,
            weight_decay=1e-6,
        )

        self.max_epoch = n_epoch
        self.start_epoch = start_epoch or 0
        self.is_siamese = is_siamese

    def _choose_calibration_layer(self, name: str) -> nn.Module:
        if name not in CALIBRATION_MAP:
            msg = f"Calibration layer '{name}' not supported. Available: {list(CALIBRATION_MAP)}"
            raise ValueError(
                msg,
            )
        return CALIBRATION_MAP[name]

    def _get_logits(self, batch: dict) -> torch.Tensor:
        if self.is_siamese:
            if self.model:
                x1 = batch["pre_image"].to(self.device, non_blocking=True, dtype=torch.float32)
                x2 = batch["post_image"].to(self.device, non_blocking=True, dtype=torch.float32)
                return self.model(x1, x2)
            return batch["logits"].float().to(self.device, non_blocking=True, dtype=torch.float32)
        if self.model:
            inputs = batch["image"].to(self.device, non_blocking=True, dtype=torch.float32)
            return self.model(inputs)
        return batch["logits"].float().to(self.device, non_blocking=True, dtype=torch.float32)

    def check_coherence(self) -> None:
        """Check that the dataloaders and batch structure are coherent and contain required keys.

        Raises:
        ------
        ValueError
            If the train dataloader is not provided.
        KeyError
            If required keys are missing in the batch.

        """
        if self.train_dataloader is None:
            raise ValueError("Train dataloader is required")

        for loader in [self.train_dataloader, self.val_dataloader]:
            if loader is None:
                continue
            batch = next(iter(loader))
            if "mask" not in batch:
                raise KeyError("Label missing in batch")

            if self.model:
                if "image" not in batch:
                    raise KeyError("Image input required when using model")
            elif "logits" not in batch:
                raise KeyError("Logits required when no model is provided")

    def train(self) -> None:
        """Training method for calibration layer."""
        for epoch in range(self.start_epoch, self.max_epoch):
            self.calibration_layer.train()
            total_loss = 0.0
            with tqdm(
                self.train_dataloader,
                desc=f"Epoch {epoch + 1}/{self.max_epoch} - Training",
                unit="batch",
            ) as t:
                for i, batch in enumerate(t):
                    targets = batch["mask"].to(self.device, non_blocking=True, dtype=torch.int64)
                    logits = self._get_logits(batch)
                    calibrated_logits = self.calibration_layer(logits)

                    loss = self.loss_fn(calibrated_logits, targets)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    total_loss += loss.item()

                    if i % 10 == 0:
                        t.set_postfix({"Loss": loss.item()})

            avg_train_loss = total_loss / len(self.train_dataloader)
            _logger.info(
                "Epoch %d/%d Avg Train Loss: %.4f", epoch + 1, self.max_epoch, avg_train_loss
            )

            if self.val_dataloader:
                self.validate(epoch)

    def validate(self, epoch: int) -> None:
        """Run Validation method."""
        self.calibration_layer.eval()
        total_loss = 0.0

        with (
            torch.no_grad(),
            tqdm(
                self.val_dataloader,
                desc=f"Epoch {epoch + 1}/{self.max_epoch} - Validation",
                unit="batch",
            ) as t,
        ):
            for _i, batch in enumerate(t):
                targets = batch["mask"].to(self.device, non_blocking=True, dtype=torch.int64)
                logits = self._get_logits(batch)
                calibrated_logits = self.calibration_layer(logits)
                loss = self.loss_fn(calibrated_logits, targets)
                total_loss += loss.item()

        avg_loss = total_loss / len(self.val_dataloader)
        _logger.info("Epoch %d/%d Avg Validation Loss: %.4f", epoch + 1, self.max_epoch, avg_loss)

    def assessing_calibration(self, n_samples: int = 10, n_bins: int = 10) -> None:
        """Evaluate the addicational calibration layer performance (mIoU) and its calibration benefit compare to the base model.

        Args:
            n_samples (int, optional): Number of samples to consider for computing calibration metrics. Defaults to 10.
            n_bins (int, optional): Number of bins to display. Defaults to 10.

        """
        self.calibration_layer.eval()
        logits_list, cal_logits_list, labels_list = [], [], []

        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Collecting Calibration Data"):
                labels = batch["mask"].to(self.device, non_blocking=True, dtype=torch.int64)
                logits = self._get_logits(batch)
                calibrated_logits = self.calibration_layer(logits)

                logits_list.append(logits.cpu())
                cal_logits_list.append(calibrated_logits.cpu())
                labels_list.append(labels.cpu())

        logits_all = torch.cat(logits_list, dim=0)
        cal_logits_all = torch.cat(cal_logits_list, dim=0)
        labels_all = torch.cat(labels_list, dim=0)

        indices = rng.random.choice(
            len(logits_all),
            size=min(n_samples, len(logits_all)),
            replace=False,
        )

        sample_logits = logits_all[indices]
        sample_cal_logits = cal_logits_all[indices]
        sample_labels = labels_all[indices]

        # Evaluate performances
        preds = logits_all.argmax(dim=1).long()
        tp, fp, fn, tn = get_stats(
            output=preds,
            target=labels_all,
            mode="multiclass",
            num_classes=self.num_classes,
        )
        # Compute metric
        iou_value = iou_score(tp, fp, fn, tn, class_weights=None, reduction="macro")
        # Compute Calibration diagrams
        make_model_diagrams(
            outputs=sample_logits,
            labels=sample_labels,
            filename="reliability_diagram_no_calibration.png",
            n_bins=n_bins,
            info=f"UnCal (mIoU : {iou_value:.1f})",
        )

        # Evaluate performances
        preds = cal_logits_all.argmax(dim=1).long()
        tp, fp, fn, tn = get_stats(
            output=preds,
            target=labels_all,
            mode="multiclass",
            num_classes=self.num_classes,
        )
        # Compute metric
        iou_value = iou_score(tp, fp, fn, tn, class_weights=None, reduction="macro")
        make_model_diagrams(
            outputs=sample_cal_logits,
            labels=sample_labels,
            filename="reliability_diagram_with_calibration.png",
            n_bins=n_bins,
            info=f"Cal (mIoU : {iou_value:.1f})",
        )
