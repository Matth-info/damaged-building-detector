# pytest test_trainer.py -v
import logging

import cv2
import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.metrics import f1_score, iou_score, precision, recall
from src.models import SiameseResNetUNet
from src.training import Trainer

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# Utils functions
def draw_circle_with_label(image_size=224, radius=50, center=None):
    """
    Draws a white circle on a black background and generates a 2-class label tensor.

    Args:
        image_size (int): The size of the square image (image_size x image_size).
        radius (int): Radius of the circle.
        center (tuple): Coordinates (x, y) of the circle's center. Defaults to image center.

    Returns:
        torch.Tensor: Image tensor of shape (1, 3, image_size, image_size)
        torch.Tensor: Label tensor of shape (1, image_size, image_size)
    """
    if center is None:
        center = (image_size // 2, image_size // 2)

    # Create a black RGB image
    image = np.zeros((image_size, image_size, 3), dtype=np.uint8)

    # Draw a white circle on all three channels
    cv2.circle(image, center, radius, (255, 255, 255), thickness=-1)

    # Create the label tensor
    label = np.zeros((image_size, image_size), dtype=np.uint8)
    grayscale_image = image[..., 0]  # Use one channel to create the label mask
    label = (grayscale_image > 0).astype(np.uint8)  # Circle footprint

    # Convert to torch tensors and normalize image
    image_tensor = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0  # (3, H, W)
    label_tensor = torch.tensor(label, dtype=torch.long)  # (H, W)

    return image_tensor, label_tensor


class RoundDataset(Dataset):
    def __init__(
        self,
        length: int = 64,
        siamese: bool = True,
        image_size: int = 224,
        radius: int = 50,
        center=None,
    ):
        self.length = length
        self.image, self.label = draw_circle_with_label(image_size, radius, center)
        self.siamese = siamese

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if self.siamese:
            return {"pre_image": self.image, "post_image": self.image, "mask": self.label}

        return {"image": self.image, "mask": self.label}


@pytest.mark.parametrize(
    "model_class, model_kwargs, dataset_cls, dataset_kwargs",
    [
        (
            SiameseResNetUNet,
            {
                "in_channels": 3,
                "out_channels": 2,
                "backbone_name": "resnet18",
                "pretrained": False,
                "mode": "conc",
            },
            RoundDataset,
            {"length": 5, "siamese": True, "image_size": 224, "radius": 50, "center": None},
        )
    ],
)
def test_trainer_module(model_class, model_kwargs, dataset_cls, dataset_kwargs):
    # Set up device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize dataset and dataloader
    dataset = dataset_cls(**dataset_kwargs)
    train_dl = DataLoader(dataset, batch_size=1, shuffle=True)

    # Initialize model
    model = model_class(**model_kwargs).to(device)

    # Define loss function, optimizer, and metrics
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = None
    metrics = [f1_score, iou_score, precision, recall]

    # Early stopping parameters
    early_stopping_params = {"patience": 3, "trigger_times": 0}

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        train_dl=train_dl,
        valid_dl=train_dl,
        test_dl=train_dl,
        optimizer=optimizer,
        scheduler=scheduler,
        nb_epochs=5,
        num_classes=model_kwargs["out_channels"],
        loss_fn=criterion,
        metrics=metrics,
        experiment_name="Trainer_Testing",
        siamese=dataset.siamese,
        is_mixed_precision=True,
        tta=False,
        training_log_interval=1,
        device=device,
        debug=False,
        checkpoint_interval=2,
        early_stopping_params=early_stopping_params,
        reduction="weighted",
        class_weights=[0.5, 0.5],
        class_names=["No_circle", "Circle"],
    )

    # Run training
    trainer.train()

    # Assertions to ensure training completes without errors
    assert trainer.get_hyperparameters() is not None, "Trainer give access to its hyperparameters"
    assert len(trainer.history["train_loss"]) > 0, "Training loss history should be available"
    assert len(trainer.history["val_loss"]) > 0, "Validation loss history should be available"
