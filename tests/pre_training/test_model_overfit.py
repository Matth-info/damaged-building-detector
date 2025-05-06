import cv2
import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.optim as optim

from src.models import BiT, ChangeFormer, ResNet_UNET, SiameseResNetUNet, TinyCD


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
    image_tensor = (
        torch.tensor(image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
    )  # (1, 3, H, W)
    label_tensor = torch.tensor(label, dtype=torch.long).unsqueeze(0)  # (1, H, W)

    return image_tensor, label_tensor


@pytest.mark.parametrize(
    "model_class, kwargs",
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
        ),
        (
            TinyCD,
            {
                "bkbn_name": "efficientnet_b4",
                "pretrained": False,
                "output_layer_bkbn": "3",
                "out_channels": 2,
                "freeze_backbone": False,
            },
        ),
        (
            BiT,
            {
                "input_nc": 3,
                "output_nc": 2,
                "with_pos": "learned",
                "resnet_stages_num": 4,
                "backbone": "resnet18",
            },
        ),
    ],
)
def test_model_overfit_on_single_sample(model_class, kwargs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_class(**kwargs).to(device).train()

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Generate a single sample and label
    input_shape = (1, 3, 224, 224)

    inputs, target = draw_circle_with_label(
        image_size=input_shape[-1], radius=int(input_shape[-1] / 4), center=None
    )

    # Send inputs, targets and criterion to device
    inputs, target = inputs.to(device), target.to(device)
    # For Siamese models, duplicate input
    if model_class in [SiameseResNetUNet, BiT, TinyCD, ChangeFormer]:

        def forward_pass():
            return model(x1=inputs, x2=inputs)

    else:

        def forward_pass():
            return model(inputs)

    # Train for a few iterations
    for _ in range(100):  # More iterations ensure memorization
        optimizer.zero_grad()
        outputs = forward_pass()

        # Reshape output for CrossEntropyLoss
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

    # Final forward pass
    outputs = forward_pass()
    predicted = torch.argmax(outputs, dim=1)  # Get predicted class indices

    # Check if the model perfectly memorized the single sample
    overfit = (predicted == target).float().mean() >= 0.9
    assert overfit, "Model failed to overfit the single sample."
