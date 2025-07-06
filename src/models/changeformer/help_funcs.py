import warnings
from torch import nn
from torch import Tensor
import torch.functional as F


def resize(
    input,
    size=None,
    scale_factor=None,
    mode="nearest",
    align_corners=None,
    warning=True,
) -> Tensor:
    """Resize the input tensor using the specified size, scale factor, mode, and align_corners option.

    Args:
        input (Tensor): The input tensor to be resized.
        size (tuple or None): Output spatial size.
        scale_factor (float or None): Multiplier for spatial size.
        mode (str): Algorithm used for upsampling.
        align_corners (bool or None): Geometrically, align the corners of input and output tensors.
        warning (bool): Whether to show a warning if align_corners is set and size is not aligned.

    Returns:
        Tensor: Resized tensor.
    """
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if (
                    (output_h > 1 and output_w > 1 and input_h > 1 and input_w > 1)
                    and (output_h - 1) % (input_h - 1)
                    and (output_w - 1) % (input_w - 1)
                ):
                    warnings.warn(
                        f"When align_corners={align_corners}, "
                        "the output would more aligned if "
                        f"input size {(input_h, input_w)} is `x+1` and "
                        f"out size {(output_h, output_w)} is `nx+1`",
                        stacklevel=2,
                    )
    return F.interpolate(input, size, scale_factor, mode, align_corners)


# Difference module
def conv_diff(in_channels, out_channels) -> nn.Module:
    """Create a sequential module for computing the difference between feature maps using convolutional layers.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.

    Returns:
        nn.Module: A sequential module with convolution, ReLU, and batch normalization layers.
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(out_channels),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(),
    )


# Intermediate prediction module
def make_prediction(in_channels, out_channels) -> nn.Module:
    """Create a sequential module for intermediate prediction using convolutional layers.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.

    Returns:
        nn.Module: A sequential module with convolution, ReLU, and batch normalization layers.
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(out_channels),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
    )
