import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


class GradCAM:
    def __init__(self, model, target_layer):
        """
        Initialize the Grad-CAM object with the model and target layer.

        Args:
            model (torch.nn.Module): Pretrained model for which Grad-CAM is to be computed.
            target_layer (str): Name of the convolutional layer to be used for Grad-CAM.
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_layers()

    def hook_layers(self):
        """
        Set hooks to capture gradients and activations from the target layer.
        """

        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        # Register hooks to the target layer
        for name, module in self.model.named_modules():
            if name == self.target_layer:
                module.register_forward_hook(forward_hook)
                module.register_backward_hook(backward_hook)

    def generate_cam(self, input_tensor, target_class=None):
        """
        Generate Grad-CAM heatmap for the specified class.

        Args:
            input_tensor (torch.Tensor): Input image tensor, expected to be normalized and shaped as (1, C, H, W).
            target_class (int): Target class for which Grad-CAM is computed. If None, uses the predicted class.

        Returns:
            np.ndarray: Heatmap of the same spatial size as the target convolution layer's activations.
        """
        # Forward pass
        output = self.model(input_tensor)

        # If target_class is None, use the predicted class
        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # Backward pass for the target class
        self.model.zero_grad()
        output[:, target_class].backward()

        # Get pooled gradients and activations
        pooled_gradients = torch.mean(self.gradients, dim=(2, 3))
        activations = self.activations[0]

        # Weight each channel in the activations by the corresponding gradients
        for i in range(pooled_gradients.size(0)):
            activations[i, :, :] *= pooled_gradients[i]

        # Compute the heatmap and apply ReLU
        heatmap = torch.sum(activations, dim=0).cpu().detach().numpy()
        heatmap = np.maximum(heatmap, 0)  # Apply ReLU

        # Normalize the heatmap to range [0, 1]
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

        # Resize the heatmap to match input image size
        heatmap = np.resize(
            heatmap, (input_tensor.size(2), input_tensor.size(3))
        )  # Resize to match input image
        return heatmap

    def __call__(self, input_tensor, target_class=None):
        """
        Callable method to compute Grad-CAM heatmap.

        Args:
            input_tensor (torch.Tensor): Input image tensor.
            target_class (int): Target class (optional).

        Returns:
            np.ndarray: Heatmap array.
        """
        return self.generate_cam(input_tensor, target_class)


# Example usage:
# model = <Your convolutional model here>
# grad_cam = GradCAM(model, target_layer='layer_name')

# input_tensor = <Your input tensor here, shaped as (1, C, H, W)>
# heatmap = grad_cam(input_tensor)

# To visualize the heatmap using matplotlib:


def display_gradcam(input_tensor, heatmap):
    """
    Function to display Grad-CAM using matplotlib.

    Args:
        input_tensor (torch.Tensor): The input image tensor.
        heatmap (np.ndarray): The generated Grad-CAM heatmap.
    """
    # Convert the input tensor to an image (from tensor format to numpy format)
    img = input_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    img -= img.min()
    img /= img.max()  # Normalize image for better visualization

    # Create a colormap for the heatmap
    heatmap = np.uint8(255 * heatmap)  # Convert to 0-255 range
    colormap = plt.get_cmap("jet")
    heatmap_colored = colormap(heatmap)[:, :, :3]  # Use the jet colormap

    # Superimpose the heatmap on the original image
    superimposed_img = (
        heatmap_colored * 0.4 + img
    )  # Adjust transparency by multiplying by 0.4

    # Plot the original image and the superimposed heatmap
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(superimposed_img)
    plt.title("Grad-CAM Heatmap")
    plt.axis("off")

    plt.show()


# Example of calling the `display_gradcam` function
# Assuming `input_tensor` is the input image tensor and `heatmap` is generated
# from the Grad-CAM class:

# display_gradcam(input_tensor, heatmap)
