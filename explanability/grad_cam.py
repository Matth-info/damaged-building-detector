# Grad-CAM implementation 
import torch
import torch.nn.functional as F
import numpy as np
import cv2

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
        heatmap = cv2.resize(heatmap, (input_tensor.size(3), input_tensor.size(2)))  # Resize to match input image
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

# To visualize the heatmap:
# img = input_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
# heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
# superimposed_img = heatmap * 0.4 + img
# cv2.imshow("Grad-CAM", superimposed_img)