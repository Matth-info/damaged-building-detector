import matplotlib.pyplot as plt
import numpy as np
import torch

def display_predictions_batch(images, mask_predictions, mask_labels):
    """
    Displays a batch of images alongside their predicted masks and ground truth masks.

    Args:
        images (torch.Tensor or numpy.ndarray): Batch of input images, shape (N, C, H, W) or (N, H, W, C).
        mask_predictions (torch.Tensor or numpy.ndarray): Batch of predicted masks, shape (N, H, W) or (N, H, W, C).
        mask_labels (torch.Tensor or numpy.ndarray): Batch of ground truth masks, shape (N, H, W) or (N, H, W, C).
    """
    # Convert tensors to numpy arrays if needed
    if isinstance(images, torch.Tensor):
        images = images.detach().cpu().numpy()
    if isinstance(mask_predictions, torch.Tensor):
        mask_predictions = mask_predictions.detach().cpu().numpy()
    if isinstance(mask_labels, torch.Tensor):
        mask_labels = mask_labels.detach().cpu().numpy()
    
    batch_size = images.shape[0]  # Number of images in the batch

    for i in range(batch_size):
        image = images[i]
        mask_prediction = mask_predictions[i]
        mask_label = mask_labels[i]
        
        # Handle grayscale or channel-first images
        if image.ndim == 3 and image.shape[0] in [1, 3]:  # (C, H, W) format
            image = np.transpose(image, (1, 2, 0))  # Convert to (H, W, C)
        
        # Normalize image for better visualization (if needed)
        if image.max() > 1:
            image = image / 255.0  # Assuming image is in [0, 255]
        
        # Create the plot
        plt.figure(figsize=(12, 4))
        
        # Show the input image
        plt.subplot(1, 3, 1)
        plt.imshow(image)
        plt.axis('off')
        plt.title("Input Image")
        
        # Show the predicted mask
        plt.subplot(1, 3, 2)
        plt.imshow(mask_prediction, cmap='jet', interpolation='none')
        plt.axis('off')
        plt.title("Predicted Mask")
        
        # Show the ground truth mask
        plt.subplot(1, 3, 3)
        plt.imshow(mask_label, cmap='jet', interpolation='none')
        plt.axis('off')
        plt.title("Ground Truth Mask")
        
        plt.tight_layout()
        plt.show()