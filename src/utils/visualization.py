import matplotlib.pyplot as plt
import numpy as np
from typing import List
import torch
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks

def unormalize_images(image, std=np.array([0.229, 0.224, 0.225]), mean=np.array([0.485, 0.456, 0.406])):
    return (image * std + mean).clip(0, 1)

def display_semantic_predictions_batch(images, mask_predictions, mask_labels, normalized = True):
    """
    Displays a batch of images alongside their predicted masks and ground truth masks,
    with images unnormalized using ImageNet mean and std.

    Args:
        images (torch.Tensor or numpy.ndarray): Batch of input images, shape (N, C, H, W) or (N, H, W, C).
        mask_predictions (torch.Tensor or numpy.ndarray): Batch of predicted masks, shape (N, H, W) or (N, H, W, C).
        mask_labels (torch.Tensor or numpy.ndarray): Batch of ground truth masks, shape (N, H, W) or (N, H, W, C).
    """
    # ImageNet mean and std for unnormalization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

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
        
        # Handle channel-first images (C, H, W) and unnormalize
        if image.ndim == 3 and image.shape[0] == 3:  # RGB images
            image = np.transpose(image, (1, 2, 0))  # Convert to (H, W, C)
            image = unormalize_images(image) if normalized else image # Unnormalize and clip to [0, 1]
        elif image.ndim == 3 and image.shape[0] == 1:  # Grayscale
            image = image[0]  # Remove channel dimension

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

def display_instance_predictions_batch(model, batch, device='cuda', score_threshold=0.6, max_images=5, display: List[str]=None):
    """
    Visualizes predictions from the model on a given dataset.

    Parameters:
    - model: The model to use for inference.
    - batch: batch data = (images, tagets).
    - device: The device to run the inference on ('cuda' or 'cpu').
    - score_threshold: The threshold above which predictions are considered valid.
    - max_images: The number of images to display.
    """
    assert display is not None, "Display 'boxes' and/or 'masks"
    # Set model to evaluation mode
    model.eval()

    # Loop through the data loader (for visualization, we can stop after a few images)
    with torch.no_grad():
        (images, targets) = batch 
        images = images.to(device)

        # Make predictions
        predictions = model(images)[:max_images]  # Predictions are a list of dicts with "boxes", "labels", "scores", "masks"

        for i in range(len(predictions)):
            pred = predictions[i]
            image = images[i]

            # Normalize and convert to uint8
            output_image = (255.0 * (image - image.min()) / (image.max() - image.min())).to(torch.uint8)

            # Filter out low-confidence predictions 
            mask = pred["scores"] > score_threshold 
            pred_boxes = pred["boxes"][mask]
            pred_labels = [f"{score:.3f}" for score in pred["scores"][mask]]
            pred_masks = pred["masks"][mask]

            # Draw bounding boxes
            if "boxes" in display:
                output_image = draw_bounding_boxes(output_image, pred_boxes.long(), labels=pred_labels, colors="red")


            # Draw segmentation masks
            if "masks" in display:
                if pred_masks.numel() > 0:  # Ensure there are masks to draw
                    masks = (pred_masks > 0.5).squeeze(1)  # Binarize the masks
                    output_image = draw_segmentation_masks(output_image, masks, alpha=0.5, colors="blue")

            # Move the output image to the CPU and convert to NumPy array for plotting
            output_image = output_image.cpu()

            # Plot the image
            plt.figure(figsize=(8, 8))
            plt.imshow(output_image.permute(1, 2, 0)) 
            plt.axis('off')
            plt.show()