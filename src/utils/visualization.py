import matplotlib.pyplot as plt
import numpy as np
from typing import List
import torch
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
import os
import numpy as np
import matplotlib.pyplot as plt
import torch

def renormalize_image(image : torch.Tensor | np.ndarray , mean : list[float] | tuple[float] , std : list[float] | tuple[float], device: str):
    """
    Renormalizes an image tensor or NumPy array by reversing the normalization process.

    Args:
        image (torch.Tensor or np.ndarray): Normalized image (C, H, W) with values in range ~N(0,1).
        mean (list or tuple): Mean values used for normalization (per channel).
        std (list or tuple): Standard deviation values used for normalization (per channel).
        device (str): 'cpu' or 'cuda' 
    Returns:
        torch.Tensor or np.ndarray: Renormalized image with pixel values in range [0, 1].
    """
    if isinstance(image, torch.Tensor):
        mean = torch.tensor(mean).view(-1, 1, 1).to(device)  # Shape (C, 1, 1)
        std = torch.tensor(std).view(-1, 1, 1).to(device)    # Shape (C, 1, 1)
        renormalized_image = image * std + mean
        return torch.clamp(renormalized_image, 0, 1)  # Keep values in [0, 1]

    elif isinstance(image, np.ndarray):
        mean = np.array(mean).reshape(-1, 1, 1)  # Shape (C, 1, 1)
        std = np.array(std).reshape(-1, 1, 1)    # Shape (C, 1, 1)
        renormalized_image = image * std + mean
        return np.clip(renormalized_image, 0, 1)  # Keep values in [0, 1]
    
    else:
        raise TypeError("Input should be a torch.Tensor or a np.ndarray")

def display_semantic_predictions_batch(images, 
                                       mask_predictions,
                                       mask_labels, 
                                       normalized=None,
                                       folder_path=None):
    """
    Displays a batch of images alongside their predicted masks and ground truth masks,
    and optionally saves the images in a folder.
    
    Args:
        images (torch.Tensor or numpy.ndarray): Batch of input images, shape (N, C, H, W) or (N, H, W, C).
        mask_predictions (torch.Tensor or numpy.ndarray): Batch of predicted masks, shape (N, H, W) or (N, H, W, C).
        mask_labels (torch.Tensor or numpy.ndarray): Batch of ground truth masks, shape (N, H, W) or (N, H, W, C).
        normalized (dict): Dictionary containing "mean" and "std" for unnormalization.
        folder_path (str): Path to the folder where images will be saved. If None, images are not saved.
    """
    
    if normalized is None:
        bool_normalized = False 
    else:
        bool_normalized = True 
        mean = normalized.get("mean")
        std = normalized.get("std")

    # Convert tensors to numpy arrays if needed
    if isinstance(images, torch.Tensor):
        images = images.detach().cpu().numpy()
    if isinstance(mask_predictions, torch.Tensor):
        mask_predictions = mask_predictions.detach().cpu().numpy()
    if isinstance(mask_labels, torch.Tensor):
        mask_labels = mask_labels.detach().cpu().numpy()
    
    batch_size = images.shape[0]  # Number of images in the batch

    # Create the folder if folder_path is provided
    if folder_path is not None:
        os.makedirs(folder_path, exist_ok=True)

    for i in range(batch_size):
        image = images[i]
        mask_prediction = mask_predictions[i]
        mask_label = mask_labels[i]
        
        # Handle channel-first images (C, H, W) and unnormalize
        if image.ndim == 3 and image.shape[0] == 3:  # RGB images
            image = np.transpose(image, (1, 2, 0))  # Convert to (H, W, C)
            image = renormalize_image(image, std, mean) if bool_normalized else image  # Unnormalize and clip to [0, 1]
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
        plt.imshow(image, alpha=0.5)
        plt.imshow(mask_prediction, cmap='jet', interpolation='none', alpha=0.5)
        plt.axis('off')
        plt.title("Predicted Mask")
        
        # Show the ground truth mask
        plt.subplot(1, 3, 3)
        plt.imshow(image, alpha=0.5)
        plt.imshow(mask_label, cmap='jet', interpolation='none', alpha=0.5)
        plt.axis('off')
        plt.title("Ground Truth Mask")
        
        plt.tight_layout()

        # Save the plot if folder_path is provided
        if folder_path is not None:
            file_path = os.path.join(folder_path, f"sample_{i + 1}.png")
            plt.savefig(file_path)
            print(f"Saved: {file_path}")
        
        plt.show()
    

def display_instance_predictions_batch(images, mask_predictions, score_threshold=0.6, max_images=5, display: List[str]=None):
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

    # Loop through the data loader (for visualization, we can stop after a few images)

    # Make predictions
    predictions = mask_predictions[:max_images]  # Predictions are a list of dicts with "boxes", "labels", "scores", "masks"

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
        plt.figure(figsize=(12, 4))
        plt.imshow(output_image.permute(1, 2, 0)) 
        plt.axis('off')
        plt.show()


def display_semantic_siamese_predictions_batch(
    pre_images,
    post_images,
    mask_predictions,
    mask_labels,
    normalized=None,
    folder_path=None
):
    """
    Displays a batch of pre-event and post-event images alongside their predicted masks 
    and ground truth masks, and optionally saves the images in a folder.
    
    Args:
        pre_images (torch.Tensor or numpy.ndarray): Batch of pre-event images, shape (N, C, H, W) or (N, H, W, C).
        post_images (torch.Tensor or numpy.ndarray): Batch of post-event images, shape (N, C, H, W) or (N, H, W, C).
        mask_predictions (torch.Tensor or numpy.ndarray): Batch of predicted masks, shape (N, H, W) or (N, H, W, C).
        mask_labels (torch.Tensor or numpy.ndarray): Batch of ground truth masks, shape (N, H, W) or (N, H, W, C).
        normalized (dict): Dictionary containing "mean" and "std" for unnormalization.
        folder_path (str): Path to the folder where images will be saved. If None, images are not saved.
    """
    
    # Determine normalization parameters
    bool_normalized = normalized is not None
    mean, std = normalized.get("mean"), normalized.get("std") if bool_normalized else (0, 1)

    # Convert tensors to numpy arrays if needed
    if isinstance(pre_images, torch.Tensor):
        pre_images = pre_images.detach().cpu().numpy()
        post_images = post_images.detach().cpu().numpy()
    if isinstance(mask_predictions, torch.Tensor):
        mask_predictions = mask_predictions.detach().cpu().numpy()
    if isinstance(mask_labels, torch.Tensor):
        mask_labels = mask_labels.detach().cpu().numpy()

    batch_size = pre_images.shape[0]  # Number of samples in the batch

    # Create the output folder if folder_path is provided
    if folder_path is not None:
        os.makedirs(folder_path, exist_ok=True)

    for i in range(batch_size):
        pre_image = pre_images[i]
        post_image = post_images[i]
        mask_prediction = mask_predictions[i]
        mask_label = mask_labels[i]

        # Handle channel-first images (C, H, W) and unnormalize if needed
        def process_image(image):
            if image.ndim == 3 and image.shape[0] == 3:  # RGB
                image = np.transpose(image, (1, 2, 0))  # Convert to (H, W, C)
                return renormalize_image(image, mean, std) if bool_normalized else image
            elif image.ndim == 3 and image.shape[0] == 1:  # Grayscale
                return image[0]  # Remove channel dimension
            return image

        pre_image = process_image(pre_image)
        post_image = process_image(post_image)

        # Create the plot
        plt.figure(figsize=(16, 8))

        # Display pre-event image
        plt.subplot(1, 4, 1)
        plt.imshow(pre_image)
        plt.axis('off')
        plt.title("Pre-Event Image")

        # Display post-event image
        plt.subplot(1, 4, 2)
        plt.imshow(post_image)
        plt.axis('off')
        plt.title("Post-Event Image")

        # Show the predicted mask
        plt.subplot(1, 4, 3)
        plt.imshow(mask_prediction, cmap='jet', interpolation='none')
        plt.axis('off')
        plt.title("Predicted Mask")

        # Show the ground truth mask
        plt.subplot(1, 4, 4)
        plt.imshow(mask_label, cmap='jet', interpolation='none')
        plt.axis('off')
        plt.title("Ground Truth Mask")

        plt.tight_layout()

        # Save the plot if folder_path is provided
        if folder_path is not None:
            file_path = os.path.join(folder_path, f"sample_{i + 1}.png")
            plt.savefig(file_path)
            print(f"Saved: {file_path}")

        plt.show()