# Utils files for Dataset definitions 
import torch 


__all__ = ["custom_collate_fn"]

def custom_collate_fn(batch):
    """
    Custom collate function to handle variable-sized targets in a batch.
    Required for Instance Segmentation Dataloader 
    
    Parameters:
        batch (list): List of (image, target) tuples.
    
    Returns:
        Tuple: (images, targets)
    """
    images = []
    targets = []

    for img, target in batch:
        images.append(img)
        targets.append(target)

    # Stack images into a single batch tensor
    images = torch.stack(images, dim=0)

    # Return images and list of targets
    return images, targets

