import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
from transformers import (
    AutoConfig,
    AutoModelForSemanticSegmentation,
    AutoImageProcessor,
)
import numpy as np

def extract_dimension(image):
    if isinstance(image, torch.Tensor) or isinstance(image, np.ndarray):
        return image.shape[-2:]  # Returns (height, width)
    else:
        return (image.height, image.width)

class Segformer(nn.Module):
    """ 
    A Wrapper Torch module to load Semantic Segmentation models from HuggingFace Hub.
    """
    def __init__(self, 
                 model_name: str = 'nvidia/segformer-b1-finetuned-ade-512-512', 
                 label2id: Optional[Dict[str, int]] = None, 
                 num_labels: int = 2,
                 freeze_encoder = True
                 ):
        super(Segformer, self).__init__()
        
        # Load configuration with label2id if provided, ensuring correct label alignment.
        self.config = AutoConfig.from_pretrained(
            model_name,
            trust_remote_code=True,
            num_labels=num_labels,
            label2id=label2id or {}
        )
        
        # Load the semantic segmentation model
        self.model = AutoModelForSemanticSegmentation.from_pretrained(
            model_name,
            config=self.config,
            trust_remote_code=True,
            ignore_mismatched_sizes=True
        )
        
        # Load image processor with appropriate options for rescaling and label reduction
        self.image_processor = AutoImageProcessor.from_pretrained(
            model_name,
            do_reduce_labels=False,
            do_rescale=False,
            trust_remote_code=True
        )
        self.freeze_encoder = freeze_encoder
        if self.freeze_encoder:
            self._freeze_encoder()

    def _freeze_encoder(self):
        for param in self.model.segformer.parameters():
            param.requires_grad = False
        print("Encoder weights have been frozen.")
        
   
    def forward(self, image):
        # Process input image to match model requirements

        original_size = extract_dimension(image)

        inputs = self.image_processor(images=image, return_tensors="pt")
        
        # Forward pass through the model
        outputs = self.model(**inputs)
        
        logits = F.interpolate(outputs.logits, size=original_size, mode='bilinear', align_corners=False)

        return logits
    
    @torch.no_grad()
    def predict(self, image):
        """
        Predicts the segmentation mask for a given input image.
        
        Args:
            image: Input image as a PIL.Image or torch.Tensor (H, W, C) in RGB format.
        
        Returns:
            pred_seg: Predicted segmentation mask as a numpy array (same size than image).
        """

        original_size = extract_dimension(image) 

        inputs = self.image_processor(images=image, return_tensors="pt")
        
        # Get model output
        outputs = self.model(**inputs)

        # Scale logits to the size of the labels
        predicted_masks  = nn.functional.interpolate(
            outputs.logits,
            size=original_size,  # Match label dimensions
            mode="bilinear",
            align_corners=False,
        )
        
        return torch.argmax(predicted_masks, dim=1).cpu().numpy()

