from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoModelForSemanticSegmentation,
)


def extract_dimension(image):
    if isinstance(image, torch.Tensor) or isinstance(image, np.ndarray):
        return image.shape[-2:]  # Returns (height, width)
    else:
        return (image.height, image.width)


class Segformer(nn.Module):
    """
    A Wrapper Torch module to load Semantic Segmentation models from HuggingFace Hub.
    """

    def __init__(
        self,
        model_name: str = "nvidia/segformer-b1-finetuned-ade-512-512",
        label2id: Optional[Dict[str, int]] = None,
        num_labels: int = 2,
        freeze_encoder=True,
        **kwargs,
    ):
        super().__init__()

        # Load configuration with label2id if provided, ensuring correct label alignment.
        self.config = AutoConfig.from_pretrained(
            model_name,
            trust_remote_code=True,
            num_labels=num_labels,
            label2id=label2id or {},
        )

        # Load the semantic segmentation model
        self.model = AutoModelForSemanticSegmentation.from_pretrained(
            model_name,
            config=self.config,
            trust_remote_code=True,
            ignore_mismatched_sizes=True,
        )

        # Load image processor with appropriate options for rescaling and label reduction
        self.image_processor = AutoImageProcessor.from_pretrained(
            model_name, do_reduce_labels=False, do_rescale=False, trust_remote_code=True
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

        inputs = self.image_processor(images=image, return_tensors="pt").to(image.device)

        # Forward pass through the model
        outputs = self.model(**inputs)

        logits = F.interpolate(
            outputs.logits, size=original_size, mode="bilinear", align_corners=False
        )

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

        inputs = self.image_processor(images=image, return_tensors="pt").to(image.device)

        # Get model output
        outputs = self.model(**inputs)

        # Scale logits to the size of the labels
        predicted_masks = nn.functional.interpolate(
            outputs.logits,
            size=original_size,  # Match label dimensions
            mode="bilinear",
            align_corners=False,
        )

        return torch.argmax(predicted_masks, dim=1).cpu().numpy()

    def save(self, path: str):
        """
        Save the model and configuration to a specified directory.

        Args:
            path: Path to the directory where the model will be saved.

        Example : Segformer.load(path="../models/Segformer_cloud_seg")
        """
        print(f"Saving model to {path}...")
        self.model.save_pretrained(path)
        self.image_processor.save_pretrained(path)
        print("Model and image processor saved successfully.")

    @classmethod
    def load(cls, path: str, freeze_encoder=True):
        """
        Load a saved model and configuration from a specified directory.

        Args:
            path: Path to the directory from which the model will be loaded.
            freeze_encoder: Whether to freeze the encoder weights of the model.

        Returns:
            An instance of the `Segformer` class.
        """
        print(f"Loading model from {path}...")
        config = AutoConfig.from_pretrained(path, trust_remote_code=True, local_files_only=True)
        model = AutoModelForSemanticSegmentation.from_pretrained(
            path, config=config, trust_remote_code=True, local_files_only=True
        )
        image_processor = AutoImageProcessor.from_pretrained(
            path, trust_remote_code=True, local_files_only=True
        )

        # Create an instance of the Segformer class
        segformer_instance = cls(
            model_name=path,
            label2id=config.label2id,
            num_labels=config.num_labels,
            freeze_encoder=freeze_encoder,
        )

        # Replace loaded components into the instance
        segformer_instance.model = model
        segformer_instance.image_processor = image_processor
        if freeze_encoder:
            segformer_instance._freeze_encoder()

        print("Model loaded successfully.")
        return segformer_instance
