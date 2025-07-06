from __future__ import annotations

"""A Prithvi Segmentation model inspired by Instageo package."""
import json
import logging
from pathlib import Path
from typing import Literal

import torch
import yaml
from torch import nn

from src.models.foundational.backbones import prithvi_1, prithvi_2
from src.models.foundational.utils.download import download_file
from src.models.foundational.utils.seg_heads import (
    ResidualUpsamplingBlock,
    Seg_Conv_Decoder,
    UpscalingBlock,
)


class PrithviSeg(nn.Module):
    """Prithvi for Segmentation task."""

    def __init__(
        self,
        temporal_step: int = 1,
        image_size: int = 224,
        num_classes: int = 2,
        *,
        freeze_backbone: bool = True,
        head_type: Literal["res_block", None] = None,
        prithvi_version: Literal[1, 2] = 1,
        checkpoint_folder: str | None = None,
    ) -> None:
        """Initialize the PrithviSeg model.

        This model is designed for image segmentation tasks on remote sensing data.
        It loads Prithvi configuration and weights and sets up a ViTEncoder backbone
        along with a segmentation head.

        Args:
            temporal_step (int): Size of temporal dimension.
            image_size (int): Size of input image.
            num_classes (int): Number of target classes.
            freeze_backbone (bool): Flag to freeze ViT transformer backbone weights.
            checkpoint_folder (str): Folder storing model weights.
            prithvi_version (int): Prithvi backbone version, choose between 1 or 2 (add Temporal and Location encoding).
            head_type (str): Decoder head type. Only None and res_block available.
        """
        super().__init__()
        if checkpoint_folder:
            weights_dir = Path(checkpoint_folder) / "models" / "prithvi"
            weights_dir.mkdir(parents=True, exist_ok=True)

        #### Download Prithvi model
        self.prithvi_version = prithvi_version
        if self.prithvi_version == 1 and checkpoint_folder:
            weights_path = weights_dir / "Prithvi_EO_V1_100M.pt"
            cfg_path = weights_dir / "config.yaml"
            download_file(
                "https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-1.0-100M/resolve/main/Prithvi_EO_V1_100M.pt?download=true",  # noqa
                weights_path,
                always_download=False,
            )
            download_file(
                "https://huggingface.co/ibm-nasa-geospatial/Prithvi-100M/raw/main/config.yaml",  # noqa
                cfg_path,
                always_download=False,
            )

            checkpoint = torch.load(weights_path, map_location="cpu")
            with Path.open(cfg_path) as f:
                model_config = yaml.safe_load(f)

            model_args = model_config["model_args"]

        elif self.prithvi_version == 2 and checkpoint_folder:
            weights_path = weights_dir / "Prithvi_EO_V2_300M.pt"
            cfg_path = weights_dir / "config.json"
            download_file(
                "https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M/resolve/main/Prithvi_EO_V2_300M.pt?download=true",  # noqa
                weights_path,
                always_download=False,
            )
            download_file(
                "https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M/raw/main/config.json",  # noqa
                cfg_path,
                always_download=True,
            )

            checkpoint = torch.load(weights_path, map_location="cpu")
            with Path.open(cfg_path) as f:
                model_config = json.load(f)

            model_args = model_config["pretrained_cfg"]
        elif self.prithvi_version == 1 and not checkpoint_folder:
            # default arguments
            model_args = {
                "decoder_depth": 8,
                "decoder_embed_dim": 512,
                "decoder_num_heads": 16,
                "depth": 12,
                "embed_dim": 768,
                "img_size": 224,
                "in_chans": 6,
                "num_frames": 3,
                "num_heads": 12,
                "patch_size": 16,
                "tubelet_size": 1,
            }
        elif self.prithvi_version == 2 and not checkpoint_folder:
            model_args = {
                "img_size": 224,
                "num_frames": 4,
                "patch_size": [1, 16, 16],
                "in_chans": 6,
                "embed_dim": 1024,
                "depth": 24,
                "num_heads": 16,
                "decoder_embed_dim": 512,
                "decoder_depth": 8,
                "decoder_num_heads": 16,
                "mlp_ratio": 4,
                "coords_encoding": [],
                "coords_scale_learn": False,
                "mask_ratio": 0.75,
                "norm_pix_loss": False,
                "bands": ["B02", "B03", "B04", "B05", "B06", "B07"],
                "mean": [1087.0, 1342.0, 1433.0, 2734.0, 1958.0, 1363.0],
                "std": [2248.0, 2179.0, 2178.0, 1850.0, 1242.0, 1049.0],
                "origin_url": "https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M",
                "paper_ids": "arXiv:X.X",
            }

        model_args.update(
            num_frames=temporal_step,
            img_size=image_size,
        )

        self.model_args = model_args
        self.temporal_step = temporal_step

        # instantiate model
        if self.prithvi_version == 1:
            model = prithvi_1.PrithviViTEncoder(**model_args)
        elif self.prithvi_version == 2:
            model = prithvi_2.PrithviViTEncoder(**model_args)

        # Freeze Backbone
        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
            logging.info("Backbone has been frozen.")
        else:
            for param in model.parameters():
                param.requires_grad = True
            logging.info("Backbone weights are trainable.")

        embed_dims = model_args["embed_dim"]  # 768 (1.0) , 1024 (2.0)

        if checkpoint:
            # ensure encoder weight extraction
            filtered_checkpoint_state_dict = {
                key[len("encoder.") :]: value
                for key, value in checkpoint.items()
                if key.startswith("encoder.")
            }

            # The positional embeddings are reset (set to zeros)
            # to allow for fine-tuning on a different dataset or input resolution.
            filtered_checkpoint_state_dict["pos_embed"] = torch.zeros(
                1, (temporal_step * (image_size // 16) ** 2 + 1), embed_dims
            )

            _ = model.load_state_dict(filtered_checkpoint_state_dict, strict=False)

        # Expect input shape (batch_size, Channels, Timestamps, Height, Width)
        self.prithvi_backbone = model

        embed_dims = [
            (model_args["embed_dim"] * model_args["num_frames"]) // (2**i) for i in range(5)
        ]
        # 1.0 => [2304, 1152, 576, 288, 144]
        # 2.0 => [3072, 1536, 768, 384, 192]

        if head_type is None:
            self.segmentation_head = nn.Sequential(
                *[UpscalingBlock(embed_dims[i], embed_dims[i + 1]) for i in range(4)],
                nn.Conv2d(kernel_size=1, in_channels=embed_dims[-1], out_channels=num_classes),
            )
        elif head_type == "res_block":
            self.segmentation_head = nn.Sequential(
                *[ResidualUpsamplingBlock(embed_dims[i], embed_dims[i + 1]) for i in range(4)],
                nn.Conv2d(kernel_size=1, in_channels=embed_dims[-1], out_channels=num_classes),
            )
        else:
            raise ValueError("Head Type for Decoder takes only None, res_block")

    def forward(
        self,
        x: torch.Tensor,
        temporal_coords: None | torch.Tensor = None,
        location_coords: None | torch.Tensor = None,
    ) -> torch.Tensor:
        """Define the forward pass of the model.

        Args:
            x (torch.Tensor): The input tensor representing the image. (both version)
            temporal_coords : None | torch.Tensor (only Prithvi 2.0)
            location_coords: None | torch.Tensor (only Prithvi 2.0)

        Returns:
            torch.Tensor: Output tensor after image segmentation.
        """
        prithvi_version = self.prithvi_version

        if prithvi_version == 1:
            features = self.prithvi_backbone(x)
        elif prithvi_version == 2:  # noqa: PLR2004
            features = self.prithvi_backbone.forward_features(x, temporal_coords, location_coords)

        reshaped_features = self.prithvi_backbone.prepare_features_for_image_model(features)

        return self.segmentation_head(reshaped_features)
