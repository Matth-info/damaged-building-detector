from __future__ import annotations

import logging

import torch
from torch import nn


def posemb_sincos_2d(
    h: int, w: int, dim: int, temperature: int = 10000, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Generate 2D sine-cosine positional embeddings.

    Args:
        h (int): Height of the grid.
        w (int): Width of the grid.
        dim (int): Embedding dimension (must be a multiple of 4).
        temperature (int, optional): Temperature parameter for scaling. Defaults to 10000.
        dtype (torch.dtype, optional): Data type of the output tensor. Defaults to torch.float32.

    Returns:
        torch.Tensor: The positional embedding tensor of shape (h * w, dim).
    """
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    if (dim % 4) != 0:
        raise ValueError("feature dimension must be multiple of 4 for sincos emb")
    omega = torch.arange(dim // 4) / (dim // 4 - 1)
    omega = 1.0 / (temperature**omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)


def posemb_sincos_2d_with_gsd(
    h: int,
    w: int,
    dim: int,
    gsd: float = 1.0,
    temperature: int = 10000,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Generate 2D sine-cosine positional embeddings with ground sample distance (GSD) scaling.

    Args:
        h (int): Height of the grid (number of rows).
        w (int): Width of the grid (number of columns).
        dim (int): Embedding dimension. Must be a multiple of 4.
        gsd (float or torch.Tensor, optional): Ground sample distance scaling factor. Default is 1.0.
        temperature (int, optional): Temperature parameter for frequency scaling. Default is 10000.
        dtype (torch.dtype, optional): Data type of the output tensor. Default is torch.float32.

    Returns:
        torch.Tensor: A tensor of shape (h * w, dim) containing the positional embeddings.
    """
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    if (dim % 4) != 0:
        raise ValueError("feature dimension must be multiple of 4 for sincos emb")

    gsd = gsd.to(x.device)
    omega = torch.arange(dim // 4) / (dim // 4 - 1)
    omega = 1.0 / (temperature ** (2 * omega / dim)) * (gsd / 1.0)  # Adjusted for g

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)


def posemb_sincos_1d(
    waves: int | torch.Tensor,
    dim: int,
    temperature: int = 10000,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Generate 1D sinusoidal positional embeddings.

    Args:
        waves (int or torch.Tensor): Number of positions (if int) or a tensor of positions.
        dim (int): The embedding dimension. Must be an even number.
        temperature (int, optional): Temperature parameter controlling the frequency scaling. Default is 10000.
        dtype (torch.dtype, optional): Data type of the output tensor. Default is torch.float32.

    Returns:
        torch.Tensor: A tensor of shape (waves, dim) containing the positional embeddings.

    Raises:
        AssertionError: If `dim` is not a multiple of 2.

    Notes:
        The function creates sinusoidal embeddings as described in "Attention is All You Need" (Vaswani et al., 2017).
    """
    if dim % 2 != 0:
        raise ValueError("Feature dimension must be a multiple of 2 for sincos embedding")

    waves = torch.arange(waves) if isinstance(waves, int) else waves

    omega = torch.arange(dim // 2, device=waves.device) / (dim // 2 - 1)
    omega = 1.0 / (temperature**omega)

    scaled_waves = waves[:, None] * omega[None, :]
    pe = torch.cat((scaled_waves.sin(), scaled_waves.cos()), dim=1)

    return pe.type(dtype)


# --------------------------------------------------------
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# Position embedding utils
# --------------------------------------------------------

import numpy as np
import torch


# --------------------------------------------------------
# 2D sine-cosine position embedding
# References:
# Transformer: https://github.com/tensorflow/models/blob/master/official/nlp/transformer/model_utils.py
# MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------
def get_2d_sincos_pos_embed(
    embed_dim: int, grid_size: int, *, cls_token: bool = False
) -> np.ndarray:
    """Get 2D sincos positional embedding.

    Args:
        embed_dim (int): Dimension of the embedding.
        grid_size (int): Size of the grid height and width.
        cls_token (bool, optional): Whether to include a class token at the beginning. Default is False.

    Returns:
        pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_with_resolution(
    embed_dim: int,
    grid_size: int,
    res: np.ndarray,
    *,
    cls_token: bool = False,
    device: str = "cpu",
) -> torch.Tensor:
    """Get 2D sincos positional embedding with resolution information.

    Args:
        embed_dim (int): Dimension of the embedding.
        grid_size (int): Size of the grid height and width.
        res (np.ndarray): Array of size n, representing the resolution of a pixel (e.g., in meters).
        cls_token (bool, optional): Whether to include a class token at the beginning. Default is False.
        device (str, optional): Device on which to place the tensors. Default is "cpu".

    Returns:
        torch.Tensor: pos_embed of shape [n, grid_size*grid_size, embed_dim] or [n, 1+grid_size*grid_size, embed_dim] (with or without cls_token).
    """
    res = res.to(device)
    grid_h = torch.arange(grid_size, dtype=torch.float32, device=device)
    grid_w = torch.arange(grid_size, dtype=torch.float32, device=device)
    grid = torch.meshgrid(
        grid_w, grid_h, indexing="xy"
    )  # here h goes first,direction reversed for numpy
    grid = torch.stack(grid, dim=0)  # 2 x h x w

    grid = torch.einsum("chw,n->cnhw", grid, res)  # 2 x n x h x w
    _, n, h, w = grid.shape
    pos_embed = get_2d_sincos_pos_embed_from_grid_torch(embed_dim, grid)  #  # (nxH*W, D/2)
    pos_embed = pos_embed.reshape(n, h * w, embed_dim)
    if cls_token:
        pos_embed = torch.cat(
            [
                torch.zeros([n, 1, embed_dim], dtype=torch.float32, device=pos_embed.device),
                pos_embed,
            ],
            dim=1,
        )
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim: int, grid: list) -> np.ndarray:
    """Get 2d sincos positional embedding.

    Args:
        embed_dim: output dimension for each position
        grid: a list of positions to be encoded: size (M,)

    Returns:
        out: (M, D)
    """
    if embed_dim % 2 != 0:
        raise ValueError("Embedding dimension should be even.")

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    return np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)


def get_2d_sincos_pos_embed_from_grid_torch(embed_dim: int, grid: list) -> torch.Tensor:
    """Get 2d sincos positional embedding from grid torch.

    Args:
        embed_dim: output dimension for each position
        grid: a list or tensor containing the grid positions to be encoded, typically of shape (2, H, W) or similar.

    Returns:
        out: (M, D)
    """
    if embed_dim % 2 != 0:
        raise ValueError("Embedding dimension should be even.")

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid_torch(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid_torch(embed_dim // 2, grid[1])  # (H*W, D/2)

    return torch.cat([emb_h, emb_w], dim=1)  # (H*W, D)


def get_1d_sincos_pos_embed_from_grid_torch(embed_dim: int, pos: list) -> torch.Tensor:
    """Get 1d sincos positional embedding from grid torch.

    Args:
        embed_dim: output dimension for each position
        pos: a list of positions to be encoded: size (M,)

    Returns:
        out: (M, D)
    """
    if embed_dim % 2 != 0:
        raise ValueError("Embedding dimension should be even.")
    old_shape = pos
    omega = torch.arange(embed_dim // 2, dtype=torch.float32, device=pos.device)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = torch.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = torch.sin(out)  # (M, D/2)
    emb_cos = torch.cos(out)  # (M, D/2)

    return torch.cat([emb_sin, emb_cos], dim=1)  # (M, D)


def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: list) -> np.ndarray:
    """Get 1d sincos positional embedding from grid.

    Args:
        embed_dim: output dimension for each position
        pos: a list of positions to be encoded: size (M,)

    Returns:
        out: (M, D)
    """
    if embed_dim % 2 != 0:
        raise ValueError("Embedding dimension should be even.")
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    return np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)


# --------------------------------------------------------
# Interpolate position embeddings for high-resolution
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------


def interpolate_pos_embed(new_model: nn.Module, checkpoint_model: nn.Module) -> None:
    """Interpolate positional embedding for adapting pretrained positional embedding layer to high resolution images.

    Args:
        new_model (nn.Module): torch Model module.
        checkpoint_model (dict): weight dictionary.
    """
    if "pos_embed" in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model[
            "pos_embed"
        ]  # extract positional embedding layer weights.
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = new_model.patch_embed.num_patches
        num_extra_tokens = new_model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches**0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            logging.info(
                "Position interpolate from %dx%d to %dx%d",
                orig_size,
                orig_size,
                new_size,
                new_size,
            )
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(
                0, 3, 1, 2
            )
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens,
                size=(new_size, new_size),
                mode="bicubic",
                align_corners=False,
            )
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model["pos_embed"] = new_pos_embed
