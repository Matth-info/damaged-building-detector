import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn
from typing import Optional, Callable


class TwoLayerConv2d(nn.Sequential):
    """Two Layer 2D convolution.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int, optional): Size of the convolving kernel. Default is 3.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3) -> None:  # noqa: D107
        super().__init__(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                stride=1,
                bias=False,
            ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                stride=1,
            ),
        )


class Residual(nn.Module):
    """Residual Layer.

    Args:
        fn (Callable): Function or module to apply in the residual block.
    """

    def __init__(self, fn: Callable) -> None:  # noqa: D107
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor.
            **kwargs: Additional arguments for fn.

        Returns:
            torch.Tensor: Output tensor after applying residual connection.
        """
        return self.fn(x, **kwargs) + x


class Residual2(nn.Module):
    """Siamese Residual Layer.

    Args:
        fn (Callable): Function or module to apply in the residual block.
    """

    def __init__(self, fn: Callable) -> None:  # noqa: D107
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor, x2: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): First input tensor.
            x2 (torch.Tensor): Second input tensor.
            **kwargs: Additional arguments for fn.

        Returns:
            torch.Tensor: Output tensor after applying residual connection.
        """
        return self.fn(x, x2, **kwargs) + x


class PreNorm(nn.Module):
    """Pre-normalization layer (apply normalization before function calling).

    Args:
        dim (int): Input feature dimension.
        fn (Callable): Function or module to apply after normalization.
    """

    def __init__(self, dim: int, fn: Callable) -> None:  # noqa: D107
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor.
            **kwargs: Additional arguments for fn.

        Returns:
            torch.Tensor: Output tensor after normalization and function application.
        """
        return self.fn(self.norm(x), **kwargs)


class PreNorm2(nn.Module):
    """Siamese Pre-normalization layer (apply normalization before function calling).

    Args:
        dim (int): Input feature dimension.
        fn (Callable): Function or module to apply after normalization.
    """

    def __init__(self, dim: int, fn: Callable) -> None:  # noqa: D107
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x: torch.Tensor, x2: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): First input tensor.
            x2 (torch.Tensor): Second input tensor.
            **kwargs: Additional arguments for fn.

        Returns:
            torch.Tensor: Output tensor after normalization and function application.
        """
        return self.fn(self.norm(x), self.norm(x2), **kwargs)


class FeedForward(nn.Module):
    """Feed Forward Network (FFN).

    Args:
        dim (int): Input and output feature dimension.
        hidden_dim (int): Hidden layer dimension.
        dropout (float, optional): Dropout probability. Default is 0.0.
    """

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        """Initialize a FFN.

        Args:
            dim (int): Input and output feature dimension.
            hidden_dim (int): Hidden layer dimension.
            dropout (float, optional): Dropout probability. Default is 0.0.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after feedforward network.
        """
        return self.net(x)


class Cross_Attention(nn.Module):
    """Cross Attention layer for neural networks.

    This module implements a cross-attention mechanism, where queries are projected from input `x` and keys/values are projected from memory input `m`. The attention scores are computed between `x` and `m`, optionally applying a mask, and the attended output is returned.

    Args:
        dim (int): Input and output feature dimension.
        heads (int, optional): Number of attention heads. Default is 8.
        dim_head (int, optional): Dimension of each attention head. Default is 64.
        dropout (float, optional): Dropout probability applied to the output. Default is 0.0.
        softmax (bool, optional): Whether to apply softmax to attention scores. Default is True.

    Inputs:
        x (torch.Tensor): Query tensor of shape (batch_size, seq_len, dim).
        m (torch.Tensor): Memory tensor (for keys and values) of shape (batch_size, mem_len, dim).
        mask (Optional[torch.Tensor]): Optional boolean mask tensor of shape (batch_size, mem_len), where True indicates valid positions.

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, seq_len, dim) after applying cross-attention.
    """

    def __init__(  # noqa: D107
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        softmax: bool = True,
    ) -> None:
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim**-0.5

        self.softmax = softmax
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(
        self,
        x: torch.Tensor,
        m: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): Query tensor of shape (batch_size, seq_len, dim).
            m (torch.Tensor): Memory tensor (for keys and values) of shape (batch_size, mem_len, dim).
            mask (Optional[torch.Tensor]): Optional boolean mask tensor of shape (batch_size, mem_len).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, dim) after applying cross-attention.
        """
        b, n, _, h = *x.shape, self.heads
        q = self.to_q(x)
        k = self.to_k(m)
        v = self.to_v(m)

        q, k, v = (rearrange(t, "b n (h d) -> b h n d", h=h) for t in [q, k, v])

        dots = torch.einsum("bhid,bhjd->bhij", q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], "mask has incorrect dimensions"
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask

        attn = dots.softmax(dim=-1) if self.softmax else dots

        out = torch.einsum("bhij,bhjd->bhid", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.to_out(out)

        return out


class Attention(nn.Module):
    """Attention layer.

    Args:
        dim (int): Input and output feature dimension.
        heads (int, optional): Number of attention heads. Default is 8.
        dim_head (int, optional): Dimension of each attention head. Default is 64.
        dropout (float, optional): Dropout probability applied to the output. Default is 0.0.

    Inputs:
        x (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim).
        mask (Optional[torch.Tensor]): Optional boolean mask tensor of shape (batch_size, seq_len).

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, seq_len, dim) after applying attention.
    """

    def __init__(  # noqa: D107
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim**-0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim).
            mask (Optional[torch.Tensor]): Optional boolean mask tensor of shape (batch_size, seq_len).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, dim) after applying attention.
        """
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = (rearrange(t, "b n (h d) -> b h n d", h=h) for t in qkv)

        dots = torch.einsum("bhid,bhjd->bhij", q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], "mask has incorrect dimensions"
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask

        attn = dots.softmax(dim=-1)

        out = torch.einsum("bhij,bhjd->bhid", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.to_out(out)
        return out


class Transformer(nn.Module):
    """Transformer Layer."""

    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        dropout: float,
    ) -> None:
        """Initialize a transformer layer.

        Args:
            dim (int): Input and output feature dimension.
            depth (int): Number of transformer layers.
            heads (int): Number of attention heads.
            dim_head (int): Dimension of each attention head.
            mlp_dim (int): Hidden dimension of the feedforward network.
            dropout (float): Dropout probability.
        """
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Residual(
                            PreNorm(
                                dim,
                                Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                            ),
                        ),
                        Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))),
                    ],
                ),
            )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim).
            mask (Optional[torch.Tensor]): Optional boolean mask tensor.

        Returns:
            torch.Tensor: Output tensor after transformer layers.
        """
        for attn, ff in self.layers:
            x = attn(x, mask=mask)
            x = ff(x)
        return x


class TransformerDecoder(nn.Module):
    """Transformer Decoder Layer.

    Args:
        dim (int): Input and output feature dimension.
        depth (int): Number of transformer decoder layers.
        heads (int): Number of attention heads.
        dim_head (int): Dimension of each attention head.
        mlp_dim (int): Hidden dimension of the feedforward network.
        dropout (float): Dropout probability.
        softmax (bool, optional): Whether to apply softmax to attention scores. Default is True.
    """

    def __init__(  # noqa: D107
        self,
        dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        dropout: float,
        softmax: bool = True,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Residual2(
                            PreNorm2(
                                dim,
                                Cross_Attention(
                                    dim,
                                    heads=heads,
                                    dim_head=dim_head,
                                    dropout=dropout,
                                    softmax=softmax,
                                ),
                            ),
                        ),
                        Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))),
                    ],
                ),
            )

    def forward(
        self,
        x: torch.Tensor,
        m: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): Target (query) tensor of shape (batch_size, seq_len, dim).
            m (torch.Tensor): Memory tensor of shape (batch_size, mem_len, dim).
            mask (Optional[torch.Tensor]): Optional boolean mask tensor.

        Returns:
            torch.Tensor: Output tensor after transformer decoder layers.
        """
        for attn, ff in self.layers:
            x = attn(x, m, mask=mask)
            x = ff(x)
        return x
