import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from src.models.foundational.utils.pos_embed import posemb_sincos_1d


class FeedForward(nn.Module):
    """Feed Forward Network."""

    def __init__(self, dim: int, hidden_dim: int) -> None:
        """Define a Feed Forward Network."""
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.net(x)


class Attention(nn.Module):
    """Attention Layer."""

    def __init__(
        self, dim: int, heads: int = 8, dim_head: int = 64, *, fused_attn: bool = True
    ) -> None:
        """Initialize Attention Layer."""
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head**-0.5
        self.norm = nn.LayerNorm(dim)
        self.fused_attn = fused_attn

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = (rearrange(t, "b n (h d) -> b h n d", h=self.heads) for t in qkv)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        else:
            attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale
            attn = attn.softmax(dim=-1)
            x = torch.matmul(attn, v)

        x = rearrange(x, "b h n d -> b n (h d)")
        return self.to_out(x)


class Transformer(nn.Module):
    """Implement a Transformer Layer."""

    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        *,
        fused_attn: bool,
    ) -> None:
        """Initialize a Transformer Layer with the specified configuration.

        Args:
            dim (int): The dimensionality of the input and output features.
            depth (int): The number of transformer sub-layers to stack.
            heads (int): The number of attention heads in the multi-head attention mechanism.
            dim_head (int): The dimensionality of each attention head.
            mlp_dim (int): The dimensionality of the feedforward network within each layer.
            fused_attn (bool): Whether to use fused attention operations for efficiency.

        """
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(dim, heads=heads, dim_head=dim_head, fused_attn=fused_attn),
                        FeedForward(dim, mlp_dim),
                    ]
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward Pass."""
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


class FCBlock(nn.Module):
    """FCBlock."""

    def __init__(self, size: int) -> None:
        """Initialize the FCBlock (Residual Linear Layer).

        Args:
            size (int): input/output size of Linear Layer.
        """
        self.l1 = nn.Linear(size, size)
        self.l2 = nn.Linear(size, size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward passes."""
        y = F.gelu(self.l1(x))
        y = F.gelu(self.l2(y))
        return x + y


class WavesTransformer(nn.Module):
    """A transformer-based module for generating dynamic weights and biases from input wave embeddings."""

    def __init__(
        self,
        wave_dim: int,
        output_dim: int,
        num_latent_tokens: int,
        embed_dim: int,
        *,
        is_decoder: bool,
        num_heads: int = 4,
        num_layers: int = 1,
    ) -> None:
        """Initialize the WavesTransformer.

        Args:
            wave_dim (int): Dimension of the wave embedding.
            output_dim (int): Output dimension for the generated weights.
            num_latent_tokens (int): Number of latent tokens.
            embed_dim (int): Embedding dimension.
            is_decoder (bool): Whether the module is used as a decoder.
            num_heads (int, optional): Number of attention heads. Defaults to 4.
            num_layers (int, optional): Number of transformer layers. Defaults to 1.
        """
        self.num_latent_tokens = num_latent_tokens
        self.is_decoder = is_decoder
        layer = nn.TransformerEncoderLayer(
            d_model=wave_dim,
            nhead=num_heads,
            activation="gelu",
            dropout=0,
            norm_first=False,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers)

        self.fc_weight = nn.Linear(wave_dim, output_dim)
        self.fc_bias = None if self.is_decoder else nn.Linear(wave_dim, embed_dim)

        self.weight_tokens = nn.Parameter(torch.randn(self.num_latent_tokens, wave_dim) * 0.02)
        self.bias_token = nn.Parameter(torch.randn(1, wave_dim) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = torch.cat([self.weight_tokens, x, self.bias_token], dim=0)
        out = self.encoder(x)
        weights = self.fc_weight(out[self.num_latent_tokens : -1] + x[self.num_latent_tokens : -1])
        bias = None if self.is_decoder else self.fc_bias(out[-1])
        return weights, bias


class DynamicEmbedding(nn.Module):
    """Dynamic Embedding."""

    def __init__(
        self,
        wave_dim: int,
        num_latent_tokens: int,
        patch_size: int,
        embed_dim: int,
        *,
        is_decoder: int = False,
    ) -> None:
        """Initialize the DynamicEmbedding module.

        Args:
            wave_dim (int): Dimension of the wave embedding.
            num_latent_tokens (int): Number of latent tokens.
            patch_size (int): Size of the patch.
            embed_dim (int): Embedding dimension.
            is_decoder (bool, optional): Whether the module is used as a decoder. Defaults to False.
        """
        super().__init__()
        self.wave_dim = wave_dim
        self.num_latent_tokens = num_latent_tokens
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.is_decoder = is_decoder
        self.output_dim = (patch_size**2) * embed_dim

        self.weight_generator = WavesTransformer(
            wave_dim,
            self.output_dim,
            self.num_latent_tokens,
            self.embed_dim,
            is_decoder,
        )
        self.fclayer = FCBlock(self.wave_dim)

        self.initialize_weights()

    def forward(
        self, batch: torch.Tensor, waves: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass."""
        waves = posemb_sincos_1d(waves, self.wave_dim)
        waves = waves.to(batch.device)
        waves = self.fclayer(waves)
        weight, bias = self.weight_generator(waves)

        if self.is_decoder:
            dynamic_weight = rearrange(
                weight,
                "cin (k1 k2 cout) -> (cin k1 k2) cout",
                k1=self.patch_size,
                k2=self.patch_size,
                cout=self.embed_dim,
            )
            if bias is not None:
                bias = rearrange(bias, "b -> (b)")
            dynamic_out = F.linear(batch, dynamic_weight * 0.02, bias=bias)
            x = dynamic_out
        else:
            dynamic_weight = rearrange(
                weight,
                "cin (cout k1 k2) -> cout cin k1 k2",
                k1=self.patch_size,
                k2=self.patch_size,
            )
            if bias is not None:
                bias = rearrange(bias, "b -> (b)")
            dynamic_out = F.conv2d(batch, dynamic_weight * 0.02, bias=bias, stride=self.patch_size)
            x = rearrange(dynamic_out, "b c h w -> b (h w) c")

        return x, waves

    def initialize_weights(self) -> None:
        """Initialize weights using Xavier initialization."""
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
