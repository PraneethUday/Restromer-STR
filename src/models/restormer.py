"""Restormer: Efficient Transformer for High-Resolution Image Restoration.

Faithful reimplementation of the architecture from Zamir et al., CVPR 2022
(arXiv:2111.09881), laid out so the official `single_image_defocus_deblurring.pth`
checkpoint loads without key remapping.

Two ideas carry the architecture:

* **MDTA** (Multi-Dconv Head Transposed Attention) applies self-attention across
  the *channel* dimension rather than the spatial one, so cost is linear in
  pixel count instead of quadratic. That is what makes full-resolution document
  pages tractable.
* **GDFN** (Gated-Dconv Feed-Forward Network) gates one depth-wise branch by
  another, letting the block suppress the less informative features before they
  propagate.

The encoder-decoder is a 4-level U-Net with pixel-unshuffle downsampling, so no
information is discarded on the way down.
"""

from __future__ import annotations

import numbers

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# --------------------------------------------------------------------------- #
# Normalisation
# --------------------------------------------------------------------------- #
def to_3d(x: torch.Tensor) -> torch.Tensor:
    return rearrange(x, "b c h w -> b (h w) c")


def to_4d(x: torch.Tensor, h: int, w: int) -> torch.Tensor:
    return rearrange(x, "b (h w) c -> b c h w", h=h, w=w)


class BiasFreeLayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBiasLayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    """LayerNorm over the channel axis of an NCHW tensor."""

    def __init__(self, dim: int, layer_norm_type: str = "WithBias"):
        super().__init__()
        if layer_norm_type == "BiasFree":
            self.body = BiasFreeLayerNorm(dim)
        else:
            self.body = WithBiasLayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


# --------------------------------------------------------------------------- #
# Core blocks
# --------------------------------------------------------------------------- #
class FeedForward(nn.Module):
    """Gated-Dconv Feed-Forward Network (GDFN)."""

    def __init__(self, dim: int, ffn_expansion_factor: float, bias: bool):
        super().__init__()
        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, 1, bias=bias)
        self.dwconv = nn.Conv2d(
            hidden_features * 2,
            hidden_features * 2,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=hidden_features * 2,
            bias=bias,
        )
        self.project_out = nn.Conv2d(hidden_features, dim, 1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        return self.project_out(F.gelu(x1) * x2)


class Attention(nn.Module):
    """Multi-Dconv Head Transposed Attention (MDTA).

    Attention is computed over channels, so complexity is O(HW) rather than
    O((HW)^2) - the property that lets this run on full pages.
    """

    def __init__(self, dim: int, num_heads: int, bias: bool):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias
        )
        self.project_out = nn.Conv2d(dim, dim, 1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        k = rearrange(k, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        v = rearrange(v, "b (head c) h w -> b head c (h w)", head=self.num_heads)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = attn @ v
        out = rearrange(
            out, "b head c (h w) -> b (head c) h w", head=self.num_heads, h=h, w=w
        )
        return self.project_out(out)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        ffn_expansion_factor: float,
        bias: bool,
        layer_norm_type: str,
    ):
        super().__init__()
        self.norm1 = LayerNorm(dim, layer_norm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, layer_norm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


# --------------------------------------------------------------------------- #
# Resampling
# --------------------------------------------------------------------------- #
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c: int = 3, embed_dim: int = 48, bias: bool = False):
        super().__init__()
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class Downsample(nn.Module):
    """Halve resolution, double channels - lossless via pixel-unshuffle."""

    def __init__(self, n_feat: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelUnshuffle(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


# --------------------------------------------------------------------------- #
# Restormer
# --------------------------------------------------------------------------- #
class Restormer(nn.Module):
    """Defaults match the official single-image defocus deblurring config."""

    def __init__(
        self,
        inp_channels: int = 3,
        out_channels: int = 3,
        dim: int = 48,
        num_blocks: tuple[int, ...] = (4, 6, 6, 8),
        num_refinement_blocks: int = 4,
        heads: tuple[int, ...] = (1, 2, 4, 8),
        ffn_expansion_factor: float = 2.66,
        bias: bool = False,
        LayerNorm_type: str = "WithBias",  # noqa: N803 - matches checkpoint config
        dual_pixel_task: bool = False,
    ):
        super().__init__()

        def block(d: int, h: int) -> TransformerBlock:
            return TransformerBlock(d, h, ffn_expansion_factor, bias, LayerNorm_type)

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim, bias)

        self.encoder_level1 = nn.Sequential(*[block(dim, heads[0]) for _ in range(num_blocks[0])])
        self.down1_2 = Downsample(dim)

        self.encoder_level2 = nn.Sequential(
            *[block(dim * 2, heads[1]) for _ in range(num_blocks[1])]
        )
        self.down2_3 = Downsample(dim * 2)

        self.encoder_level3 = nn.Sequential(
            *[block(dim * 4, heads[2]) for _ in range(num_blocks[2])]
        )
        self.down3_4 = Downsample(dim * 4)

        self.latent = nn.Sequential(*[block(dim * 8, heads[3]) for _ in range(num_blocks[3])])

        self.up4_3 = Upsample(dim * 8)
        self.reduce_chan_level3 = nn.Conv2d(dim * 8, dim * 4, 1, bias=bias)
        self.decoder_level3 = nn.Sequential(
            *[block(dim * 4, heads[2]) for _ in range(num_blocks[2])]
        )

        self.up3_2 = Upsample(dim * 4)
        self.reduce_chan_level2 = nn.Conv2d(dim * 4, dim * 2, 1, bias=bias)
        self.decoder_level2 = nn.Sequential(
            *[block(dim * 2, heads[1]) for _ in range(num_blocks[1])]
        )

        self.up2_1 = Upsample(dim * 2)
        # Level 1 keeps 2*dim channels - no reduction, per the paper.
        self.decoder_level1 = nn.Sequential(
            *[block(dim * 2, heads[0]) for _ in range(num_blocks[0])]
        )

        self.refinement = nn.Sequential(
            *[block(dim * 2, heads[0]) for _ in range(num_refinement_blocks)]
        )

        self.dual_pixel_task = dual_pixel_task
        if dual_pixel_task:
            self.skip_conv = nn.Conv2d(dim, dim * 2, 1, bias=bias)

        self.output = nn.Conv2d(dim * 2, out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img: torch.Tensor) -> torch.Tensor:
        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)

        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)

        inp_enc_level4 = self.down3_4(out_enc_level3)
        latent = self.latent(inp_enc_level4)

        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)
        out_dec_level1 = self.refinement(out_dec_level1)

        if self.dual_pixel_task:
            out_dec_level1 = out_dec_level1 + self.skip_conv(inp_enc_level1)
            return self.output(out_dec_level1)

        # Global residual: the network predicts the correction, not the image.
        return self.output(out_dec_level1) + inp_img


def load_pretrained(
    weights_path: str, device: str = "cpu", **kwargs
) -> Restormer:
    """Instantiate Restormer and load an official checkpoint.

    Handles both the raw `state_dict` layout and the `{'params': ...}` wrapper
    used by the BasicSR-style release archives.
    """
    model = Restormer(**kwargs)
    ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
    state = ckpt.get("params", ckpt.get("state_dict", ckpt)) if isinstance(ckpt, dict) else ckpt
    state = {k.replace("module.", "", 1): v for k, v in state.items()}

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[restormer] missing keys: {len(missing)} (first: {missing[:3]})")
    if unexpected:
        print(f"[restormer] unexpected keys: {len(unexpected)} (first: {unexpected[:3]})")

    return model.to(device).eval()
