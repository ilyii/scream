import math
from typing import List, Optional, Tuple

import torch
from torch import Tensor
from torch import nn as nn

PARAMS = {
        "extractor_mode": "group_norm",
        "extractor_conv_config": [
            (512, 10, 5),
            (512, 3, 2),
            (512, 3, 2),
            (512, 3, 2),
            (512, 3, 2),
            (512, 2, 2),
            (512, 2, 2),
        ],
        "extractor_conv_bias": False,
        "encoder_embed_dim": 768,
        "encoder_projection_dropout": 0.1,
        "encoder_pos_conv_kernel": 128,
        "encoder_pos_conv_groups": 16,
        "encoder_num_layers": 12,
        "encoder_num_heads": 12,
        "encoder_attention_dropout": 0.1,
        "encoder_ff_interm_features": 3072,
        "encoder_ff_interm_dropout": 0.0,
        "encoder_dropout": 0.1,
        "encoder_layer_norm_first": False,
        "encoder_layer_drop": 0.05,
        "aux_num_out": 29,
}

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, bias, layer_norm):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.layer_norm = layer_norm
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, bias=bias)

    def forward(self, x, length):
        x = self.conv(x)
        if self.layer_norm is not None:
            x = self.layer_norm(x)
        x = nn.functional.gelu(x)

        if length is not None:
            length = torch.div(length - self.kernel_size, self.stride, rounding_mode="floor") + 1
            length = torch.max(torch.zeros_like(length), length)
        return x, length


class FeatureProjection(nn.Module):
    def __init__(self, in_features, out_features, dropout):
        super().__init__()
        self.layer_norm = nn.LayerNorm(in_features)
        self.projection = nn.Linear(in_features, out_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.layer_norm(x)
        x = self.projection(x)
        x = self.dropout(x)
        return x


class ConvolutionalPositionalEmbedding(nn.Module):
    def __init__(self, embed_dim, kernel_size, groups):
        super().__init__()
        self.embed_dim = embed_dim
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(
            in_channels=embed_dim,
            out_channels=embed_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=groups,
        )

        self.conv = nn.utils.parametrizations.weight_norm(self.conv, name="weight", dim=2)
        self.num_remove = 1 if kernel_size % 2 == 0 else 0

    # def __prepare_scriptable__(self):
    #     if self.conv.__class__.__name__ == "ParametrizedConv1d":
    #         _LG.warning("Removing weight_norm from %s", self.__class__.__name__)
    #         torch.nn.utils.parametrize.remove_parametrizations(self.conv, "weight")
    #     return self

    def forward(self, x):
        x = x.transpose(-2, -1)
        x = self.conv(x)
        if self.num_remove > 0:
            x = x[..., : -self.num_remove]
        x = torch.nn.functional.gelu(x)
        x = x.transpose(-2, -1)
        return x


class SelfAttention(nn.Module):
    """Multihead Self Attention module

    Args:
        embed_dim (int): Total dimension of the model.
        num_heads (int): The number of heads.
        dropout (float, optional):
            Dropout probability on attn_output_weights. Default: ``0.0``
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        head_dim = embed_dim // num_heads
        if head_dim * num_heads != embed_dim:
            raise ValueError(f"`embed_dim ({embed_dim})` is not divisible by `num_heads ({num_heads})`")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = head_dim

        self.scaling = self.head_dim**-0.5

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

    def forward(
        self,
        x: Tensor,
        attention_mask: Optional[Tensor] = None,
        position_bias: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Args:
            x (Tensor): shape: ``[batch_size, sequence_length, embed_dim]``.
            attention_mask (Tensor or ``None``, optional):
                shape: ``[batch_size, 1, sequence_length, sequence_length]``
            position_bias: Not used. Only for the compatibility with :py:class:`WavLMSelfAttention`.
            key_padding_mask (Tensor or ``None``): Not used. Only for the compatibility with
                :py:class:`WavLMSelfAttention`.
        Returns:
            (Tensor, ``None``): The resulting attention output and ``None`` (necessary for compatibility
                with :py:class:`WavLMSelAttention`).
                Attention output shape: ``[batch, sequence_length, embed_dim]``.
        """
        if x.ndim != 3 or x.shape[2] != self.embed_dim:
            raise ValueError(
                f"The expected input shape is (batch, sequence, embed_dim=={self.embed_dim}). " f"Found {x.shape}."
            )
        batch_size, length, embed_dim = x.size()
        if attention_mask is not None:
            shape_ = (batch_size, 1, length, length)
            if attention_mask.size() != shape_:
                raise ValueError(f"The expected attention mask shape is {shape_}. " f"Found {attention_mask.size()}.")

        shape = (batch_size, length, self.num_heads, self.head_dim)
        q = self.q_proj(x).view(*shape).transpose(2, 1)  # B, nH, L, Hd
        k = self.k_proj(x).view(*shape).transpose(2, 1)  # B, nH, L, Hd
        v = self.v_proj(x).view(*shape).transpose(2, 1)  # B, nH, L, Hd
        dropout = self.dropout if self.training else 0.0
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attention_mask, dropout_p=dropout, is_causal=False
        )
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, -1, self.num_heads * self.head_dim)
        output = self.out_proj(attn_output)
        return output, None  # Necessary for compatibility with WavLMSelAttention


class FeedForward(nn.Module):
    """Layer that follows attention layer in encoder layer."""

    def __init__(
        self,
        io_features: int,
        intermediate_features: int,
        intermediate_dropout: float,
        output_dropout: float,
    ):
        super().__init__()
        self.intermediate_dense = nn.Linear(io_features, intermediate_features)
        self.intermediate_dropout = nn.Dropout(intermediate_dropout)
        self.output_dense = nn.Linear(intermediate_features, io_features)
        self.output_dropout = nn.Dropout(output_dropout)

    def forward(self, x):
        """
        Args:
            x (Tensor): shape: `(batch, sequence_length, io_features)`
        Returns:
            x (Tensor): shape: `(batch, sequence_length, io_features)`
        """
        x = self.intermediate_dense(x)
        x = torch.nn.functional.gelu(x)
        x = self.intermediate_dropout(x)

        x = self.output_dense(x)
        x = self.output_dropout(x)
        return x


class EncoderLayer(nn.Module):
    """A layer unit in encoder. Combines multihead self attention and feed forward."""

    def __init__(
        self,
        attention: nn.Module,
        dropout: float,
        layer_norm_first: bool,
        feed_forward: nn.Module,
    ):
        super().__init__()
        self.attention = attention
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(attention.embed_dim)
        self.layer_norm_first = layer_norm_first
        self.feed_forward = feed_forward
        self.final_layer_norm = nn.LayerNorm(attention.embed_dim)

    def forward(
        self,
        x: Tensor,
        attention_mask: Optional[Tensor] = None,
        position_bias: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Args:
            x (Tensor): Input of shape ``(batch, sequence_length, embed_dim)``.
            attention_mask (Tensor or ``None``, optional): attention mask
                of shape ``(batch, 1, sequence_length, sequence_length)``. (Default: ``None``)
            position_bias (Tensor or ``None``, optional): position bias of shape
                ``(batch_size * num_heads, src_len, src_len)``.
                Only necessary for WavLM model, ``None`` otherwise. (Default: ``None``)
            key_padding_mask (Tensor or ``None``, optional): key padding mask of shape ``(batch_size, src_len)``.
                Only used for WavLM model, ignored otherwise. (Default: ``None``)
        Returns:
            (x, position_bias): Shapes are the same as in the input. Position bias is only relevant for WaLM model,
                ``None`` otherwise.
        """
        residual = x

        if self.layer_norm_first:
            x = self.layer_norm(x)

        x, position_bias = self.attention(
            x, attention_mask=attention_mask, position_bias=position_bias, key_padding_mask=key_padding_mask
        )

        x = self.dropout(x)
        x = residual + x

        if self.layer_norm_first:
            x = x + self.feed_forward(self.final_layer_norm(x))
        else:
            x = self.layer_norm(x)
            x = self.final_layer_norm(x + self.feed_forward(x))
        return x, position_bias


class Transformer(nn.Module):
    def __init__(
        self,
        pos_conv_embed: nn.Module,
        dropout: float,
        layers: nn.Module,
        layer_norm_first: bool,
        layer_drop: float,
    ):
        super().__init__()
        self.pos_conv_embed = pos_conv_embed
        self.layer_norm = nn.LayerNorm(pos_conv_embed.embed_dim)
        self.layer_norm_first = layer_norm_first
        self.layer_drop = layer_drop
        self.dropout = nn.Dropout(dropout)
        self.layers = layers

    def _preprocess(self, x: Tensor):
        x = x + self.pos_conv_embed(x)

        if self.layer_norm_first:
            x = self.layer_norm(x)

        x = self.dropout(x)
        return x

    def forward(
        self,
        x: Tensor,
        attention_mask: Optional[Tensor] = None,
        position_bias: Optional[Tensor] = None,
    ) -> Tensor:
        x = self._preprocess(x)
        for layer in self.layers:
            if not (self.training and torch.rand(1).item() <= self.layer_drop):
                x, position_bias = layer(x, attention_mask, position_bias=position_bias)

        if not self.layer_norm_first:
            x = self.layer_norm(x)
        return x

    def get_intermediate_outputs(
        self,
        x: Tensor,
        attention_mask: Optional[Tensor] = None,
        num_layers: Optional[int] = None,
    ) -> List[Tensor]:
        if num_layers is not None:
            if not 0 < num_layers <= len(self.layers):
                raise ValueError(f"`num_layers` must be between [1, {len(self.layers)}]")

        ret: List[Tensor] = []
        position_bias = None
        x = self._preprocess(x)
        for layer in self.layers:
            x, position_bias = layer(x, attention_mask, position_bias=position_bias)
            ret.append(x)
            if num_layers is not None and len(ret) >= num_layers:
                return ret
        return ret



class FeatureExtractor(nn.Module):
    def __init__(self, kwargs):
        super().__init__()        
        assert kwargs["extractor_mode"] in ["group_norm", "layer_norm"], ValueError("Invalid norm mode")
        
        norm_mode = kwargs["extractor_mode"]


        self.conv_layers = nn.ModuleList()
        in_channels = 1
        for i, (out_channels, kernel_size, stride) in enumerate(kwargs["extractor_conv_config"]):
            normalization = None
            if norm_mode == "group_norm" and i == 0:
                normalization = nn.GroupNorm(
                    num_groups=out_channels,
                    num_channels=out_channels,
                    affine=True,
                )
            elif norm_mode == "layer_norm":
                normalization = nn.LayerNorm(
                    normalized_shape=out_channels,
                    elementwise_affine=True,
                )
            self.conv_layers.append(
                ConvBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    bias=kwargs["extractor_conv_bias"],
                    layer_norm=normalization,
                )
            )
            in_channels = out_channels

    def forward(self, x, length):
        assert x.ndim == 2, ValueError(f"Expected 2D tensor, got {x.ndim}D tensor instead.")

        x = x.unsqueeze(1)
        for layer in self.conv_layers:
            x, length = layer(x, length)
        x = x.transpose(1, 2)
        return x, length


class Encoder(nn.Module):
    def __init__(self, kwargs):
        super().__init__()
        self.feature_projection = FeatureProjection(kwargs["extractor_conv_config"][-1][0], kwargs["encoder_embed_dim"], kwargs["encoder_projection_dropout"])
        pos_conv = ConvolutionalPositionalEmbedding(kwargs["encoder_embed_dim"], kwargs["encoder_pos_conv_kernel"], kwargs["encoder_pos_conv_groups"])

        encoder_layers = nn.ModuleList()
        for _ in range(kwargs["encoder_num_layers"]):
            attention = SelfAttention(
                embed_dim=kwargs["encoder_embed_dim"],
                num_heads=kwargs["encoder_num_heads"],
                dropout=kwargs["encoder_attention_dropout"],
            )
            feed_forward = FeedForward(
                io_features=kwargs["encoder_embed_dim"],
                intermediate_features=kwargs["encoder_ff_interm_features"],
                intermediate_dropout=kwargs["encoder_ff_interm_dropout"],
                output_dropout=kwargs["encoder_dropout"],
            )
            encoder_layers.append(
                EncoderLayer(
                    attention=attention,
                    dropout=kwargs["encoder_dropout"],
                    layer_norm_first=kwargs["encoder_layer_norm_first"],
                    feed_forward=feed_forward,
                )
            )
        self.transformer = Transformer(
            pos_conv_embed=pos_conv,
            dropout=kwargs["encoder_dropout"],
            layers=encoder_layers,
            layer_norm_first=not kwargs["encoder_layer_norm_first"],
            layer_drop=kwargs["encoder_layer_drop"],
        )


    def _preprocess(
        self,
        features,
        lengths= None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        x = self.feature_projection(features)

        mask: Optional[Tensor] = None
        if lengths is not None:
            batch_size, max_len, _ = x.shape
            # create mask for padded elements and zero-out them
            mask = torch.arange(max_len, device=lengths.device).expand(batch_size, max_len) >= lengths[:, None]
            x[mask] = 0.0
            # extend the mask to attention shape and set weight
            mask = -10000.0 * mask[:, None, None, :].to(dtype=features.dtype)
            mask = mask.expand(batch_size, 1, max_len, max_len)
        return x, mask



    def forward(
        self,
        features: Tensor,
        lengths: Optional[Tensor] = None,
    ) -> Tensor:
        x, mask = self._preprocess(features, lengths)
        x = self.transformer(x, attention_mask=mask)
        return x

    def extract_features(
        self,
        features: Tensor,
        lengths: Optional[Tensor] = None,
        num_layers: Optional[int] = None,
    ) -> List[Tensor]:
        x, masks = self._preprocess(features, lengths)
        return self.transformer.get_intermediate_outputs(x, attention_mask=masks, num_layers=num_layers)
    

class Wav2Vec(nn.Module):
    def __init__(self, kwargs):
        super().__init__()
        self.feature_extractor = FeatureExtractor(kwargs)
        self.encoder = Encoder(kwargs)
        self.aux = torch.nn.Linear(kwargs["encoder_embed_dim"], kwargs["aux_num_out"])

    @torch.jit.export
    def extract_features(self, waveforms: Tensor, lengths = None, num_layers = None):
        x, lengths = self.feature_extractor(waveforms, lengths)
        x = self.encoder.extract_features(x, lengths, num_layers)
        return x, lengths

    def forward(self, waveforms: Tensor, lengths = None):
        x, lengths = self.feature_extractor(waveforms, lengths)
        x = self.encoder(x, lengths)
        if self.aux is not None:
            x = self.aux(x)
        return x, lengths