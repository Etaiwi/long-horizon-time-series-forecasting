"""Supervised PatchTST core adapted from the official Apache-2.0 implementation.

Source: https://github.com/yuqinie98/PatchTST
Pinned reconstruction reference: commit 204c21efe0b39603ad6e2ca640ef5896646ab1a9
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _validate_series(series: Tensor, input_length: int, channels: int) -> None:
    if not isinstance(series, Tensor):
        raise TypeError("series must be a PyTorch tensor.")
    if series.ndim != 3:
        raise ValueError(
            "series must have shape [batch, time, channels]; "
            f"received {tuple(series.shape)}."
        )
    if series.shape[1] != input_length:
        raise ValueError(
            f"Expected {input_length} input time steps; received {series.shape[1]}."
        )
    if series.shape[2] != channels:
        raise ValueError(f"Expected {channels} channels; received {series.shape[2]}.")
    if not torch.is_floating_point(series):
        raise TypeError("series must use floating-point values.")


class _Transpose(nn.Module):
    def __init__(self, first: int, second: int) -> None:
        super().__init__()
        self.first = first
        self.second = second

    def forward(self, values: Tensor) -> Tensor:
        return values.transpose(self.first, self.second)


class _RevIN(nn.Module):
    """Reversible instance normalization used by the official supervised model."""

    def __init__(
        self,
        channels: int,
        *,
        affine: bool = False,
        subtract_last: bool = False,
        epsilon: float = 1e-5,
    ) -> None:
        super().__init__()
        self.affine = affine
        self.subtract_last = subtract_last
        self.epsilon = epsilon
        if affine:
            self.affine_weight = nn.Parameter(torch.ones(channels))
            self.affine_bias = nn.Parameter(torch.zeros(channels))

    def normalize(self, values: Tensor) -> Tensor:
        if self.subtract_last:
            self.last = values[:, -1:, :]
            centered = values - self.last
        else:
            self.mean = values.mean(dim=1, keepdim=True).detach()
            centered = values - self.mean

        self.stdev = torch.sqrt(
            values.var(dim=1, keepdim=True, unbiased=False) + self.epsilon
        ).detach()
        normalized = centered / self.stdev
        if self.affine:
            normalized = normalized * self.affine_weight + self.affine_bias
        return normalized

    def denormalize(self, values: Tensor) -> Tensor:
        if self.affine:
            values = (values - self.affine_bias) / (
                self.affine_weight + self.epsilon * self.epsilon
            )
        values = values * self.stdev
        return values + (self.last if self.subtract_last else self.mean)


class _ScaledDotProductAttention(nn.Module):
    def __init__(
        self,
        model_dimension: int,
        attention_heads: int,
        attention_dropout: float,
        *,
        residual_attention: bool,
    ) -> None:
        super().__init__()
        head_dimension = model_dimension // attention_heads
        self.scale = nn.Parameter(
            torch.tensor(head_dimension**-0.5),
            requires_grad=False,
        )
        self.attn_dropout = nn.Dropout(attention_dropout)
        self.residual_attention = residual_attention

    def forward(
        self,
        queries: Tensor,
        keys: Tensor,
        values: Tensor,
        previous_scores: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        scores = torch.matmul(queries, keys) * self.scale
        if previous_scores is not None:
            scores = scores + previous_scores
        weights = self.attn_dropout(F.softmax(scores, dim=-1))
        output = torch.matmul(weights, values)
        return output, weights, scores


class _MultiheadAttention(nn.Module):
    def __init__(
        self,
        model_dimension: int,
        attention_heads: int,
        *,
        attention_dropout: float,
        projection_dropout: float,
        residual_attention: bool,
    ) -> None:
        super().__init__()
        if model_dimension % attention_heads:
            raise ValueError("model_dimension must be divisible by attention_heads.")

        self.n_heads = attention_heads
        self.d_k = model_dimension // attention_heads
        self.d_v = self.d_k
        self.W_Q = nn.Linear(model_dimension, model_dimension)
        self.W_K = nn.Linear(model_dimension, model_dimension)
        self.W_V = nn.Linear(model_dimension, model_dimension)
        self.res_attention = residual_attention
        self.sdp_attn = _ScaledDotProductAttention(
            model_dimension,
            attention_heads,
            attention_dropout,
            residual_attention=residual_attention,
        )
        self.to_out = nn.Sequential(
            nn.Linear(model_dimension, model_dimension),
            nn.Dropout(projection_dropout),
        )

    def forward(
        self,
        queries: Tensor,
        previous_scores: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        batch_size = queries.size(0)
        q = (
            self.W_Q(queries)
            .view(batch_size, -1, self.n_heads, self.d_k)
            .transpose(1, 2)
        )
        k = (
            self.W_K(queries)
            .view(batch_size, -1, self.n_heads, self.d_k)
            .permute(0, 2, 3, 1)
        )
        v = (
            self.W_V(queries)
            .view(batch_size, -1, self.n_heads, self.d_v)
            .transpose(1, 2)
        )
        output, _, scores = self.sdp_attn(q, k, v, previous_scores)
        output = (
            output.transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, self.n_heads * self.d_v)
        )
        return self.to_out(output), scores


class _EncoderLayer(nn.Module):
    def __init__(
        self,
        patch_count: int,
        model_dimension: int,
        attention_heads: int,
        feedforward_dimension: int,
        *,
        dropout: float,
        attention_dropout: float,
    ) -> None:
        super().__init__()
        self.res_attention = True
        self.self_attn = _MultiheadAttention(
            model_dimension,
            attention_heads,
            attention_dropout=attention_dropout,
            projection_dropout=dropout,
            residual_attention=True,
        )
        self.dropout_attn = nn.Dropout(dropout)
        self.norm_attn = nn.Sequential(
            _Transpose(1, 2),
            nn.BatchNorm1d(model_dimension),
            _Transpose(1, 2),
        )
        self.ff = nn.Sequential(
            nn.Linear(model_dimension, feedforward_dimension),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dimension, model_dimension),
        )
        self.dropout_ffn = nn.Dropout(dropout)
        self.norm_ffn = nn.Sequential(
            _Transpose(1, 2),
            nn.BatchNorm1d(model_dimension),
            _Transpose(1, 2),
        )
        self.patch_count = patch_count

    def forward(
        self,
        source: Tensor,
        previous_scores: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        attended, scores = self.self_attn(source, previous_scores)
        source = self.norm_attn(source + self.dropout_attn(attended))
        transformed = self.ff(source)
        source = self.norm_ffn(source + self.dropout_ffn(transformed))
        return source, scores


class _Encoder(nn.Module):
    def __init__(
        self,
        patch_count: int,
        *,
        encoder_layers: int,
        model_dimension: int,
        attention_heads: int,
        feedforward_dimension: int,
        dropout: float,
        attention_dropout: float,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                _EncoderLayer(
                    patch_count,
                    model_dimension,
                    attention_heads,
                    feedforward_dimension,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                )
                for _ in range(encoder_layers)
            ]
        )

    def forward(self, source: Tensor) -> Tensor:
        scores = None
        for layer in self.layers:
            source, scores = layer(source, scores)
        return source


class _ChannelIndependentEncoder(nn.Module):
    def __init__(
        self,
        patch_count: int,
        patch_length: int,
        *,
        encoder_layers: int,
        model_dimension: int,
        attention_heads: int,
        feedforward_dimension: int,
        dropout: float,
        attention_dropout: float,
    ) -> None:
        super().__init__()
        self.patch_num = patch_count
        self.patch_len = patch_length
        self.W_P = nn.Linear(patch_length, model_dimension)
        position = torch.empty(patch_count, model_dimension)
        nn.init.uniform_(position, -0.02, 0.02)
        self.W_pos = nn.Parameter(position)
        self.dropout = nn.Dropout(dropout)
        self.encoder = _Encoder(
            patch_count,
            encoder_layers=encoder_layers,
            model_dimension=model_dimension,
            attention_heads=attention_heads,
            feedforward_dimension=feedforward_dimension,
            dropout=dropout,
            attention_dropout=attention_dropout,
        )

    def forward(self, patches: Tensor) -> Tensor:
        channels = patches.shape[1]
        embedded = self.W_P(patches.permute(0, 1, 3, 2))
        tokens = embedded.reshape(
            embedded.shape[0] * channels,
            embedded.shape[2],
            embedded.shape[3],
        )
        encoded = self.encoder(self.dropout(tokens + self.W_pos))
        encoded = encoded.reshape(
            -1,
            channels,
            encoded.shape[-2],
            encoded.shape[-1],
        )
        return encoded.permute(0, 1, 3, 2)


class _FlattenHead(nn.Module):
    def __init__(
        self,
        channels: int,
        features: int,
        prediction_length: int,
        head_dropout: float,
    ) -> None:
        super().__init__()
        self.individual = False
        self.n_vars = channels
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(features, prediction_length)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, encoded: Tensor) -> Tensor:
        return self.dropout(self.linear(self.flatten(encoded)))


class _PatchTSTBackbone(nn.Module):
    def __init__(
        self,
        *,
        channels: int,
        input_length: int,
        prediction_length: int,
        patch_length: int,
        stride: int,
        encoder_layers: int,
        attention_heads: int,
        model_dimension: int,
        feedforward_dimension: int,
        dropout: float,
        fully_connected_dropout: float,
        attention_dropout: float,
        head_dropout: float,
        revin: bool,
        revin_affine: bool,
        subtract_last: bool,
    ) -> None:
        super().__init__()
        self.revin = revin
        if revin:
            self.revin_layer = _RevIN(
                channels,
                affine=revin_affine,
                subtract_last=subtract_last,
            )

        self.patch_len = patch_length
        self.stride = stride
        self.padding_patch = "end"
        self.padding_patch_layer = nn.ReplicationPad1d((0, stride))
        patch_count = math.floor((input_length - patch_length) / stride) + 2
        self.patch_num = patch_count
        self.backbone = _ChannelIndependentEncoder(
            patch_count,
            patch_length,
            encoder_layers=encoder_layers,
            model_dimension=model_dimension,
            attention_heads=attention_heads,
            feedforward_dimension=feedforward_dimension,
            dropout=dropout,
            attention_dropout=attention_dropout,
        )
        self.head_nf = model_dimension * patch_count
        self.head = _FlattenHead(
            channels,
            self.head_nf,
            prediction_length,
            head_dropout,
        )
        self.fully_connected_dropout = fully_connected_dropout

    def forward(self, values: Tensor) -> Tensor:
        if self.revin:
            values = self.revin_layer.normalize(values.transpose(1, 2))
            values = values.transpose(1, 2)

        patches = self.padding_patch_layer(values).unfold(
            dimension=-1,
            size=self.patch_len,
            step=self.stride,
        )
        encoded = self.backbone(patches.permute(0, 1, 3, 2))
        forecast = self.head(encoded)

        if self.revin:
            forecast = self.revin_layer.denormalize(forecast.transpose(1, 2))
            forecast = forecast.transpose(1, 2)
        return forecast


class PatchTST(nn.Module):
    """Channel-independent PatchTST/42 for direct multivariate forecasting."""

    def __init__(
        self,
        *,
        input_length: int,
        prediction_length: int,
        channels: int,
        patch_length: int = 16,
        stride: int = 8,
        encoder_layers: int = 3,
        attention_heads: int = 4,
        model_dimension: int = 16,
        feedforward_dimension: int = 128,
        dropout: float = 0.3,
        fully_connected_dropout: float = 0.3,
        attention_dropout: float = 0.0,
        head_dropout: float = 0.0,
        revin: bool = True,
        revin_affine: bool = False,
        subtract_last: bool = False,
    ) -> None:
        super().__init__()
        if input_length <= 0 or prediction_length <= 0 or channels <= 0:
            raise ValueError("Window lengths and channels must be positive.")
        if patch_length <= 0 or stride <= 0 or patch_length > input_length:
            raise ValueError("Patch length and stride are invalid for this input.")

        self.input_length = input_length
        self.prediction_length = prediction_length
        self.channels = channels
        self.model = _PatchTSTBackbone(
            channels=channels,
            input_length=input_length,
            prediction_length=prediction_length,
            patch_length=patch_length,
            stride=stride,
            encoder_layers=encoder_layers,
            attention_heads=attention_heads,
            model_dimension=model_dimension,
            feedforward_dimension=feedforward_dimension,
            dropout=dropout,
            fully_connected_dropout=fully_connected_dropout,
            attention_dropout=attention_dropout,
            head_dropout=head_dropout,
            revin=revin,
            revin_affine=revin_affine,
            subtract_last=subtract_last,
        )

    @property
    def patch_count(self) -> int:
        return self.model.patch_num

    def forward(self, series: Tensor) -> Tensor:
        _validate_series(series, self.input_length, self.channels)
        return self.model(series.transpose(1, 2)).transpose(1, 2)


__all__ = ["PatchTST"]
