from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .config import ModelConfig


class PatchEncoder(nn.Module):
    def __init__(self, patch_size: int, embedding_dim: int, dropout: float):
        super().__init__()
        self.patch_size = patch_size
        self.network = nn.Sequential(
            nn.Linear(patch_size, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        if waveforms.ndim != 3:
            raise ValueError("Expected waveforms with shape [batch, channels, samples]")
        samples = waveforms.shape[-1]
        usable = samples - (samples % self.patch_size)
        if usable < self.patch_size:
            raise ValueError(f"Need at least {self.patch_size} samples; got {samples}")
        patches = waveforms[..., :usable].unfold(-1, self.patch_size, self.patch_size)
        patches = patches.permute(0, 2, 1, 3)  # [batch, time, channels, patch]
        return self.network(patches)


class TemporalGraphBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        window_size: int,
        stride: int,
        temporal_decay: float,
        dropout: float,
    ):
        super().__init__()
        self.window_size = window_size
        self.stride = stride
        self.temporal_decay = temporal_decay
        self.graph_projection = nn.Linear(input_dim, input_dim, bias=False)
        self.message_projection = nn.Linear(input_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)

    def _prior(self, channels: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        time_index = torch.arange(self.window_size, device=device).repeat_interleave(channels)
        distance = (time_index[:, None] - time_index[None, :]).abs()
        return torch.pow(torch.as_tensor(self.temporal_decay, device=device, dtype=dtype), distance)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        _, time_steps, channels, feature_dim = features.shape
        if time_steps < self.window_size:
            raise ValueError(
                f"Patch sequence has {time_steps} steps, below graph window {self.window_size}"
            )
        windows = []
        for start in range(0, time_steps - self.window_size + 1, self.stride):
            window = features[:, start : start + self.window_size]
            nodes = window.reshape(window.shape[0], self.window_size * channels, feature_dim)
            queries = F.normalize(self.graph_projection(nodes), dim=-1)
            similarity = torch.bmm(queries, queries.transpose(1, 2)) / math.sqrt(feature_dim)
            similarity = similarity + torch.log(
                self._prior(channels, nodes.device, nodes.dtype).clamp_min(1e-6)
            )
            adjacency = torch.softmax(similarity, dim=-1)
            messages = torch.bmm(adjacency, nodes)
            messages = self.dropout(F.gelu(self.norm(self.message_projection(messages))))
            windows.append(
                messages.reshape(nodes.shape[0], self.window_size, channels, -1).mean(dim=1)
            )
        return torch.stack(windows, dim=1)  # [batch, windows, channels, hidden]


class CleanSTGNN(nn.Module):
    """Subject-agnostic waveform backbone used by the public pipeline.

    The implementation preserves the paper's patch embedding, dynamic
    spatio-temporal graph, temporal decay, and multi-window aggregation. It is
    intentionally free of subject-ID normalization and private project hooks.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        if len(config.window_sizes) != len(config.window_strides):
            raise ValueError("window_sizes and window_strides must have equal length")
        self.config = config
        self.patch_encoder = PatchEncoder(config.patch_size, config.embedding_dim, config.dropout)
        self.graph_blocks = nn.ModuleList(
            [
                TemporalGraphBlock(
                    config.embedding_dim,
                    config.graph_hidden_dim,
                    window,
                    stride,
                    config.temporal_decay,
                    config.dropout,
                )
                for window, stride in zip(config.window_sizes, config.window_strides)
            ]
        )
        pooled_dim = 2 * config.graph_hidden_dim * len(self.graph_blocks)
        self.encoder = nn.Sequential(
            nn.Linear(pooled_dim, 2 * config.latent_dim),
            nn.LayerNorm(2 * config.latent_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(2 * config.latent_dim, config.latent_dim),
            nn.LayerNorm(config.latent_dim),
            nn.GELU(),
        )
        self.classifier = nn.Linear(config.latent_dim, config.num_classes)

    def forward(self, waveforms: torch.Tensor, return_features: bool = False):
        patches = self.patch_encoder(waveforms)
        pooled = []
        for block in self.graph_blocks:
            branch = block(patches)
            pooled.extend([branch.mean(dim=(1, 2)), branch.amax(dim=(1, 2))])
        latent = self.encoder(torch.cat(pooled, dim=-1))
        logits = self.classifier(latent)
        return (logits, latent) if return_features else logits

    @property
    def trainable_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
