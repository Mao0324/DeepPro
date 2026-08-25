"""Structural adapters for DeepPro-Plus ablations.

All modules preserve ``[B,C,T,H,W]`` shape and keep an explicit appearance
path.  Historical variants retain a zero-initialized final projection for
reproducibility.  Scratch-specific variants use a small non-zero residual
projection so gradients reach the complete adapter on the first update.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


STRUCTURE_VARIANTS = (
    'raw_apmd',
    'raw_apmd_rms',
    'raw_apmd_channel_rms',
    'raw_apmd_motion_detrend',
    'raw_apmd_multiscale_contrast',
    'raw_apmd_hybrid_rms',
    'raw_apmd_hybrid_rms_scratch_init',
    'raw_apmd_hybrid_rms_scratch_bandpass',
    'raw_apmd_hybrid_rms_scratch_detail',
    'raw_apmd_hybrid_rms_motion_detrend',
    'raw_apmd_hybrid_rms_multiscale_contrast',
    'raw_apmd_hybrid_rms_motion_detrend_multiscale_contrast',
    'second_order',
    'lfp_shallow',
    'lfp_deep',
    'global_align',
    'local_align',
    'multiscale_head',
    'bidirectional',
    'tdc_dual_stream',
)


def _valid_frame_mask(raw_frames):
    """Identify exact zero-padding inserted by the sequence loaders."""
    if raw_frames.ndim != 5:
        raise ValueError(
            'raw_frames must have shape [B,C,T,H,W], got %s'
            % (tuple(raw_frames.shape),)
        )
    return raw_frames.detach().abs().amax(
        dim=(1, 3, 4), keepdim=True
    ).ne(0)


def _valid_temporal_neighbours(features, valid_mask, step):
    """Build temporal neighbours without treating zero-padding as motion."""
    if step <= 0:
        raise ValueError('temporal step must be positive')
    batch, _, time, _, _ = features.shape
    if valid_mask.shape != (batch, 1, time, 1, 1):
        raise ValueError(
            'valid_mask has shape %s, expected %s'
            % (
                tuple(valid_mask.shape),
                (batch, 1, time, 1, 1),
            )
        )
    if step >= time:
        return features, features

    previous = torch.cat([
        features[:, :, :1].expand(-1, -1, step, -1, -1),
        features[:, :, :-step],
    ], dim=2)
    following = torch.cat([
        features[:, :, step:],
        features[:, :, -1:].expand(-1, -1, step, -1, -1),
    ], dim=2)
    false_padding = torch.zeros_like(valid_mask[:, :, :step])
    previous_valid = torch.cat([
        false_padding, valid_mask[:, :, :-step]
    ], dim=2)
    following_valid = torch.cat([
        valid_mask[:, :, step:], false_padding
    ], dim=2)
    # For a valid query near a padded boundary, using the query itself is a
    # neutral temporal neighbour and therefore cannot create a padding edge.
    previous = torch.where(previous_valid, previous, features)
    following = torch.where(following_valid, following, features)
    return previous, following


def _group_count(channels, maximum=4):
    for groups in range(min(maximum, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


def _zero_projection(in_channels, out_channels):
    projection = nn.Conv3d(
        in_channels, out_channels, kernel_size=1, bias=False
    )
    nn.init.zeros_(projection.weight)
    return projection


def _scaled_projection(in_channels, out_channels, init_scale):
    """Create a residual projection with controlled scratch initialization."""
    init_scale = float(init_scale)
    if init_scale < 0.0:
        raise ValueError('projection init scale must be non-negative')
    if init_scale == 0.0:
        return _zero_projection(in_channels, out_channels)
    projection = nn.Conv3d(
        in_channels, out_channels, kernel_size=1, bias=False
    )
    nn.init.kaiming_normal_(projection.weight, nonlinearity='linear')
    with torch.no_grad():
        projection.weight.mul_(init_scale)
    return projection


def _masked_temporal_average(features, valid_mask, kernel_size):
    """Average valid temporal samples without turning padding into signal."""
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError('temporal averaging kernel must be positive and odd')
    valid = valid_mask.to(dtype=features.dtype)
    pooling_kernel = (kernel_size, 1, 1)
    pooling_padding = (kernel_size // 2, 0, 0)
    numerator = F.avg_pool3d(
        features * valid,
        kernel_size=pooling_kernel,
        stride=1,
        padding=pooling_padding,
    )
    denominator = F.avg_pool3d(
        valid,
        kernel_size=pooling_kernel,
        stride=1,
        padding=pooling_padding,
    )
    return torch.where(
        denominator > 0,
        numerator / denominator.clamp_min(1e-6),
        torch.zeros_like(numerator),
    )


def _local_contrast(x):
    near = F.avg_pool3d(
        x, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1)
    )
    wide = F.avg_pool3d(
        x, kernel_size=(1, 7, 7), stride=1, padding=(0, 3, 3)
    )
    return x - 0.5 * (near + wide)


def _temporal_neighbours(x, step):
    if step <= 0 or step >= x.shape[2]:
        raise ValueError(
            'temporal step must be in [1,T), got %d for T=%d'
            % (step, x.shape[2])
        )
    first = x[:, :, :1].expand(-1, -1, step, -1, -1)
    last = x[:, :, -1:].expand(-1, -1, step, -1, -1)
    previous = torch.cat([first, x[:, :, :-step]], dim=2)
    following = torch.cat([x[:, :, step:], last], dim=2)
    return previous, following


def _reduction(channels, bottleneck_channels):
    return nn.Sequential(
        nn.Conv3d(channels, bottleneck_channels, kernel_size=1, bias=False),
        nn.GroupNorm(
            _group_count(bottleneck_channels), bottleneck_channels
        ),
        nn.SiLU(inplace=True),
    )


def _fusion(in_channels, bottleneck_channels):
    return nn.Sequential(
        nn.Conv3d(
            in_channels, bottleneck_channels, kernel_size=1, bias=False
        ),
        nn.GroupNorm(
            _group_count(bottleneck_channels), bottleneck_channels
        ),
        nn.SiLU(inplace=True),
    )


def _reliability_gate(bottleneck_channels, input_multiplier=2):
    gate = nn.Sequential(
        nn.Conv3d(
            bottleneck_channels * input_multiplier,
            bottleneck_channels,
            kernel_size=1,
            bias=False,
        ),
        nn.GroupNorm(
            _group_count(bottleneck_channels), bottleneck_channels
        ),
        nn.SiLU(inplace=True),
        nn.Conv3d(bottleneck_channels, 1, kernel_size=1),
        nn.Sigmoid(),
    )
    nn.init.zeros_(gate[-2].weight)
    nn.init.constant_(gate[-2].bias, -2.0)
    return gate


class _FramewiseGroupNorm(nn.GroupNorm):
    """Apply GroupNorm independently to each frame of a 5D sequence."""

    def forward(self, features):
        if features.ndim != 5:
            return super().forward(features)
        batch, channels, time, height, width = features.shape
        frames = features.permute(0, 2, 1, 3, 4).reshape(
            batch * time, channels, height, width
        )
        frames = super().forward(frames)
        return frames.view(
            batch, time, channels, height, width
        ).permute(0, 2, 1, 3, 4).contiguous()


class _FramewiseRMSNorm(nn.Module):
    """Scale each frame without subtracting its absolute intensity mean."""

    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1, 1))

    def forward(self, features):
        if features.ndim != 5:
            raise ValueError(
                'FramewiseRMSNorm expects [B,C,T,H,W], got %s'
                % (tuple(features.shape),)
            )
        rms = features.float().square().mean(
            dim=(1, 3, 4), keepdim=True
        ).add(self.eps).sqrt().to(dtype=features.dtype)
        return features / rms * self.weight


class _FramewiseChannelRMSNorm(nn.Module):
    """Scale each frame/filter independently without subtracting its mean."""

    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1, 1))

    def forward(self, features):
        if features.ndim != 5:
            raise ValueError(
                'FramewiseChannelRMSNorm expects [B,C,T,H,W], got %s'
                % (tuple(features.shape),)
            )
        rms = features.float().square().mean(
            dim=(3, 4), keepdim=True
        ).add(self.eps).sqrt().to(dtype=features.dtype)
        return features / rms * self.weight


class _FramewiseHybridRMSNorm(nn.Module):
    """Shrink per-channel energy estimates toward the shared frame energy."""

    def __init__(self, channels, eps=1e-6, initial_channel_mix=0.5):
        super().__init__()
        if not 0.0 < initial_channel_mix < 1.0:
            raise ValueError('initial_channel_mix must be in (0, 1)')
        self.eps = float(eps)
        initial_logit = torch.logit(torch.tensor(initial_channel_mix))
        self.channel_mix_logit = nn.Parameter(
            initial_logit.expand(1, channels, 1, 1, 1).clone()
        )
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1, 1))

    def forward(self, features):
        if features.ndim != 5:
            raise ValueError(
                'FramewiseHybridRMSNorm expects [B,C,T,H,W], got %s'
                % (tuple(features.shape),)
            )
        squared = features.float().square()
        shared_moment = squared.mean(
            dim=(1, 3, 4), keepdim=True
        )
        channel_moment = squared.mean(dim=(3, 4), keepdim=True)
        channel_mix = torch.sigmoid(self.channel_mix_logit).float()
        blended_moment = torch.lerp(
            shared_moment, channel_moment, channel_mix
        )
        rms = blended_moment.add(self.eps).sqrt().to(
            dtype=features.dtype
        )
        return features / rms * self.weight

    def channel_mix(self):
        return torch.sigmoid(self.channel_mix_logit)


class RawAppearanceMotionAdapter(nn.Module):
    """Preserve raw appearance and add multi-scale first/second dynamics.

    The backbone difference stream remains the identity path.  This adapter
    separately encodes each raw frame, estimates motion at temporal offsets
    1/2/4, and injects an additive residual without a gate suppressing weak
    target evidence.  ``projection_init_scale=0`` reproduces the historical
    identity initialization; a small positive scale is intended for training
    the whole detector from scratch.
    """

    def __init__(
        self,
        channels=32,
        bottleneck_channels=8,
        temporal_steps=(1, 2, 4),
        normalization='group',
        motion_detrend=False,
        adaptive_contrast=False,
        projection_init_scale=0.0,
        temporal_bandpass=False,
        backbone_detail=False,
    ):
        super().__init__()
        if not temporal_steps or any(step <= 0 for step in temporal_steps):
            raise ValueError('temporal_steps must contain positive integers')
        self.channels = int(channels)
        self.bottleneck_channels = int(bottleneck_channels)
        self.temporal_steps = tuple(int(step) for step in temporal_steps)
        if normalization not in {
            'group', 'rms', 'channel_rms', 'hybrid_rms'
        }:
            raise ValueError(
                'normalization must be group, rms, channel_rms, or hybrid_rms'
            )
        self.normalization = normalization
        self.motion_detrend = bool(motion_detrend)
        self.adaptive_contrast = bool(adaptive_contrast)
        self.projection_init_scale = float(projection_init_scale)
        if self.projection_init_scale < 0.0:
            raise ValueError('projection_init_scale must be non-negative')
        self.temporal_bandpass = bool(temporal_bandpass)
        self.backbone_detail = bool(backbone_detail)
        self.motion_detrend_kernel = 15
        self.contrast_kernel_sizes = (3, 5, 7)
        self.temporal_bandpass_kernels = (3, 9)

        groups = _group_count(self.bottleneck_channels)

        def frame_norm():
            if self.normalization == 'rms':
                return _FramewiseRMSNorm(self.bottleneck_channels)
            if self.normalization == 'channel_rms':
                return _FramewiseChannelRMSNorm(
                    self.bottleneck_channels
                )
            if self.normalization == 'hybrid_rms':
                return _FramewiseHybridRMSNorm(
                    self.bottleneck_channels
                )
            return _FramewiseGroupNorm(
                groups, self.bottleneck_channels
            )
        self.appearance_encoder = nn.Sequential(
            nn.Conv3d(
                1,
                self.bottleneck_channels,
                kernel_size=(1, 5, 5),
                padding=(0, 2, 2),
                bias=False,
            ),
            frame_norm(),
            nn.SiLU(inplace=True),
            nn.Conv3d(
                self.bottleneck_channels,
                self.bottleneck_channels,
                kernel_size=(1, 3, 3),
                padding=(0, 1, 1),
                groups=self.bottleneck_channels,
                bias=False,
            ),
            frame_norm(),
            nn.SiLU(inplace=True),
        )
        scale_count = len(self.temporal_steps)
        self.first_order_scale_logits = nn.Parameter(torch.zeros(
            scale_count, self.bottleneck_channels
        ))
        self.second_order_scale_logits = nn.Parameter(torch.zeros(
            scale_count, self.bottleneck_channels
        ))
        if self.adaptive_contrast:
            self.contrast_scale_logits = nn.Parameter(torch.zeros(
                len(self.contrast_kernel_sizes),
                self.bottleneck_channels,
            ))
        self.motion_fusion = nn.Sequential(
            nn.Conv3d(
                self.bottleneck_channels * 2,
                self.bottleneck_channels,
                kernel_size=1,
                bias=False,
            ),
            frame_norm(),
            nn.SiLU(inplace=True),
        )
        if self.backbone_detail:
            self.backbone_detail_encoder = nn.Sequential(
                nn.Conv3d(
                    self.channels,
                    self.bottleneck_channels,
                    kernel_size=1,
                    bias=False,
                ),
                frame_norm(),
                nn.SiLU(inplace=True),
            )
        domain_count = 3 + int(self.temporal_bandpass) + int(
            self.backbone_detail
        )
        self.domain_fusion = nn.Sequential(
            nn.Conv3d(
                self.bottleneck_channels * domain_count,
                self.bottleneck_channels,
                kernel_size=(1, 3, 3),
                padding=(0, 1, 1),
                bias=False,
            ),
            frame_norm(),
            nn.SiLU(inplace=True),
        )
        self.projection = _scaled_projection(
            self.bottleneck_channels,
            self.channels,
            self.projection_init_scale,
        )

    @staticmethod
    def _channel_scale_weights(logits):
        return torch.softmax(logits, dim=0).view(
            logits.shape[0], 1, logits.shape[1], 1, 1, 1
        )

    def _detrend_motion(self, context):
        kernel = self.motion_detrend_kernel
        low_frequency = F.avg_pool3d(
            context,
            kernel_size=(1, kernel, kernel),
            stride=1,
            padding=(0, kernel // 2, kernel // 2),
            count_include_pad=False,
        )
        return context - low_frequency

    def _adaptive_local_contrast(self, appearance):
        weights = self._channel_scale_weights(
            self.contrast_scale_logits
        )
        contrast = torch.zeros_like(appearance)
        for index, kernel in enumerate(self.contrast_kernel_sizes):
            surround = F.avg_pool3d(
                appearance,
                kernel_size=(1, kernel, kernel),
                stride=1,
                padding=(0, kernel // 2, kernel // 2),
                count_include_pad=False,
            )
            contrast = contrast + weights[index] * (
                appearance - surround
            )
        return contrast, weights

    def forward(self, backbone_features, raw_frames, return_aux=False):
        if backbone_features.ndim != 5 or raw_frames.ndim != 5:
            raise ValueError(
                'backbone_features and raw_frames must both be 5D tensors'
            )
        if backbone_features.shape[0] != raw_frames.shape[0]:
            raise ValueError('backbone_features/raw_frames batch mismatch')
        if backbone_features.shape[2:] != raw_frames.shape[2:]:
            raise ValueError('backbone_features/raw_frames T/H/W mismatch')
        if backbone_features.shape[1] != self.channels:
            raise ValueError(
                'expected %d backbone channels, got %d'
                % (self.channels, backbone_features.shape[1])
            )
        if raw_frames.shape[1] != 1:
            raise ValueError(
                'raw appearance branch expects one input channel, got %d'
                % raw_frames.shape[1]
            )

        valid_mask = _valid_frame_mask(raw_frames)
        valid = valid_mask.to(dtype=raw_frames.dtype)
        appearance = self.appearance_encoder(raw_frames) * valid
        first_weights = self._channel_scale_weights(
            self.first_order_scale_logits
        )
        second_weights = self._channel_scale_weights(
            self.second_order_scale_logits
        )
        first_context = torch.zeros_like(appearance)
        second_context = torch.zeros_like(appearance)
        for index, step in enumerate(self.temporal_steps):
            previous, following = _valid_temporal_neighbours(
                appearance, valid_mask, step
            )
            first_order = 0.5 * (following - previous)
            second_order = following - 2.0 * appearance + previous
            first_context = (
                first_context + first_weights[index] * first_order
            )
            second_context = (
                second_context + second_weights[index] * second_order
            )
        first_context = first_context * valid
        second_context = second_context * valid
        if self.motion_detrend:
            first_context = self._detrend_motion(first_context) * valid
            second_context = self._detrend_motion(second_context) * valid
        motion = self.motion_fusion(torch.cat([
            first_context, second_context
        ], dim=1)) * valid
        contrast_weights = None
        if self.adaptive_contrast:
            contrast, contrast_weights = self._adaptive_local_contrast(
                appearance
            )
        else:
            contrast = _local_contrast(appearance)
        contrast = contrast * valid
        domains = [appearance, motion, contrast]
        temporal_bandpass = None
        if self.temporal_bandpass:
            short_kernel, long_kernel = self.temporal_bandpass_kernels
            short_context = _masked_temporal_average(
                appearance, valid_mask, short_kernel
            )
            long_context = _masked_temporal_average(
                appearance, valid_mask, long_kernel
            )
            temporal_bandpass = (short_context - long_context) * valid
            domains.append(temporal_bandpass)
        backbone_detail = None
        if self.backbone_detail:
            backbone_detail = self.backbone_detail_encoder(
                backbone_features * valid
            ) * valid
            domains.append(backbone_detail)
        fused = self.domain_fusion(torch.cat(domains, dim=1)) * valid
        delta = self.projection(fused)
        output = backbone_features + delta
        if not return_aux:
            return output
        auxiliary = {
            'raw_appearance': appearance,
            'first_order_context': first_context,
            'second_order_context': second_context,
            'local_contrast': contrast,
            'valid_frame_mask': valid_mask,
            'first_order_scale_weights': first_weights[:, 0, :, 0, 0, 0],
            'second_order_scale_weights': second_weights[:, 0, :, 0, 0, 0],
            'residual_delta': delta,
        }
        if temporal_bandpass is not None:
            auxiliary['temporal_bandpass'] = temporal_bandpass
        if backbone_detail is not None:
            auxiliary['backbone_detail'] = backbone_detail
        if contrast_weights is not None:
            auxiliary['contrast_scale_weights'] = (
                contrast_weights[:, 0, :, 0, 0, 0]
            )
        return output, auxiliary


class SecondOrderMotionAdapter(nn.Module):
    """Fuse appearance with first- and second-order temporal anomalies."""

    def __init__(self, channels=32, bottleneck_channels=8):
        super().__init__()
        self.reduce = _reduction(channels, bottleneck_channels)
        steps = (1, 2, 4)
        self.steps = steps
        self.motion_fusion = _fusion(
            bottleneck_channels * len(steps) * 2,
            bottleneck_channels,
        )
        self.fusion = _fusion(bottleneck_channels * 3, bottleneck_channels)
        self.gate = _reliability_gate(bottleneck_channels)
        self.projection = _zero_projection(bottleneck_channels, channels)

    def forward(self, x, return_aux=False):
        appearance = self.reduce(x)
        motion_parts = []
        for step in self.steps:
            previous, following = _temporal_neighbours(appearance, step)
            first_order = 0.5 * (following - previous)
            second_order = following - 2.0 * appearance + previous
            motion_parts.extend([first_order, second_order])
        motion = self.motion_fusion(torch.cat(motion_parts, dim=1))
        contrast = _local_contrast(appearance)
        fused = self.fusion(torch.cat([appearance, motion, contrast], dim=1))
        reliability = self.gate(
            torch.cat([motion.abs(), contrast.abs()], dim=1)
        )
        delta = self.projection(fused)
        output = x + reliability * delta
        if not return_aux:
            return output
        return output, {
            'motion': motion,
            'local_contrast': contrast,
            'reliability_gate': reliability,
            'residual_delta': delta,
        }


class LowFrequencyPurificationAdapter(nn.Module):
    """Low-frequency-guided high-frequency purification inspired by LFP."""

    def __init__(self, channels=32, bottleneck_channels=8):
        super().__init__()
        self.bottleneck_channels = bottleneck_channels
        self.reduce = _reduction(channels, bottleneck_channels)
        coefficients = torch.tensor(
            [1.0, 4.0, 6.0, 4.0, 1.0], dtype=torch.float32
        )
        kernel = torch.outer(coefficients, coefficients)
        kernel = kernel / kernel.sum()
        self.register_buffer(
            'gaussian_kernel', kernel.view(1, 1, 1, 5, 5)
        )
        self.attention = nn.Sequential(
            nn.Conv3d(
                2, 8, kernel_size=(1, 3, 3), padding=(0, 1, 1),
                bias=False,
            ),
            nn.SiLU(inplace=True),
            nn.Conv3d(8, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        self.fusion = _fusion(bottleneck_channels * 3, bottleneck_channels)
        self.projection = _zero_projection(bottleneck_channels, channels)

    def _gaussian_blur(self, x):
        kernel = self.gaussian_kernel.to(dtype=x.dtype).expand(
            self.bottleneck_channels, 1, 1, 5, 5
        )
        return F.conv3d(
            x, kernel, padding=(0, 2, 2), groups=self.bottleneck_channels
        )

    def forward(self, x, return_aux=False):
        appearance = self.reduce(x)
        low_frequency = self._gaussian_blur(appearance)
        high_frequency = appearance - low_frequency
        low_statistics = torch.cat([
            low_frequency.mean(dim=1, keepdim=True),
            low_frequency.amax(dim=1, keepdim=True),
        ], dim=1)
        attention = self.attention(low_statistics)
        purified_high = self._gaussian_blur(attention * high_frequency)
        fused = self.fusion(torch.cat([
            appearance, low_frequency, purified_high
        ], dim=1))
        delta = self.projection(fused)
        output = x + attention * delta
        if not return_aux:
            return output
        return output, {
            'low_frequency': low_frequency,
            'purified_high_frequency': purified_high,
            'frequency_attention': attention,
            'residual_delta': delta,
        }


class _AlignmentAdapter(nn.Module):
    def __init__(
        self,
        channels=32,
        bottleneck_channels=8,
        max_shift=4.0,
        dense_flow=False,
    ):
        super().__init__()
        if max_shift <= 0:
            raise ValueError('max_shift must be positive')
        self.max_shift = float(max_shift)
        self.dense_flow = bool(dense_flow)
        self.reduce = _reduction(channels, bottleneck_channels)
        hidden_channels = max(8, bottleneck_channels)
        self.motion_estimator = nn.Sequential(
            nn.Conv2d(
                bottleneck_channels * 2,
                hidden_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(_group_count(hidden_channels), hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, 2, kernel_size=3, padding=1),
        )
        nn.init.zeros_(self.motion_estimator[-1].weight)
        nn.init.zeros_(self.motion_estimator[-1].bias)
        self.fusion = _fusion(bottleneck_channels * 4, bottleneck_channels)
        self.gate = _reliability_gate(bottleneck_channels)
        self.projection = _zero_projection(bottleneck_channels, channels)

    @staticmethod
    def _base_grid(batch, height, width, device, dtype):
        y = (
            (torch.arange(height, device=device, dtype=dtype) + 0.5)
            * (2.0 / float(height)) - 1.0
        )
        x = (
            (torch.arange(width, device=device, dtype=dtype) + 0.5)
            * (2.0 / float(width)) - 1.0
        )
        grid_y = y[:, None].expand(height, width)
        grid_x = x[None, :].expand(height, width)
        return torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).expand(
            batch, -1, -1, -1
        )

    def _estimate_flow(self, appearance, reference):
        batch, channels, time, height, width = appearance.shape
        current_2d = appearance.permute(0, 2, 1, 3, 4).reshape(
            batch * time, channels, height, width
        )
        reference_2d = reference.permute(0, 2, 1, 3, 4).reshape(
            batch * time, channels, height, width
        )
        estimator_input = F.avg_pool2d(
            torch.cat([current_2d, reference_2d], dim=1),
            kernel_size=4,
            stride=4,
            ceil_mode=True,
        )
        flow_low = self.motion_estimator(estimator_input)
        if self.dense_flow:
            flow = F.interpolate(
                flow_low,
                size=(height, width),
                mode='bilinear',
                align_corners=False,
            )
        else:
            flow = flow_low.mean(dim=(2, 3), keepdim=True).expand(
                -1, -1, height, width
            )
        return torch.tanh(flow) * self.max_shift

    def _warp(self, appearance, flow):
        batch, channels, time, height, width = appearance.shape
        feature_2d = appearance.permute(0, 2, 1, 3, 4).reshape(
            batch * time, channels, height, width
        )
        grid = self._base_grid(
            batch * time,
            height,
            width,
            feature_2d.device,
            feature_2d.dtype,
        ).clone()
        grid[..., 0] += flow[:, 0] * (2.0 / float(width))
        grid[..., 1] += flow[:, 1] * (2.0 / float(height))
        aligned = F.grid_sample(
            feature_2d,
            grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=False,
        )
        return aligned.view(batch, time, channels, height, width).permute(
            0, 2, 1, 3, 4
        )

    def forward(self, x, return_aux=False):
        appearance = self.reduce(x)
        reference = appearance[:, :, appearance.shape[2] // 2:][:, :, :1]
        reference = reference.expand(-1, -1, appearance.shape[2], -1, -1)
        flow = self._estimate_flow(appearance, reference)
        aligned = self._warp(appearance, flow)
        anomaly = aligned - reference
        contrast = _local_contrast(aligned)
        fused = self.fusion(torch.cat([
            appearance, aligned, anomaly, contrast
        ], dim=1))
        reliability = self.gate(
            torch.cat([anomaly.abs(), contrast.abs()], dim=1)
        )
        delta = self.projection(fused)
        output = x + reliability * delta
        if not return_aux:
            return output
        batch, _, time, height, width = appearance.shape
        flow_5d = flow.view(batch, time, 2, height, width).permute(
            0, 2, 1, 3, 4
        )
        return output, {
            'flow': flow_5d,
            'aligned_anomaly': anomaly,
            'reliability_gate': reliability,
            'residual_delta': delta,
        }


class GlobalMotionAlignmentAdapter(_AlignmentAdapter):
    def __init__(self, channels=32, bottleneck_channels=8, max_shift=4.0):
        super().__init__(
            channels, bottleneck_channels, max_shift, dense_flow=False
        )


class LocalMotionAlignmentAdapter(_AlignmentAdapter):
    def __init__(self, channels=32, bottleneck_channels=8, max_shift=4.0):
        super().__init__(
            channels, bottleneck_channels, max_shift, dense_flow=True
        )


class MultiScaleContextAdapter(nn.Module):
    """Full-resolution multi-scale spatial head with residual fusion."""

    def __init__(self, channels=32, bottleneck_channels=8):
        super().__init__()
        self.reduce = _reduction(channels, bottleneck_channels)
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(
                    bottleneck_channels,
                    bottleneck_channels,
                    kernel_size=(1, 3, 3),
                    padding=(0, dilation, dilation),
                    dilation=(1, dilation, dilation),
                    groups=bottleneck_channels,
                    bias=False,
                ),
                nn.GroupNorm(
                    _group_count(bottleneck_channels), bottleneck_channels
                ),
                nn.SiLU(inplace=True),
            )
            for dilation in (1, 2, 3)
        ])
        self.fusion = _fusion(bottleneck_channels * 5, bottleneck_channels)
        self.gate = _reliability_gate(bottleneck_channels)
        self.projection = _zero_projection(bottleneck_channels, channels)

    def forward(self, x, return_aux=False):
        appearance = self.reduce(x)
        branches = [branch(appearance) for branch in self.branches]
        contrast = _local_contrast(appearance)
        context = self.fusion(torch.cat(
            [appearance, contrast] + branches, dim=1
        ))
        reliability = self.gate(
            torch.cat([context.abs(), contrast.abs()], dim=1)
        )
        delta = self.projection(context)
        output = x + reliability * delta
        if not return_aux:
            return output
        return output, {
            'context': context,
            'local_contrast': contrast,
            'reliability_gate': reliability,
            'residual_delta': delta,
        }


class _DepthwiseConvGRUCell(nn.Module):
    def __init__(self, channels):
        super().__init__()
        joined_channels = channels * 2
        self.gates = nn.Sequential(
            nn.Conv2d(
                joined_channels, joined_channels, kernel_size=3, padding=1,
                groups=joined_channels, bias=False,
            ),
            nn.Conv2d(joined_channels, channels * 2, kernel_size=1),
        )
        self.candidate = nn.Sequential(
            nn.Conv2d(
                joined_channels, joined_channels, kernel_size=3, padding=1,
                groups=joined_channels, bias=False,
            ),
            nn.Conv2d(joined_channels, channels, kernel_size=1),
        )

    def forward(self, x, state):
        reset, update = torch.sigmoid(
            self.gates(torch.cat([x, state], dim=1))
        ).chunk(2, dim=1)
        candidate = torch.tanh(
            self.candidate(torch.cat([x, reset * state], dim=1))
        )
        return (1.0 - update) * state + update * candidate


class BidirectionalPropagationAdapter(nn.Module):
    """Lightweight forward/backward recurrent propagation at full resolution."""

    def __init__(self, channels=32, bottleneck_channels=4):
        super().__init__()
        self.reduce = _reduction(channels, bottleneck_channels)
        self.forward_cell = _DepthwiseConvGRUCell(bottleneck_channels)
        self.backward_cell = _DepthwiseConvGRUCell(bottleneck_channels)
        self.fusion = _fusion(bottleneck_channels * 4, bottleneck_channels)
        self.gate = _reliability_gate(bottleneck_channels)
        self.projection = _zero_projection(bottleneck_channels, channels)

    def _propagate(self, appearance, reverse=False):
        indices = range(appearance.shape[2] - 1, -1, -1) if reverse else range(
            appearance.shape[2]
        )
        state = torch.zeros_like(appearance[:, :, 0])
        outputs = []
        cell = self.backward_cell if reverse else self.forward_cell
        for index in indices:
            state = cell(appearance[:, :, index], state)
            outputs.append(state)
        if reverse:
            outputs.reverse()
        return torch.stack(outputs, dim=2)

    def forward(self, x, return_aux=False):
        appearance = self.reduce(x)
        forward_features = self._propagate(appearance, reverse=False)
        backward_features = self._propagate(appearance, reverse=True)
        contrast = _local_contrast(appearance)
        fused = self.fusion(torch.cat([
            appearance, forward_features, backward_features, contrast
        ], dim=1))
        temporal_disagreement = (forward_features - backward_features).abs()
        reliability = self.gate(torch.cat([
            temporal_disagreement, contrast.abs()
        ], dim=1))
        delta = self.projection(fused)
        output = x + reliability * delta
        if not return_aux:
            return output
        return output, {
            'forward_features': forward_features,
            'backward_features': backward_features,
            'reliability_gate': reliability,
            'residual_delta': delta,
        }


class TDCDualStreamAdapter(nn.Module):
    """Parallel ordinary 3D context and explicit temporal-difference stream."""

    def __init__(self, channels=32, bottleneck_channels=8):
        super().__init__()
        self.reduce = _reduction(channels, bottleneck_channels)
        self.appearance_branches = nn.ModuleList()
        self.difference_branches = nn.ModuleList()
        for dilation in (1, 2, 4):
            branch = lambda: nn.Sequential(
                nn.Conv3d(
                    bottleneck_channels,
                    bottleneck_channels,
                    kernel_size=(3, 1, 1),
                    padding=(dilation, 0, 0),
                    dilation=(dilation, 1, 1),
                    groups=bottleneck_channels,
                    bias=False,
                ),
                nn.GroupNorm(
                    _group_count(bottleneck_channels), bottleneck_channels
                ),
                nn.SiLU(inplace=True),
            )
            self.appearance_branches.append(branch())
            self.difference_branches.append(branch())
        self.fusion = _fusion(bottleneck_channels * 4, bottleneck_channels)
        self.gate = _reliability_gate(bottleneck_channels)
        self.projection = _zero_projection(bottleneck_channels, channels)

    def forward(self, x, return_aux=False):
        appearance = self.reduce(x)
        previous, following = _temporal_neighbours(appearance, 1)
        temporal_difference = 0.5 * (following - previous)
        appearance_context = sum(
            branch(appearance) for branch in self.appearance_branches
        ) / float(len(self.appearance_branches))
        difference_context = sum(
            branch(temporal_difference)
            for branch in self.difference_branches
        ) / float(len(self.difference_branches))
        contrast = _local_contrast(appearance)
        fused = self.fusion(torch.cat([
            appearance, appearance_context, difference_context, contrast
        ], dim=1))
        reliability = self.gate(torch.cat([
            difference_context.abs(), contrast.abs()
        ], dim=1))
        delta = self.projection(fused)
        output = x + reliability * delta
        if not return_aux:
            return output
        return output, {
            'appearance_context': appearance_context,
            'difference_context': difference_context,
            'reliability_gate': reliability,
            'residual_delta': delta,
        }


def build_structure_adapter(
    variant,
    channels,
    bottleneck_channels=8,
    max_shift=4.0,
):
    if variant not in STRUCTURE_VARIANTS:
        raise ValueError(
            'Unknown structure variant %s; expected one of %s'
            % (variant, ', '.join(STRUCTURE_VARIANTS))
        )
    if bottleneck_channels <= 0:
        raise ValueError('bottleneck_channels must be positive')
    if variant == 'raw_apmd':
        return RawAppearanceMotionAdapter(channels, bottleneck_channels)
    if variant == 'raw_apmd_rms':
        return RawAppearanceMotionAdapter(
            channels, bottleneck_channels, normalization='rms'
        )
    if variant == 'raw_apmd_channel_rms':
        return RawAppearanceMotionAdapter(
            channels, bottleneck_channels, normalization='channel_rms'
        )
    if variant == 'raw_apmd_motion_detrend':
        return RawAppearanceMotionAdapter(
            channels,
            bottleneck_channels,
            normalization='rms',
            motion_detrend=True,
        )
    if variant == 'raw_apmd_multiscale_contrast':
        return RawAppearanceMotionAdapter(
            channels,
            bottleneck_channels,
            normalization='rms',
            adaptive_contrast=True,
        )
    if variant == 'raw_apmd_hybrid_rms':
        return RawAppearanceMotionAdapter(
            channels, bottleneck_channels, normalization='hybrid_rms'
        )
    if variant == 'raw_apmd_hybrid_rms_scratch_init':
        return RawAppearanceMotionAdapter(
            channels,
            bottleneck_channels,
            normalization='hybrid_rms',
            projection_init_scale=0.05,
        )
    if variant == 'raw_apmd_hybrid_rms_scratch_bandpass':
        return RawAppearanceMotionAdapter(
            channels,
            bottleneck_channels,
            normalization='hybrid_rms',
            projection_init_scale=0.05,
            temporal_bandpass=True,
        )
    if variant == 'raw_apmd_hybrid_rms_scratch_detail':
        return RawAppearanceMotionAdapter(
            channels,
            bottleneck_channels,
            normalization='hybrid_rms',
            projection_init_scale=0.05,
            backbone_detail=True,
        )
    if variant == 'raw_apmd_hybrid_rms_motion_detrend':
        return RawAppearanceMotionAdapter(
            channels,
            bottleneck_channels,
            normalization='hybrid_rms',
            motion_detrend=True,
        )
    if variant == 'raw_apmd_hybrid_rms_multiscale_contrast':
        return RawAppearanceMotionAdapter(
            channels,
            bottleneck_channels,
            normalization='hybrid_rms',
            adaptive_contrast=True,
        )
    if variant == 'raw_apmd_hybrid_rms_motion_detrend_multiscale_contrast':
        return RawAppearanceMotionAdapter(
            channels,
            bottleneck_channels,
            normalization='hybrid_rms',
            motion_detrend=True,
            adaptive_contrast=True,
        )
    if variant == 'second_order':
        return SecondOrderMotionAdapter(channels, bottleneck_channels)
    if variant in {'lfp_shallow', 'lfp_deep'}:
        return LowFrequencyPurificationAdapter(
            channels, bottleneck_channels
        )
    if variant == 'global_align':
        return GlobalMotionAlignmentAdapter(
            channels, bottleneck_channels, max_shift
        )
    if variant == 'local_align':
        return LocalMotionAlignmentAdapter(
            channels, bottleneck_channels, max_shift
        )
    if variant == 'multiscale_head':
        return MultiScaleContextAdapter(channels, bottleneck_channels)
    if variant == 'bidirectional':
        return BidirectionalPropagationAdapter(
            channels, min(4, bottleneck_channels)
        )
    if variant == 'tdc_dual_stream':
        return TDCDualStreamAdapter(channels, bottleneck_channels)
    raise AssertionError('unreachable structure variant')
