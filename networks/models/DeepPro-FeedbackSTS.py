"""Scratch-only bidirectional spatio-temporal feedback detector."""

import torch
import torch.nn as nn

try:
    from feedback_sts import FeedbackSTSBackbone
except ImportError:
    from networks.layers.feedback_sts import FeedbackSTSBackbone


class detector(nn.Module):
    def __init__(
        self,
        num_classes,
        seqlen=40,
        out_len=40,
        feedback_interval=2,
        feedback_alignment_levels=2,
        feedback_eval_tile_size=384,
        feedback_eval_tile_overlap=64,
    ):
        super().__init__()
        if seqlen != out_len:
            raise ValueError('FeedbackSTS requires seqlen == out_len')
        if feedback_eval_tile_size <= 0:
            raise ValueError('feedback_eval_tile_size must be positive')
        if not 0 <= feedback_eval_tile_overlap < feedback_eval_tile_size:
            raise ValueError('invalid feedback evaluation tile overlap')
        if feedback_eval_tile_size % 16:
            raise ValueError('feedback evaluation tile size must divide by 16')
        self.out_len = int(out_len)
        self.feedback_eval_tile_size = int(feedback_eval_tile_size)
        self.feedback_eval_tile_overlap = int(feedback_eval_tile_overlap)
        self.backbone = FeedbackSTSBackbone(
            in_channels=1,
            out_channels=num_classes,
            channels=(8, 16, 32, 64, 128),
            interval=feedback_interval,
            alignment_levels=feedback_alignment_levels,
        )

    @staticmethod
    def _tile_starts(length, tile_size, stride):
        if length <= tile_size:
            return (0,)
        starts = list(range(0, length - tile_size + 1, stride))
        final_start = length - tile_size
        if starts[-1] != final_start:
            starts.append(final_start)
        return tuple(starts)

    def _forward_tiled(self, sequence):
        batch, _, frames, height, width = sequence.shape
        tile = self.feedback_eval_tile_size
        stride = tile - self.feedback_eval_tile_overlap
        row_starts = self._tile_starts(height, tile, stride)
        column_starts = self._tile_starts(width, tile, stride)
        logits_sum = sequence.new_zeros(
            batch, frames, height, width, dtype=torch.float32
        )
        weights = sequence.new_zeros(
            1, 1, height, width, dtype=torch.float32
        )
        for row in row_starts:
            for column in column_starts:
                patch = sequence[
                    :, :, :, row:row + tile, column:column + tile
                ]
                _, patch_logits = self._forward_backbone_padded(patch)
                patch_logits = patch_logits.squeeze(1).float()
                logits_sum[
                    :, :, row:row + tile, column:column + tile
                ].add_(patch_logits)
                weights[
                    :, :, row:row + tile, column:column + tile
                ].add_(1.0)
                del patch, patch_logits
        logits = logits_sum / weights.clamp_min_(1.0)
        # The caller only uses the first item as an optional feature tensor.
        # A view avoids retaining another full-resolution allocation.
        return logits.unsqueeze(1), logits

    def _forward_backbone_padded(self, sequence):
        """Pad arbitrary validation sizes to the U-Net spatial divisor."""
        height, width = sequence.shape[-2:]
        divisor = self.backbone.spatial_divisor
        pad_height = (-height) % divisor
        pad_width = (-width) % divisor
        if pad_height or pad_width:
            sequence = torch.nn.functional.pad(
                sequence, (0, pad_width, 0, pad_height)
            )
        features, logits = self.backbone(sequence)
        return (
            features[..., :height, :width],
            logits[..., :height, :width],
        )

    def forward(self, sequence):
        if sequence.ndim != 5 or sequence.shape[1] != 1:
            raise ValueError('expected input shape [B,1,T,H,W]')
        if sequence.shape[2] != self.out_len:
            raise ValueError(
                'expected %d frames, got %d'
                % (self.out_len, sequence.shape[2])
            )
        height, width = sequence.shape[-2:]
        if (
            not self.training
            and (height > self.feedback_eval_tile_size
                 or width > self.feedback_eval_tile_size)
        ):
            return self._forward_tiled(sequence)
        features, logits = self._forward_backbone_padded(sequence)
        return features, logits.squeeze(1)
