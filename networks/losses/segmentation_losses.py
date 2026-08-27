"""Selectable binary segmentation losses for multiframe small targets.

All losses accept logits and labels shaped ``[B, T, H, W]``. The optional
``images`` argument is only required by ``tda_sls`` and should have shape
``[B, 1, T, H, W]``. ``epoch`` controls warm-up terms in SLS/STC losses.
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


LOSS_NAMES = (
    'soft_iou',
    'frame_soft_iou',
    'bce',
    'focal',
    'dice',
    'bce_dice',
    'tversky',
    'focal_tversky',
    'lovasz',
    'sls_iou',
    'tda_sls',
    'hard_focal',
    'tversky_hard_focal',
    'stc_f1',
    'f1_calibrated_ohem',
)

LOSS_DESCRIPTIONS = {
    'soft_iou': '原始 DeepPro SoftIoU（默认，保持旧实验兼容）',
    'frame_soft_iou': '带平滑项的逐帧 SoftIoU',
    'bce': 'BCEWithLogits',
    'focal': '像素级 Focal BCE',
    'dice': '逐帧 Soft Dice',
    'bce_dice': 'BCE 与 Dice 加权组合',
    'tversky': '可控制漏检/虚警权重的 Tversky',
    'focal_tversky': 'Focal-Tversky',
    'lovasz': '直接优化 IoU 代理目标的 Lovasz hinge',
    'sls_iou': '尺度与位置敏感的 SLSIoU（视频逐帧适配）',
    'tda_sls': 'SLSIoU + 逐目标局部 TDA（需 CPU 连通域，较慢）',
    'hard_focal': '正样本 + 固定 Top-K 困难背景的 Focal',
    'tversky_hard_focal': 'Tversky + Hard-Focal，推荐稳健对照',
    'stc_f1': 'Tversky+Hard-Focal+中心响应+时序一致性实验组合',
    'f1_calibrated_ohem': 'Tversky+Dice+自适应困难负样本Margin，面向F1与低虚警',
}


def loss_experiment_name(name):
    """Return a filesystem-friendly loss label for experiment directories."""
    if name == 'soft_iou':
        return 'SoftLoUloss'
    return 'Loss-' + name.replace('_', '-')


def _prepare_binary_tensors(logits, target):
    if logits.ndim == 5 and logits.size(1) == 1:
        logits = logits.squeeze(1)
    if target.ndim == 5 and target.size(1) == 1:
        target = target.squeeze(1)
    if logits.ndim != 4 or target.ndim != 4:
        raise ValueError(
            'logits and target must have shape [B,T,H,W], got %s and %s'
            % (tuple(logits.shape), tuple(target.shape))
        )
    if logits.shape != target.shape:
        raise ValueError(
            'logits and target must have the same shape, got %s and %s'
            % (tuple(logits.shape), tuple(target.shape))
        )
    if not torch.is_floating_point(logits):
        raise TypeError('logits must be a floating-point tensor.')
    target = target.to(device=logits.device, dtype=logits.dtype).clamp(0.0, 1.0)
    return logits, target


def _zero_loss(tensor):
    return tensor.sum() * 0.0


def _frame_sums(probability, target):
    intersection = (probability * target).sum(dim=(-2, -1))
    pred_sum = probability.sum(dim=(-2, -1))
    target_sum = target.sum(dim=(-2, -1))
    return intersection, pred_sum, target_sum


def _dice_loss_per_frame(logits, target, eps):
    probability = torch.sigmoid(logits)
    intersection, pred_sum, target_sum = _frame_sums(probability, target)
    score = (2.0 * intersection + eps) / (pred_sum + target_sum + eps)
    return 1.0 - score


def _tversky_loss_per_frame(logits, target, fp_weight, fn_weight, eps):
    probability = torch.sigmoid(logits)
    true_positive = (probability * target).sum(dim=(-2, -1))
    false_positive = (probability * (1.0 - target)).sum(dim=(-2, -1))
    false_negative = ((1.0 - probability) * target).sum(dim=(-2, -1))
    score = (true_positive + eps) / (
        true_positive
        + fp_weight * false_positive
        + fn_weight * false_negative
        + eps
    )
    return 1.0 - score


class BinarySegmentationLoss(nn.Module):
    """Common interface for losses used by train.py."""

    requires_images = False

    def forward(self, logits, target, images=None, epoch=None):
        raise NotImplementedError


class LegacySoftIoULoss(BinarySegmentationLoss):
    """Original DeepPro SoftIoU, including its clip-level reduction."""

    def forward(self, logits, target, images=None, epoch=None):
        logits, target = _prepare_binary_tensors(logits, target)
        probability = torch.sigmoid(logits)
        intersection = probability * target
        intersection_sum = intersection.sum(dim=(1, 2, 3))
        pred_sum = probability.sum(dim=(1, 2, 3))
        target_sum = target.sum(dim=(1, 2, 3))
        union = pred_sum + target_sum - intersection_sum
        score = torch.where(
            union > 0,
            intersection_sum / union.clamp_min(torch.finfo(union.dtype).tiny),
            torch.ones_like(union),
        )
        return 1.0 - score.mean()


class FrameSoftIoULoss(BinarySegmentationLoss):
    def __init__(self, eps=1.0):
        super().__init__()
        self.eps = eps

    def forward(self, logits, target, images=None, epoch=None):
        logits, target = _prepare_binary_tensors(logits, target)
        probability = torch.sigmoid(logits)
        intersection, pred_sum, target_sum = _frame_sums(probability, target)
        union = pred_sum + target_sum - intersection
        return (1.0 - (intersection + self.eps) / (union + self.eps)).mean()


class BCELogitsLoss(BinarySegmentationLoss):
    def forward(self, logits, target, images=None, epoch=None):
        logits, target = _prepare_binary_tensors(logits, target)
        return F.binary_cross_entropy_with_logits(logits, target)


class FocalBCELoss(BinarySegmentationLoss):
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, target, images=None, epoch=None):
        logits, target = _prepare_binary_tensors(logits, target)
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
        probability = torch.sigmoid(logits)
        pt = probability * target + (1.0 - probability) * (1.0 - target)
        alpha_t = self.alpha * target + (1.0 - self.alpha) * (1.0 - target)
        return (alpha_t * (1.0 - pt).pow(self.gamma) * bce).mean()


class DiceLoss(BinarySegmentationLoss):
    def __init__(self, eps=1.0):
        super().__init__()
        self.eps = eps

    def forward(self, logits, target, images=None, epoch=None):
        logits, target = _prepare_binary_tensors(logits, target)
        return _dice_loss_per_frame(logits, target, self.eps).mean()


class BCEDiceLoss(BinarySegmentationLoss):
    def __init__(self, bce_weight=0.5, eps=1.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.eps = eps

    def forward(self, logits, target, images=None, epoch=None):
        logits, target = _prepare_binary_tensors(logits, target)
        bce = F.binary_cross_entropy_with_logits(logits, target)
        dice = _dice_loss_per_frame(logits, target, self.eps).mean()
        return self.bce_weight * bce + (1.0 - self.bce_weight) * dice


class TverskyLoss(BinarySegmentationLoss):
    def __init__(self, fp_weight=0.6, fn_weight=0.4, eps=1.0):
        super().__init__()
        self.fp_weight = fp_weight
        self.fn_weight = fn_weight
        self.eps = eps

    def forward(self, logits, target, images=None, epoch=None):
        logits, target = _prepare_binary_tensors(logits, target)
        return _tversky_loss_per_frame(
            logits, target, self.fp_weight, self.fn_weight, self.eps
        ).mean()


class FocalTverskyLoss(TverskyLoss):
    def __init__(self, fp_weight=0.6, fn_weight=0.4, gamma=1.33, eps=1.0):
        super().__init__(fp_weight=fp_weight, fn_weight=fn_weight, eps=eps)
        self.gamma = gamma

    def forward(self, logits, target, images=None, epoch=None):
        logits, target = _prepare_binary_tensors(logits, target)
        loss = _tversky_loss_per_frame(
            logits, target, self.fp_weight, self.fn_weight, self.eps
        )
        return loss.clamp_min(0.0).pow(self.gamma).mean()


def _lovasz_grad(labels_sorted):
    pixel_count = labels_sorted.numel()
    foreground_count = labels_sorted.sum()
    intersection = foreground_count - labels_sorted.cumsum(0)
    union = foreground_count + (1.0 - labels_sorted).cumsum(0)
    jaccard = 1.0 - intersection / union.clamp_min(1.0)
    if pixel_count > 1:
        jaccard = torch.cat((jaccard[:1], jaccard[1:] - jaccard[:-1]))
    return jaccard


def _lovasz_hinge_flat(logits, labels):
    if logits.numel() == 0:
        return _zero_loss(logits)
    signs = 2.0 * labels - 1.0
    errors = 1.0 - logits * signs
    errors_sorted, permutation = torch.sort(errors, descending=True)
    labels_sorted = labels[permutation]
    return torch.dot(F.relu(errors_sorted), _lovasz_grad(labels_sorted))


class LovaszHingeLoss(BinarySegmentationLoss):
    """Binary Lovasz hinge, evaluated once per video clip."""

    def forward(self, logits, target, images=None, epoch=None):
        logits, target = _prepare_binary_tensors(logits, target)
        losses = [
            _lovasz_hinge_flat(clip_logits.reshape(-1), clip_target.reshape(-1))
            for clip_logits, clip_target in zip(logits, target)
        ]
        return torch.stack(losses).mean() if losses else _zero_loss(logits)


class SLSIoULoss(BinarySegmentationLoss):
    """Scale/location-sensitive IoU adapted to each video frame."""

    def __init__(self, eps=1.0e-6, location_weight=1.0, warmup_epochs=5):
        super().__init__()
        self.eps = eps
        self.location_weight = location_weight
        self.warmup_epochs = warmup_epochs

    def _scale_iou(self, probability, target):
        intersection, pred_sum, target_sum = _frame_sums(probability, target)
        union = pred_sum + target_sum - intersection
        iou = (intersection + self.eps) / (union + self.eps)
        distance = ((pred_sum - target_sum) * 0.5).pow(2)
        scale = (
            torch.minimum(pred_sum, target_sum) + distance + self.eps
        ) / (torch.maximum(pred_sum, target_sum) + distance + self.eps)
        return 1.0 - (scale * iou).mean()

    def _location_loss(self, probability, target):
        height, width = probability.shape[-2:]
        x = torch.arange(
            width, dtype=probability.dtype, device=probability.device
        ).view(1, 1, 1, width)
        y = torch.arange(
            height, dtype=probability.dtype, device=probability.device
        ).view(1, 1, height, 1)
        pred_x = (x * probability).mean(dim=(-2, -1))
        pred_y = (y * probability).mean(dim=(-2, -1))
        target_x = (x * target).mean(dim=(-2, -1))
        target_y = (y * target).mean(dim=(-2, -1))

        pred_angle = torch.atan2(pred_y, pred_x + self.eps)
        target_angle = torch.atan2(target_y, target_x + self.eps)
        angle_loss = (4.0 / (math.pi ** 2)) * (pred_angle - target_angle).pow(2)
        pred_length = torch.sqrt(pred_x.pow(2) + pred_y.pow(2) + self.eps)
        target_length = torch.sqrt(target_x.pow(2) + target_y.pow(2) + self.eps)
        length_score = torch.minimum(pred_length, target_length) / torch.maximum(
            pred_length, target_length
        ).clamp_min(self.eps)
        location_score = (1.0 - angle_loss).clamp_min(0.0) * length_score
        valid = (target.sum(dim=(-2, -1)) > 0).to(probability.dtype)
        return ((1.0 - location_score) * valid).sum() / valid.sum().clamp_min(1.0)

    def forward(self, logits, target, images=None, epoch=None):
        logits, target = _prepare_binary_tensors(logits, target)
        probability = torch.sigmoid(logits)
        loss = self._scale_iou(probability, target)
        if (
            self.location_weight > 0.0
            and (epoch is None or epoch >= self.warmup_epochs)
        ):
            loss = loss + self.location_weight * self._location_loss(
                probability, target
            )
        return loss


class TDASLSLoss(BinarySegmentationLoss):
    """SLSIoU plus target-difficulty-aware local object loss.

    Component boxes are found on CPU. Cropped prediction tensors stay on the
    original device, so gradients still flow to the network.
    """

    requires_images = True

    def __init__(
        self,
        eps=1.0e-6,
        location_weight=1.0,
        warmup_epochs=5,
        tda_weight=0.2,
        mean_size=0.0,
        mean_contrast=0.0,
        dilation=3,
        resize=48,
    ):
        super().__init__()
        self.eps = eps
        self.tda_weight = tda_weight
        self.mean_size = mean_size
        self.mean_contrast = mean_contrast
        self.dilation = dilation
        self.resize = resize
        self.sls = SLSIoULoss(eps, location_weight, warmup_epochs)

    def _component_records(self, target):
        try:
            from scipy import ndimage
        except ImportError as exc:
            raise RuntimeError('tda_sls requires scipy for connected components.') from exc
        target_cpu = (target.detach().to('cpu') > 0.5).numpy().astype(np.uint8)
        batch, frames, height, width = target_cpu.shape
        structure = np.ones((3, 3), dtype=np.uint8)
        records = []
        for batch_index in range(batch):
            for frame_index in range(frames):
                labels, count = ndimage.label(
                    target_cpu[batch_index, frame_index], structure=structure
                )
                for component_index in range(1, count + 1):
                    component = labels == component_index
                    rows, cols = np.nonzero(component)
                    if rows.size == 0:
                        continue
                    y0 = max(0, int(rows.min()) - self.dilation)
                    y1 = min(height, int(rows.max()) + self.dilation + 1)
                    x0 = max(0, int(cols.min()) - self.dilation)
                    x1 = min(width, int(cols.max()) + self.dilation + 1)
                    records.append((
                        batch_index, frame_index, y0, y1, x0, x1,
                        component[y0:y1, x0:x1], float(rows.size),
                    ))
        return records

    def _tda_loss(self, logits, target, images):
        if images is None:
            raise ValueError('tda_sls requires images=[B,1,T,H,W].')
        if images.ndim == 4:
            images = images.unsqueeze(1)
        expected = (
            logits.size(0), 1, logits.size(1), logits.size(2), logits.size(3)
        )
        if tuple(images.shape) != expected:
            raise ValueError(
                'tda_sls expected images shape %s, got %s'
                % (expected, tuple(images.shape))
            )
        images = images.to(device=logits.device, dtype=logits.dtype)
        probability = torch.sigmoid(logits)
        records = self._component_records(target)
        if not records:
            return _zero_loss(logits)

        object_data, sizes, contrasts = [], [], []
        for b, t, y0, y1, x0, x1, component_np, size in records:
            component = torch.as_tensor(
                component_np, dtype=logits.dtype, device=logits.device
            )
            pred_crop = F.interpolate(
                probability[b, t, y0:y1, x0:x1][None, None],
                size=(self.resize, self.resize),
                mode='bilinear',
                align_corners=False,
            )[0, 0]
            component_resized = F.interpolate(
                component[None, None],
                size=(self.resize, self.resize),
                mode='nearest',
            )[0, 0]
            image_crop = images[b, 0, t, y0:y1, x0:x1]
            target_crop = target[b, t, y0:y1, x0:x1]
            object_mean = (image_crop * component).sum() / component.sum().clamp_min(1.0)
            background = (target_crop <= 0.5).to(logits.dtype)
            background_mean = (image_crop * background).sum() / background.sum().clamp_min(1.0)
            contrast = (object_mean - background_mean).clamp_min(0.0).detach()
            size_tensor = logits.new_tensor(size)
            object_data.append((pred_crop, component_resized, size_tensor, contrast))
            sizes.append(size_tensor)
            contrasts.append(contrast)

        mean_size = (
            logits.new_tensor(self.mean_size)
            if self.mean_size > 0.0
            else torch.stack(sizes).mean().clamp_min(self.eps)
        )
        mean_contrast = (
            logits.new_tensor(self.mean_contrast)
            if self.mean_contrast > 0.0
            else torch.stack(contrasts).mean().clamp_min(self.eps)
        )
        losses = []
        for pred_crop, component, size, contrast in object_data:
            intersection = (pred_crop * component).sum()
            union = pred_crop.sum() + component.sum() - intersection
            local_iou = ((intersection + self.eps) / (union + self.eps)).clamp(
                min=self.eps, max=1.0
            )
            difficulty = (
                1.0
                + torch.sigmoid(-size / mean_size)
                + torch.sigmoid(-contrast / mean_contrast)
            )
            losses.append(
                -(1.0 - local_iou.pow(difficulty)) * torch.log(local_iou)
            )
        return torch.stack(losses).mean()

    def forward(self, logits, target, images=None, epoch=None):
        logits, target = _prepare_binary_tensors(logits, target)
        return self.sls(logits, target, epoch=epoch) + self.tda_weight * self._tda_loss(
            logits, target, images
        )


class HardFocalLoss(BinarySegmentationLoss):
    """Focal loss using all positives and fixed Top-K background per clip."""

    def __init__(self, alpha=0.75, gamma=2.0, negative_topk=4096):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.negative_topk = negative_topk

    def forward(self, logits, target, images=None, epoch=None):
        logits, target = _prepare_binary_tensors(logits, target)
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
        probability = torch.sigmoid(logits)
        positive_focal = (1.0 - probability).pow(self.gamma) * bce * target
        negative_focal = probability.pow(self.gamma) * bce * (1.0 - target)
        losses = []
        for positive_clip, negative_clip, target_clip in zip(
            positive_focal, negative_focal, target
        ):
            positive_loss = positive_clip.sum() / target_clip.sum().clamp_min(1.0)
            negative_flat = negative_clip.reshape(-1)
            k = min(self.negative_topk, negative_flat.numel())
            negative_loss = (
                torch.topk(negative_flat, k=k, sorted=False).values.mean()
                if k > 0 else _zero_loss(negative_flat)
            )
            losses.append(
                self.alpha * positive_loss + (1.0 - self.alpha) * negative_loss
            )
        return torch.stack(losses).mean() if losses else _zero_loss(logits)


class TverskyHardFocalLoss(BinarySegmentationLoss):
    def __init__(
        self,
        fp_weight=0.6,
        fn_weight=0.4,
        eps=1.0,
        focal_alpha=0.75,
        focal_gamma=2.0,
        negative_topk=4096,
        hard_focal_weight=0.25,
    ):
        super().__init__()
        self.tversky = TverskyLoss(fp_weight, fn_weight, eps)
        self.hard_focal = HardFocalLoss(
            focal_alpha, focal_gamma, negative_topk
        )
        self.hard_focal_weight = hard_focal_weight

    def forward(self, logits, target, images=None, epoch=None):
        return self.tversky(logits, target) + self.hard_focal_weight * self.hard_focal(
            logits, target
        )


class F1CalibratedOHEMLoss(BinarySegmentationLoss):
    """F1-oriented overlap loss with balanced adaptive hard-negative mining."""

    component_names = ('tversky', 'dice', 'hard_margin', 'hard_weight')

    def __init__(
        self,
        fp_weight=0.6,
        fn_weight=0.4,
        eps=1.0,
        dice_weight=0.15,
        hard_weight=0.10,
        negative_ratio=4.0,
        min_negatives=256,
        max_negatives=4096,
        margin=1.0,
        warmup_epochs=5,
        ramp_epochs=10,
    ):
        super().__init__()
        if fp_weight < 0.0 or fn_weight < 0.0 or fp_weight + fn_weight <= 0.0:
            raise ValueError('F1 OHEM class weights must be non-negative and non-zero.')
        if dice_weight < 0.0 or hard_weight < 0.0:
            raise ValueError('F1 OHEM component weights must be non-negative.')
        if negative_ratio <= 0.0:
            raise ValueError('F1 OHEM negative_ratio must be positive.')
        if min_negatives <= 0 or max_negatives < min_negatives:
            raise ValueError('F1 OHEM negative limits are invalid.')
        if margin < 0.0 or warmup_epochs < 0 or ramp_epochs <= 0:
            raise ValueError('F1 OHEM margin/warm-up/ramp settings are invalid.')
        self.fp_weight = fp_weight
        self.fn_weight = fn_weight
        self.eps = eps
        self.dice_weight = dice_weight
        self.hard_weight = hard_weight
        self.negative_ratio = negative_ratio
        self.min_negatives = min_negatives
        self.max_negatives = max_negatives
        self.margin = margin
        self.warmup_epochs = warmup_epochs
        self.ramp_epochs = ramp_epochs
        self.last_components = {}

    def _hard_margin_loss(self, logits, target, valid_frames=None):
        losses = []
        class_weight_sum = self.fp_weight + self.fn_weight
        positive_weight = self.fn_weight / class_weight_sum
        negative_weight = self.fp_weight / class_weight_sum
        for clip_index, (clip_logits, clip_target) in enumerate(
            zip(logits, target)
        ):
            if valid_frames is not None:
                clip_mask = valid_frames[clip_index]
                clip_logits = clip_logits[clip_mask]
                clip_target = clip_target[clip_mask]
            flat_logits = clip_logits.reshape(-1)
            positive_mask = clip_target.reshape(-1) > 0.5
            positive_logits = flat_logits[positive_mask]
            negative_logits = flat_logits[~positive_mask]

            if negative_logits.numel() > 0:
                negative_losses = F.softplus(self.margin + negative_logits)
                positive_count = int(positive_logits.numel())
                adaptive_count = int(math.ceil(self.negative_ratio * positive_count))
                k = min(
                    negative_losses.numel(),
                    self.max_negatives,
                    max(self.min_negatives, adaptive_count),
                )
                negative_loss = torch.topk(
                    negative_losses, k=k, sorted=False
                ).values.mean()
            else:
                negative_loss = _zero_loss(flat_logits)

            if positive_logits.numel() > 0:
                positive_loss = F.softplus(
                    self.margin - positive_logits
                ).mean()
                losses.append(
                    positive_weight * positive_loss
                    + negative_weight * negative_loss
                )
            else:
                losses.append(negative_loss)
        return torch.stack(losses).mean() if losses else _zero_loss(logits)

    def _current_hard_weight(self, epoch):
        if epoch is None:
            return self.hard_weight
        if epoch < self.warmup_epochs:
            return 0.0
        progress = min(
            1.0,
            float(epoch - self.warmup_epochs + 1) / float(self.ramp_epochs),
        )
        return self.hard_weight * progress

    def forward(
        self, logits, target, images=None, epoch=None, valid_frames=None
    ):
        logits, target = _prepare_binary_tensors(logits, target)
        if valid_frames is not None:
            valid_frames = valid_frames.to(
                device=logits.device, dtype=torch.bool
            )
            if valid_frames.shape != logits.shape[:2]:
                raise ValueError(
                    'valid_frames must have shape [B,T], got %s for logits %s'
                    % (tuple(valid_frames.shape), tuple(logits.shape))
                )
            if not torch.all(valid_frames.any(dim=1)):
                raise ValueError(
                    'Every training clip must contain a valid frame.'
                )
        tversky_per_frame = _tversky_loss_per_frame(
            logits, target, self.fp_weight, self.fn_weight, self.eps
        )
        dice_per_frame = _dice_loss_per_frame(
            logits, target, self.eps
        )
        if valid_frames is None:
            tversky = tversky_per_frame.mean()
            dice = dice_per_frame.mean()
        else:
            tversky = tversky_per_frame[valid_frames].mean()
            dice = dice_per_frame[valid_frames].mean()
        hard_margin = self._hard_margin_loss(
            logits, target, valid_frames=valid_frames
        )
        current_hard_weight = self._current_hard_weight(epoch)
        total = (
            tversky
            + self.dice_weight * dice
            + current_hard_weight * hard_margin
        )
        self.last_components = {
            'tversky': tversky.detach(),
            'dice': dice.detach(),
            'hard_margin': hard_margin.detach(),
            'hard_weight': logits.new_tensor(current_hard_weight),
        }
        return total


def _gaussian_kernel(kernel_size, sigma):
    coordinates = torch.arange(kernel_size, dtype=torch.float32)
    coordinates = coordinates - (kernel_size - 1) * 0.5
    kernel_1d = torch.exp(-(coordinates.pow(2)) / (2.0 * sigma * sigma))
    kernel_1d = kernel_1d / kernel_1d.sum()
    return torch.outer(kernel_1d, kernel_1d)[None, None]


class STCF1Loss(TverskyHardFocalLoss):
    """Experimental center-aware and temporal-consistency F1 surrogate."""

    def __init__(
        self,
        fp_weight=0.6,
        fn_weight=0.4,
        eps=1.0,
        focal_alpha=0.75,
        focal_gamma=2.0,
        negative_topk=4096,
        hard_focal_weight=0.25,
        center_weight=0.1,
        temporal_weight=0.05,
        warmup_epochs=5,
        center_kernel_size=7,
        center_sigma=1.5,
        temporal_radius=8,
    ):
        super().__init__(
            fp_weight, fn_weight, eps, focal_alpha, focal_gamma,
            negative_topk, hard_focal_weight,
        )
        self.center_weight = center_weight
        self.temporal_weight = temporal_weight
        self.warmup_epochs = warmup_epochs
        self.temporal_radius = temporal_radius
        self.register_buffer(
            'center_kernel', _gaussian_kernel(center_kernel_size, center_sigma)
        )

    def _blur(self, tensor):
        batch, frames, height, width = tensor.shape
        kernel = self.center_kernel.to(dtype=tensor.dtype)
        blurred = F.conv2d(
            tensor.reshape(batch * frames, 1, height, width),
            kernel,
            padding=kernel.size(-1) // 2,
        )
        return blurred.reshape(batch, frames, height, width)

    def _center_loss(self, probability_blur, target_blur, target):
        similarity = F.cosine_similarity(
            probability_blur.flatten(start_dim=2),
            target_blur.flatten(start_dim=2),
            dim=-1,
            eps=1.0e-6,
        )
        valid = (target.sum(dim=(-2, -1)) > 0).to(probability_blur.dtype)
        return ((1.0 - similarity) * valid).sum() / valid.sum().clamp_min(1.0)

    def _temporal_loss(self, probability_blur, target_blur, target):
        if probability_blur.size(1) < 2:
            return _zero_loss(probability_blur)
        pixel_loss = F.smooth_l1_loss(
            probability_blur[:, 1:] - probability_blur[:, :-1],
            target_blur[:, 1:] - target_blur[:, :-1],
            reduction='none',
        )
        target_tube = torch.maximum(target[:, 1:], target[:, :-1])
        batch, frames, height, width = target_tube.shape
        kernel_size = 2 * self.temporal_radius + 1
        target_tube = F.max_pool2d(
            target_tube.reshape(batch * frames, 1, height, width),
            kernel_size=kernel_size,
            stride=1,
            padding=self.temporal_radius,
        ).reshape(batch, frames, height, width)
        return (pixel_loss * target_tube).sum() / target_tube.sum().clamp_min(1.0)

    def forward(self, logits, target, images=None, epoch=None):
        logits, target = _prepare_binary_tensors(logits, target)
        base_loss = super().forward(logits, target)
        if (
            (epoch is not None and epoch < self.warmup_epochs)
            or (self.center_weight <= 0.0 and self.temporal_weight <= 0.0)
        ):
            return base_loss
        probability_blur = self._blur(torch.sigmoid(logits))
        target_blur = self._blur(target)
        return (
            base_loss
            + self.center_weight * self._center_loss(
                probability_blur, target_blur, target
            )
            + self.temporal_weight * self._temporal_loss(
                probability_blur, target_blur, target
            )
        )


def _validate_parameters(
    name, eps, sls_eps, focal_alpha, focal_gamma, tversky_fp_weight,
    tversky_fn_weight, tversky_gamma, bce_weight, hard_negative_topk,
    hard_focal_weight, sls_location_weight, sls_warmup_epochs,
    tda_weight, tda_mean_size, tda_mean_contrast, tda_dilation,
    stc_center_weight, stc_temporal_weight, stc_warmup_epochs,
    f1_ohem_dice_weight, f1_ohem_hard_weight,
    f1_ohem_negative_ratio, f1_ohem_min_negatives,
    f1_ohem_margin, f1_ohem_warmup_epochs, f1_ohem_ramp_epochs,
):
    if name not in LOSS_NAMES:
        raise ValueError(
            'unknown loss %r; available: %s' % (name, ', '.join(LOSS_NAMES))
        )
    if eps <= 0.0 or sls_eps <= 0.0:
        raise ValueError('loss_eps and sls_eps must be positive.')
    if not 0.0 <= focal_alpha <= 1.0:
        raise ValueError('focal_alpha must be in [0, 1].')
    if focal_gamma < 0.0 or tversky_gamma <= 0.0:
        raise ValueError('focal_gamma must be >= 0 and tversky_gamma > 0.')
    if tversky_fp_weight < 0.0 or tversky_fn_weight < 0.0:
        raise ValueError('Tversky weights must be non-negative.')
    if tversky_fp_weight + tversky_fn_weight <= 0.0:
        raise ValueError('At least one Tversky weight must be positive.')
    if not 0.0 <= bce_weight <= 1.0:
        raise ValueError('bce_weight must be in [0, 1].')
    if hard_negative_topk <= 0:
        raise ValueError('hard_negative_topk must be positive.')
    if hard_focal_weight < 0.0 or sls_location_weight < 0.0 or tda_weight < 0.0:
        raise ValueError('Combined-loss weights must be non-negative.')
    if stc_center_weight < 0.0 or stc_temporal_weight < 0.0:
        raise ValueError('STC weights must be non-negative.')
    if name == 'f1_calibrated_ohem':
        if f1_ohem_dice_weight < 0.0 or f1_ohem_hard_weight < 0.0:
            raise ValueError('F1 OHEM component weights must be non-negative.')
        if f1_ohem_negative_ratio <= 0.0:
            raise ValueError('F1 OHEM negative ratio must be positive.')
        if f1_ohem_min_negatives <= 0:
            raise ValueError('F1 OHEM minimum negatives must be positive.')
        if f1_ohem_min_negatives > hard_negative_topk:
            raise ValueError(
                'F1 OHEM minimum negatives cannot exceed hard_negative_topk.'
            )
        if f1_ohem_margin < 0.0:
            raise ValueError('F1 OHEM margin must be non-negative.')
        if f1_ohem_warmup_epochs < 0 or f1_ohem_ramp_epochs <= 0:
            raise ValueError('F1 OHEM warm-up/ramp settings are invalid.')
    if sls_warmup_epochs < 0 or stc_warmup_epochs < 0:
        raise ValueError('Warm-up epochs must be non-negative.')
    if tda_mean_size < 0.0 or tda_mean_contrast < 0.0:
        raise ValueError('TDA dataset means must be non-negative; use 0 for batch means.')
    if tda_dilation < 0:
        raise ValueError('tda_dilation must be non-negative.')


def build_segmentation_loss(
    name,
    eps=1.0,
    sls_eps=1.0e-6,
    focal_alpha=0.75,
    focal_gamma=2.0,
    tversky_fp_weight=0.6,
    tversky_fn_weight=0.4,
    tversky_gamma=1.33,
    bce_weight=0.5,
    hard_negative_topk=4096,
    hard_focal_weight=0.25,
    sls_location_weight=1.0,
    sls_warmup_epochs=5,
    tda_weight=0.2,
    tda_mean_size=0.0,
    tda_mean_contrast=0.0,
    tda_dilation=3,
    stc_center_weight=0.1,
    stc_temporal_weight=0.05,
    stc_warmup_epochs=5,
    f1_ohem_dice_weight=0.15,
    f1_ohem_hard_weight=0.10,
    f1_ohem_negative_ratio=4.0,
    f1_ohem_min_negatives=256,
    f1_ohem_margin=1.0,
    f1_ohem_warmup_epochs=5,
    f1_ohem_ramp_epochs=10,
):
    """Build one of the command-line selectable losses."""
    _validate_parameters(
        name, eps, sls_eps, focal_alpha, focal_gamma, tversky_fp_weight,
        tversky_fn_weight, tversky_gamma, bce_weight, hard_negative_topk,
        hard_focal_weight, sls_location_weight, sls_warmup_epochs,
        tda_weight, tda_mean_size, tda_mean_contrast, tda_dilation,
        stc_center_weight, stc_temporal_weight, stc_warmup_epochs,
        f1_ohem_dice_weight, f1_ohem_hard_weight,
        f1_ohem_negative_ratio, f1_ohem_min_negatives,
        f1_ohem_margin, f1_ohem_warmup_epochs, f1_ohem_ramp_epochs,
    )
    if name == 'soft_iou':
        return LegacySoftIoULoss()
    if name == 'frame_soft_iou':
        return FrameSoftIoULoss(eps)
    if name == 'bce':
        return BCELogitsLoss()
    if name == 'focal':
        return FocalBCELoss(focal_alpha, focal_gamma)
    if name == 'dice':
        return DiceLoss(eps)
    if name == 'bce_dice':
        return BCEDiceLoss(bce_weight, eps)
    if name == 'tversky':
        return TverskyLoss(tversky_fp_weight, tversky_fn_weight, eps)
    if name == 'focal_tversky':
        return FocalTverskyLoss(
            tversky_fp_weight, tversky_fn_weight, tversky_gamma, eps
        )
    if name == 'lovasz':
        return LovaszHingeLoss()
    if name == 'sls_iou':
        return SLSIoULoss(sls_eps, sls_location_weight, sls_warmup_epochs)
    if name == 'tda_sls':
        return TDASLSLoss(
            sls_eps, sls_location_weight, sls_warmup_epochs, tda_weight,
            tda_mean_size, tda_mean_contrast, tda_dilation,
        )
    if name == 'hard_focal':
        return HardFocalLoss(focal_alpha, focal_gamma, hard_negative_topk)
    if name == 'tversky_hard_focal':
        return TverskyHardFocalLoss(
            tversky_fp_weight, tversky_fn_weight, eps, focal_alpha,
            focal_gamma, hard_negative_topk, hard_focal_weight,
        )
    if name == 'stc_f1':
        return STCF1Loss(
            tversky_fp_weight, tversky_fn_weight, eps, focal_alpha,
            focal_gamma, hard_negative_topk, hard_focal_weight,
            stc_center_weight, stc_temporal_weight, stc_warmup_epochs,
        )
    if name == 'f1_calibrated_ohem':
        return F1CalibratedOHEMLoss(
            tversky_fp_weight, tversky_fn_weight, eps,
            dice_weight=f1_ohem_dice_weight,
            hard_weight=f1_ohem_hard_weight,
            negative_ratio=f1_ohem_negative_ratio,
            min_negatives=f1_ohem_min_negatives,
            max_negatives=hard_negative_topk,
            margin=f1_ohem_margin,
            warmup_epochs=f1_ohem_warmup_epochs,
            ramp_epochs=f1_ohem_ramp_epochs,
        )
    raise AssertionError('unreachable loss selection: %s' % name)
