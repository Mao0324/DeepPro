"""Loss functions used by DeepPro training scripts."""

from .segmentation_losses import (
    LOSS_DESCRIPTIONS,
    LOSS_NAMES,
    build_segmentation_loss,
    loss_experiment_name,
)

__all__ = [
    'LOSS_DESCRIPTIONS',
    'LOSS_NAMES',
    'build_segmentation_loss',
    'loss_experiment_name',
]
