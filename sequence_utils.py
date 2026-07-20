import torch


def _frame_index(value):
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError('Frame index must contain exactly one value.')
        return int(value.item())
    return int(value)


def frame_range_length(first_end):
    first_frame, last_frame = (_frame_index(value) for value in first_end)
    return last_frame - first_frame + 1


class SequenceAccumulator:
    """Merge temporally overlapping inference windows for one sequence."""

    def __init__(self):
        self.predictions = None
        self.targets = None
        self.centroids = None
        self.first_frame = None
        self.last_frame = None

    def add(self, predictions, targets, first_end, centroids=None):
        if predictions.ndim != 4 or targets.ndim != 4:
            raise ValueError('Predictions and targets must have shape [B, T, H, W].')
        if predictions.shape != targets.shape:
            raise ValueError(
                'Prediction and target shapes differ: %s versus %s.'
                % (tuple(predictions.shape), tuple(targets.shape))
            )
        if centroids is not None and centroids.shape != targets.shape:
            raise ValueError(
                'Centroid and target shapes differ: %s versus %s.'
                % (tuple(centroids.shape), tuple(targets.shape))
            )

        first_frame, last_frame = (_frame_index(value) for value in first_end)
        window_length = frame_range_length(first_end)
        if predictions.shape[1] < window_length:
            raise ValueError(
                'Frame range [%d, %d] has length %d, but window contains only %d frames.'
                % (first_frame, last_frame, window_length, predictions.shape[1])
            )
        predictions = predictions[:, :window_length]
        targets = targets[:, :window_length]
        if centroids is not None:
            centroids = centroids[:, :window_length]

        if self.predictions is None:
            self.predictions = predictions
            self.targets = targets
            self.centroids = centroids
            self.first_frame = first_frame
            self.last_frame = last_frame
            return

        if first_frame < self.first_frame or last_frame < self.last_frame:
            raise ValueError('Sequence windows must be supplied in chronological order.')
        if first_frame > self.last_frame + 1:
            raise ValueError(
                'Gap between sequence windows: previous end %d, next start %d.'
                % (self.last_frame, first_frame)
            )
        if (self.centroids is None) != (centroids is None):
            raise ValueError('Centroids must either be supplied for every window or omitted.')

        overlap_length = max(0, self.last_frame - first_frame + 1)
        if overlap_length:
            accumulated_start = first_frame - self.first_frame
            accumulated_end = accumulated_start + overlap_length
            self.predictions[:, accumulated_start:accumulated_end] = torch.maximum(
                self.predictions[:, accumulated_start:accumulated_end],
                predictions[:, :overlap_length],
            )

        if overlap_length < predictions.shape[1]:
            self.predictions = torch.cat(
                (self.predictions, predictions[:, overlap_length:]), dim=1
            )
            self.targets = torch.cat(
                (self.targets, targets[:, overlap_length:]), dim=1
            )
            if centroids is not None:
                self.centroids = torch.cat(
                    (self.centroids, centroids[:, overlap_length:]), dim=1
                )

        self.last_frame = max(self.last_frame, last_frame)
