import os
import torch
import torch.nn as nn
import numpy as np
import math
from skimage import measure


class ShootingRules(nn.Module):
    def __init__(self):
        super(ShootingRules, self).__init__()

        return
    @staticmethod
    def _prepare_target(target_one):
        """Extract target geometry once and reuse it for every threshold."""
        labelimage = measure.label(target_one, connectivity=2)
        props = measure.regionprops(
            labelimage,
            intensity_image=target_one,
            cache=True,
        )
        loc_len2 = 4
        box2_map = np.ones(target_one.shape)
        target_coordinates = []
        for prop in props:
            pixel_coords = prop.coords
            target_coordinates.append(pixel_coords)
            for i_pixel in pixel_coords:
                box2_map[
                    i_pixel[0]-loc_len2:i_pixel[0]+loc_len2+1,
                    i_pixel[1]-loc_len2:i_pixel[1]+loc_len2+1,
                ] = 0
        return target_coordinates, box2_map

    @staticmethod
    def _threshold_counts(
        output_one,
        target_coordinates,
        box2_map,
        detect_threshold,
    ):
        """Apply the original shooting rules for one prepared target map."""
        thresholded = output_one.copy()
        thresholded[np.where(thresholded < detect_threshold)] = 0
        thresholded[np.where(thresholded >= detect_threshold)] = 1

        loc_len1 = 1
        true_num = 0
        for pixel_coords in target_coordinates:
            true_flag = 0
            for i_pixel in pixel_coords:
                target_area = thresholded[
                    i_pixel[0]-loc_len1:i_pixel[0]+loc_len1+1,
                    i_pixel[1]-loc_len1:i_pixel[1]+loc_len1+1,
                ]
                if target_area.sum() >= 1:
                    true_flag = 1
            if true_flag == 1:
                true_num += 1

        false_num = np.count_nonzero(thresholded * box2_map)
        return false_num, true_num

    def evaluate_thresholds(self, output, target, detect_thresholds):
        """Evaluate many thresholds without relabeling the same target image."""
        thresholds = np.asarray(detect_thresholds)
        false_numbers = np.zeros(thresholds.shape, dtype=np.int64)
        true_numbers = np.zeros(thresholds.shape, dtype=np.int64)
        target_numbers = np.zeros(thresholds.shape, dtype=np.int64)

        for i_batch in range(output.shape[0]):
            output_one = output[i_batch, :, :]
            target_one = target[i_batch, :, :]
            target_coordinates, box2_map = self._prepare_target(target_one)
            target_numbers += len(target_coordinates)
            if np.isfinite(output_one).all():
                # For finite sigmoid probabilities, the original rule detects
                # a target iff the maximum value in any of its 3x3 pixel
                # neighborhoods reaches the threshold.
                target_peaks = np.full(
                    len(target_coordinates),
                    -np.inf,
                    dtype=output_one.dtype,
                )
                for target_index, pixel_coords in enumerate(target_coordinates):
                    for i_pixel in pixel_coords:
                        target_area = output_one[
                            i_pixel[0]-1:i_pixel[0]+2,
                            i_pixel[1]-1:i_pixel[1]+2,
                        ]
                        if target_area.size > 0:
                            target_peaks[target_index] = max(
                                target_peaks[target_index],
                                target_area.max(),
                            )
                if target_peaks.size > 0:
                    true_numbers += np.count_nonzero(
                        target_peaks[:, None] >= thresholds[None, :],
                        axis=0,
                    )

                # Sorting once is equivalent to thresholding and counting the
                # false-alarm region 27 times, but avoids 27 full image copies.
                false_values = np.sort(output_one[box2_map != 0].reshape(-1))
                false_numbers += false_values.size - np.searchsorted(
                    false_values,
                    thresholds,
                    side='left',
                )
            else:
                # Preserve the legacy behavior for invalid predictions, where
                # NaN comparison and count_nonzero semantics are unusual.
                for threshold_index, threshold in enumerate(thresholds):
                    false_num, true_num = self._threshold_counts(
                        output_one,
                        target_coordinates,
                        box2_map,
                        threshold,
                    )
                    false_numbers[threshold_index] += false_num
                    true_numbers[threshold_index] += true_num
        return false_numbers, true_numbers, target_numbers

    def forward(self, output, target, DetectTh):
        false_numbers, true_numbers, target_numbers = self.evaluate_thresholds(
            output,
            target,
            [DetectTh],
        )
        return (
            int(false_numbers[0]),
            int(true_numbers[0]),
            int(target_numbers[0]),
        )


