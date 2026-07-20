import torch
import torch.nn as nn
import numpy as np
from skimage import measure


class ShootingRules(nn.Module):
    """Shooting rules with boundary-safe target neighborhoods."""

    def __init__(self):
        super(ShootingRules, self).__init__()

    def forward(self, output, target, DetectTh):
        FalseNum = 0
        TrueNum = 0
        TgtNum = 0

        for i_batch in range(output.shape[0]):
            output_one = output[i_batch, :, :].copy()
            target_one = target[i_batch, :, :].copy()

            output_one[np.where(output_one < DetectTh)] = 0
            output_one[np.where(output_one >= DetectTh)] = 1

            labelimage = measure.label(target_one, connectivity=2)
            props = measure.regionprops(
                labelimage,
                intensity_image=target_one,
                cache=True,
            )

            TgtNum += len(props)
            LocLen1 = 1
            LocLen2 = 4

            Box2_map = np.ones(output_one.shape)
            height, width = output_one.shape
            for prop in props:
                True_flag = 0

                for i_pixel in prop.coords:
                    box2_top = max(0, i_pixel[0] - LocLen2)
                    box2_bottom = min(height, i_pixel[0] + LocLen2 + 1)
                    box2_left = max(0, i_pixel[1] - LocLen2)
                    box2_right = min(width, i_pixel[1] + LocLen2 + 1)
                    Box2_map[
                        box2_top:box2_bottom,
                        box2_left:box2_right,
                    ] = 0

                    target_top = max(0, i_pixel[0] - LocLen1)
                    target_bottom = min(height, i_pixel[0] + LocLen1 + 1)
                    target_left = max(0, i_pixel[1] - LocLen1)
                    target_right = min(width, i_pixel[1] + LocLen1 + 1)
                    Tgt_area = output_one[
                        target_top:target_bottom,
                        target_left:target_right,
                    ]
                    if Tgt_area.sum() >= 1:
                        True_flag = 1
                if True_flag == 1:
                    TrueNum += 1

            False_output_one = output_one * Box2_map
            FalseNum += np.count_nonzero(False_output_one)

        return FalseNum, TrueNum, TgtNum
