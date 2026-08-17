# Raw-APMD: an F1-first DeepPro structural candidate

## Why this candidate

The strongest completed local validation result currently has centroid
precision `0.926414`, recall `0.648766`, and F1 `0.763120`; its website score
is `85.72`.  Precision is already high while recall is the limiting term.
The original DeepPro-Plus stem starts with spatial difference convolution and
then uses spatio-temporal difference residual blocks.  That is effective for
clutter rejection, but it has no independent route through which the absolute
appearance of a dim target can reach TPro.

Raw-APMD therefore targets missed weak targets rather than adding another
aggressive suppressor.  It combines three ideas:

1. a framewise raw-appearance encoder that never passes through the difference
   stem;
2. channel-adaptive first- and second-order temporal dynamics at offsets
   `1, 2, 4`;
3. a zero-initialized, ungated signed residual added to the pretrained
   32-channel feature immediately before TPro.

The design is consistent with the DeepPro temporal-profile formulation,
TDCNet's complementary appearance/temporal-difference streams, and IRDINO's
second-order motion modeling:

- DeepPro: https://arxiv.org/abs/2506.12766
- TDCNet: https://arxiv.org/abs/2511.09352
- IRDINO: https://openaccess.thecvf.com/content/CVPR2026F/html/Xu_IRDINO_Adapting_DINOv3_with_Second-Order_Motion_Awareness_for_Moving_Infrared_CVPRF_2026_paper.html

## Data flow

```text
normalized raw sequence
  |-- pretrained SDifference + STD blocks ------------------------|
  |                                                               +--> TPro --> head
  `-- framewise 5x5/depthwise 3x3 appearance encoder              |
        |-- first-order temporal context, offsets 1/2/4 --|        |
        |-- second-order temporal context, offsets 1/2/4 -+--> fusion --> zero-init residual
        `-- local spatial contrast -----------------------|
```

The scale mixtures use a softmax independently for every bottleneck channel.
There is deliberately no reliability gate: the previous BRTD2 ablation showed
that removing the gate was better, and an input-dependent gate can suppress
exactly the weak responses whose recall needs to improve.

## Compatibility and safety

- Model: `DeepPro-Plus_BRTD3`
- Variant: `raw_apmd`
- New parameter prefix: `brtd.*`
- New parameters: `2,496` with bottleneck width 8
- The final projection is exactly zero initialized, so loading the original
  DeepPro-Plus checkpoint gives exactly identical initial logits.
- Padding frames are detected from the loader's exact zero padding.  Temporal
  neighbours crossing a padded boundary fall back to the current valid frame,
  and all new residuals on padded frames are exactly zero.
- Normalization is framewise, so placing the same clip before or after padding
  produces the same valid-frame correction.
- Training snapshots both the model and `structure_adapters.py`; `test.py` can
  therefore reconstruct the exact model without depending on later source
  edits.

Run the focused sanity check:

```bash
/home/devbox/project/model/miniconda3/envs/sjyPID/bin/python \
  tools/check_brtd3.py --variant raw_apmd --device cpu \
  --seqlen 40 --height 16 --width 16
```

## Training and submission

Do not start this while the current eight-GPU batch is running.  When one GPU
is idle, the following launcher uses Screen, `sjyPID`, 100 epochs, SwanLab,
date-grouped logs, validation checkpoint/threshold/min-area selection, tracked
TXT generation, ZIP validation, and SHA-256 output:

```bash
bash tools/launch_raw_apmd_experiment.sh GPU_ID
```

The first controlled run uses adapter LR `0.001`, pretrained backbone LR
`0.005` (`base_lr_mult=5.0`), seed 46, `f1_calibrated_ohem`, and valid-frame
masking.  This matches the successful plain DeepPro backbone learning-rate
scale while giving the new residual branch a conservative rate.  Its result
must be judged by centroid F1, precision, and recall under the same sweep—not
by the final epoch or pixel IoU alone.
