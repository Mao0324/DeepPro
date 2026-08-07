# DeepPro-Plus BRTD2 research note

## Evidence used for the redesign

The current BRTD run lowers validation recall and increases false alarms while
its three temporal branches still share the same five-frame receptive field.
The redesign therefore prioritizes target preservation and stable temporal
fusion instead of adding another aggressive difference operator.

| Work | Relevant idea | Decision for BRTD2 |
|---|---|---|
| [DeepPro](https://arxiv.org/abs/2506.12766) | Global temporal-profile saliency is highly discriminative for dim targets. | Keep the pretrained TPro backbone unchanged. |
| [RFR](https://arxiv.org/abs/2409.12448), [official code](https://github.com/XinyiYing/RFR) | Satellite-video features benefit from alignment, recurrent propagation, temporal correlation and spatial/frequency modulation. | Put conservative temporal refinement on semantic features. Do not copy its custom DCN dependency into this lightweight branch. |
| [BIRD](https://arxiv.org/abs/2508.15415) | Local and global temporal evidence should be propagated bidirectionally and jointly optimized. | Use symmetric temporal filtering with multiple true time spans; keep bidirectional recurrence as a later ablation because TPro already models the full clip. |
| [TDCNet](https://arxiv.org/abs/2511.09352) | Explicit motion differences and ordinary spatio-temporal features are complementary. | Preserve a normal appearance path and use temporal evidence only for a gated residual correction. |
| [MoCoPnet](https://arxiv.org/abs/2201.01014), [official code](https://github.com/XinyiYing/MoCoPnet) | Local motion alignment and local contrast priors strengthen weak targets. | Use local contrast as gate evidence, not as a replacement for target-bearing features. |
| [Group Normalization](https://arxiv.org/abs/1803.08494) | GroupNorm avoids batch-dependent running statistics. | Remove BatchNorm from the new adapter and router. |

## Implemented BRTD2 changes

1. The adapter is moved from the shallow 8-channel stem to the semantic
   32-channel feature after `layer1`.
2. A bottleneck keeps the module small: 1,284 parameters versus 889 in BRTD1.
3. Temporal branches use kernel size 3 with dilations 1, 2 and 4, producing
   receptive fields of 3, 5 and 9 frames instead of three variants of the same
   five-frame field.
4. An appearance feature is always fused with temporal context so stationary
   or slowly moving targets are not forced through a high-pass path.
5. Local contrast affects reliability gating but never replaces the appearance
   feature.
6. The router starts with uniform weights and the gate starts at
   `sigmoid(-2) = 0.1192`.
7. The output projection is zero initialized. Loading the original
   DeepPro-Plus checkpoint therefore gives exactly identical logits before the
   first optimization step.

## First controlled experiment

Use the configuration of the 82.13 submission and change only the model and
learning rates:

```bash
python -u train.py \
  --gpu 0 --gpu_num 1 \
  --model DeepPro-Plus_BRTD2 \
  --dataset SatVideoIRSDT_v1 \
  --datapath /home/devbox/project/model/sjy/CSIG2026/datasets/SatVideoIRSDT_v1 \
  --savepath /home/devbox/project/model/sjy/CSIG2026/Deeppro_v2/DeepPro-main/log \
  --optimizer Adam --learning_rate 0.001 --base_lr_mult 0.1 \
  --decay_rate 0.0001 --batch_size 20 --epoch 50 \
  --seqlen 40 --patch_size 128 --sample_rate 0.04 \
  --step_size 10 --lr_decay 0.7 --threshold_eval 0.5 \
  --train_workers 8 --val_workers 4 --prefetch_factor 2 \
  --loss tversky_hard_focal \
  --tversky_fp_weight 0.6 --tversky_fn_weight 0.4 \
  --base_ckpt /home/devbox/project/model/sjy/CSIG2026/Deeppro_v2/DeepPro-main/pretrained/SatVideoIRSDT_DeepPro-Plus_pretrained_init.pth \
  --brtd_use_background 1 --brtd_adaptive_tdc 1 \
  --brtd_use_gate 1 --brtd_zero_init 1 \
  --eval_chunk_rows 64 --resume never --seed 46 \
  --deterministic 0 --run_test_after_train 0 --use_swanlab 0
```

The first comparison must use the same split, seed, sampling rate, loss and
threshold as the old best model. Threshold sweeps should only be performed
after BRTD2 improves validation mIoU, object Pd and Fa.

## Ablation order

1. BRTD2 full model.
2. Disable local-contrast evidence with `--brtd_use_background 0`.
3. Disable adaptive routing with `--brtd_adaptive_tdc 0`.
4. Disable the reliability gate with `--brtd_use_gate 0`.
5. If recall still falls, move the adapter back before `layer1` only as a
   controlled placement ablation; do not combine this change with a new loss.

Track validation object Pd/Fa and the standard deviation of mIoU over the last
ten epochs in addition to the best pixel mIoU. The previous BRTD failure was
visible in all three measurements.
