# DeepPro/BRTD3 structure round 2

This round is intentionally prepared but not launched while the 2026-08-11
valid-frame/BRTD ablations are running.

## Shared controls

- Dataset: `SatVideoIRSDT_v1`
- Environment: `sjyPID`
- Initialization: `pretrained/SatVideoIRSDT_DeepPro-Plus_pretrained_init.pth`
- Loss: `f1_calibrated_ohem`, padded frames masked
- Epochs: 100 maximum
- Early stopping: validation F1, patience 30, minimum delta 1e-4, counting from epoch 15
- Resume: checkpointed early-stopping best value and patience counter
- Seed: 46
- Adapter LR: 0.001; pretrained backbone LR: 0.0001
- SwanLab: cloud mode
- Outputs: date-grouped below `log/sem_seg`
- Finalization: top-three saved validation-F1 checkpoints, centroid threshold
  and minimum-area sweep, trajectory tracking, ZIP validation and SHA-256

## One-variable candidates

| GPU | Variant | Purpose |
| --- | --- | --- |
| 0 | `second_order` | explicit first/second-order temporal anomaly |
| 1 | `tdc_dual_stream` | ordinary 3D and temporal-difference streams |
| 2 | `lfp_shallow` | low-frequency-guided purification after the stem |
| 3 | `lfp_deep` | low-frequency-guided purification before TPro |
| 4 | `global_align` | background-dominant global translation alignment |
| 5 | `local_align` | dense local feature alignment |
| 6 | `multiscale_head` | full-resolution dilated spatial context head |
| 7 | `bidirectional` | forward/backward recurrent feature propagation |

All branches are zero-initialized residual adapters under `brtd.*`. The old
DeepPro-Plus state dict therefore loads without unexpected keys, and initial
logits exactly equal the pretrained baseline.

## Commands

Preview only (safe while GPUs are busy):

```bash
bash tools/launch_structure_round2_8gpu.sh --dry-run
```

Launch after all eight GPUs are idle:

```bash
bash tools/launch_structure_round2_8gpu.sh
```

The launcher refuses to start if any GPU uses more than 1024 MiB.
