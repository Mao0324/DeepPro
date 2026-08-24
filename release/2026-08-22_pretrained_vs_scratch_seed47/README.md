# 2026-08-22 Hybrid-RMS 预训练/随机初始化对照实验发布集

本目录是从完整实验目录中抽取的、适合纳入 GitHub 的可审计发布集。它包含 8 组实验的最佳后处理 checkpoint、训练与 SwanLab 侧车日志、阈值扫描结果、提交包和验证记录，但不包含体积很大的逐帧概率缓存与所有中间 epoch 权重。

完整项目迁移、模型与损失演化、历史实验结果和新服务器复现步骤见仓库根目录的 `MIGRATION_HANDOFF_2026-08-24.md`。

## 核心结论

- 数据集：SatVideoIRSDT_v1。
- 模型主干：DeepPro-Plus_BRTD3。
- 损失：F1CalibratedOHEM。
- 随机种子：47。
- 4 种结构分别进行 pretrained 与 scratch 配对训练，共 8 个任务，对应 GPU 0--7。
- 全局最佳：`raw_apmd_hybrid_rms_motion_detrend_multiscale_contrast` + pretrained。
- 全局最佳选中 epoch：75。
- 全局最佳后处理：threshold=0.17，min_area=2。
- 全局最佳本地代理指标：Precision=0.933339，Recall=0.694785，F1=0.796586。
- 四个结构上 pretrained 均优于 scratch，F1 增益依次为 +0.013183、+0.016950、+0.014857、+0.026110。

这里的 Precision/Recall/F1 是项目后处理器基于本地验证标签计算的 proxy 指标，不应与比赛服务器的官方 Score 直接等同。

## 8 组最终选优结果

| 结构 | 初始化 | 选中 epoch | threshold | min_area | Precision | Recall | F1 | pretrained 相对 scratch |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| hybrid_rms | pretrained | 24 | 0.10 | 2 | 0.916369 | 0.690557 | 0.787597 | +0.013183 |
| hybrid_rms | scratch | 86 | 0.22 | 2 | 0.948090 | 0.654517 | 0.774414 | 基线 |
| hybrid_rms + motion_detrend | pretrained | 55 | 0.17 | 2 | 0.929883 | 0.685829 | 0.789424 | +0.016950 |
| hybrid_rms + motion_detrend | scratch | 72 | 0.10 | 2 | 0.934858 | 0.658154 | 0.772474 | 基线 |
| hybrid_rms + multiscale_contrast | pretrained | 24 | 0.10 | 2 | 0.923900 | 0.679033 | 0.782763 | +0.014857 |
| hybrid_rms + multiscale_contrast | scratch | 86 | 0.13 | 2 | 0.923623 | 0.657119 | 0.767906 | 基线 |
| hybrid_rms + motion_detrend + multiscale_contrast | pretrained | 75 | 0.17 | 2 | 0.933339 | 0.694785 | **0.796586** | **+0.026110** |
| hybrid_rms + motion_detrend + multiscale_contrast | scratch | 39 | 0.14 | 2 | 0.931505 | 0.656915 | 0.770476 | 基线 |

同一数据也保存在 `results.csv`，便于脚本读取。

## 目录内容

每个 `runs/<run_slug>/` 包含：

- `checkpoint_epoch_<N>.pth`：由全 epoch/阈值扫描最终选中的权重，不是简单复制训练损失最低权重；
- `training.log`：原始训练日志；
- `swanlab_sidecar.log` 与 `swanlab_sidecar.json`：SwanLab 补传过程及状态；
- `selected_submission.json`：最终阈值、最小面积、P/R/F1 与计数；
- `sweep_epoch_<N>.json/.csv`：该 epoch 的完整阈值/面积扫描；
- `eval_epoch_<N>.txt`：选中 epoch 的分割评估输出；
- `submission_validation.txt`：最终 ZIP 的格式/数量校验；
- `pipeline.status`：流水线完成记录；
- `submission_best_proxy_f1.zip` 及原始 `.sha256`：可直接用于提交的质心轨迹文件包。

`source_snapshot/` 保存了全局最佳实验目录内冻结的模型、适配器与损失源码。仓库根目录中的对应源文件是迁移时的工作版本；快照用于确认当次实验实际运行的代码。

`SHA256SUMS` 对发布集内所有实验产物做统一校验。迁移或下载后在本目录执行：

```bash
sha256sum -c SHA256SUMS
```

## 复现实验

迁移后先按根目录交接文档修改数据集、仓库和 Conda 环境硬编码路径。确认 8 张 GPU 空闲后，从仓库根目录执行：

```bash
conda activate sjyPID
bash tools/launch_hybrid_rms_pretrain_ablation_8gpu.sh --dry-run
bash tools/launch_hybrid_rms_pretrain_ablation_8gpu.sh
```

启动器使用 `screen` 创建 8 个独立会话，GPU 0/1、2/3、4/5、6/7 分别构成同结构的 pretrained/scratch 配对。默认使用 seed 47、SwanLab cloud 模式、Top-5 epoch 后处理和 `0.10:0.95:0.01` 阈值扫描。

## 未纳入 GitHub 的内容

- 数据集与标注；
- Conda 环境本体；
- 每个实验的全量 epoch checkpoint；
- `postprocess/probabilities/` 等逐帧概率缓存；
- `log/`、`swanlog/` 下的其他历史运行和重复生成物；
- 3GB 完整迁移压缩包。

这些内容仍保留在完整迁移归档 `DeepPro-main_full_transfer_2026-08-24.tar.gz` 中。
