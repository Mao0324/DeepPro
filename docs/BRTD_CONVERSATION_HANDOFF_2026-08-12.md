# DeepPro / BRTD 新对话交接文档

> 生成时间：2026-08-12 03:50 UTC  
> 适用仓库：`/home/devbox/project/model/sjy/CSIG2026/Deeppro_v2/DeepPro-main`  
> 本文档取代旧的 2026-07-22 交接文档作为当前入口；旧文档仍保留用于追溯。

## 0. 新对话应先做什么

新 Codex 对话开始后，请先完整阅读本文件。不要立即启动、停止、清理或重跑实验；先做以下只读检查：

```bash
cd /home/devbox/project/model/sjy/CSIG2026/Deeppro_v2/DeepPro-main
git status --short --branch
screen -ls
nvidia-smi
```

然后核对 `log/sem_seg/2026-08-11/_screen_logs/` 下 8 个日志的最新 epoch。本文的运行状态是时间快照，可能在新对话开始前已经变化。

## 1. 总目标与当前优先级

项目目标是在 JinSight Challenge V3 的红外视频卫星空中动目标检测赛道提高最终得分。轨迹完整度和轨迹准确度目前已能拿满，当前主优化目标是比赛的质心检测 F1。

最高优先级：

1. 等待当前 8 个 ValidFrames/F1-OHEM 消融任务完成。
2. 不以 pixel IoU/pixel F1 单独选模型；对多个 checkpoint 做质心代理 F1 扫描。
3. 比较 padding 有效帧掩码和 BRTD/BRTD2 组件消融是否真实提升。
4. 先做低成本后处理与模型融合优化，再决定是否启动 BRTD3 八卡结构实验。
5. 结构收益最终至少补两个随机种子；当前大多数实验仅使用 seed 46 且 `deterministic=0`。

## 2. 环境、路径与 Git 基线

| 项目 | 当前值 |
|---|---|
| 仓库 | `/home/devbox/project/model/sjy/CSIG2026/Deeppro_v2/DeepPro-main` |
| 数据集 | `/home/devbox/project/model/sjy/CSIG2026/datasets/SatVideoIRSDT_v1` |
| Python | `/home/devbox/project/model/miniconda3/envs/sjyPID/bin/python` |
| Conda 环境 | `sjyPID` |
| 分支 | `BRTD-Adapter` |
| 当前提交 | `5052f64`，`添加 DeepPro-Plus_BRTD3 模型及相关工具用于结构候选实验` |
| 工作区 | 生成本文档前代码工作区干净 |
| 主远程 | `CSIG2026 https://github.com/Mao0324/DeepPro.git` |
| 另一远程 | `origin https://github.com/ShiJiayu-del/DeepPro-main.git` |
| 预训练权重 | `pretrained/SatVideoIRSDT_DeepPro-Plus_pretrained_init.pth` |
| 日志根目录 | `log/sem_seg`，按日期分组 |

`HEAD` 与 `CSIG2026/BRTD-Adapter` 一致；`origin/BRTD-Adapter` 落后于当前主线。后续推送目标若无新指示，应先确认使用 `CSIG2026/BRTD-Adapter`。

## 3. 已确认的平台成绩与基线

轨迹两项为满分时，总分减去 10 可近似看作检测 F1 百分数：

| 方案 | 轮次/选择 | 总分 | 网站检测 F1 | 结论 |
|---|---:|---:|---:|---|
| DeepPro-Plus + F1-OHEM | 约 50 轮 / epoch45 | 84.91 | 约 0.7491 | 最强单模型历史基线 |
| DeepPro-Plus + F1-OHEM | 100 轮 | 84.13 | 约 0.7413 | 继续训练反而下降 |
| BRTD + F1-OHEM | 50 轮 | 82.35 | 约 0.7235 | 召回偏低 |
| BRTD + F1-OHEM | 100 轮 | 83.23 | 约 0.7323 | 延长训练有收益，但仍低于 DeepPro |
| BRTD2 + F1-OHEM | 最佳约 epoch45 | 80.89 | 约 0.7089 | 后续基本平台化 |
| DeepPro epoch45 + BRTD 融合 | 阈值/面积扫描 | **85.59** | **0.7559** | 当前网站最高记录 |

质心代理评估记录：

- DeepPro epoch45：Precision `0.92094`，Recall `0.64637`，F1 `0.75960`。
- DeepPro epoch95/100 最佳模型：F1 `0.75747`。
- BRTD 最佳模型：F1 `0.73819`。
- DeepPro epoch45 + BRTD：Precision `0.93315`，Recall `0.64528`，F1 `0.76296`。
- 网站融合 F1 比代理低约 `0.007`，但历史排序一致。

当前主要瓶颈是召回：即使最佳融合仍漏检约 35% 的目标。BRTD 的已证实价值主要是过滤 DeepPro 假阳性和提供互补性，而不是单模型提高召回。

关键证据文件：

- `analysis/f1_optimization_2026-08-07/results/selected_ensemble.json`
- `analysis/f1_optimization_2026-08-07/results/deeppro_epoch45.json`
- `analysis/f1_optimization_2026-08-07/results/deeppro_best_e100.json`

## 4. 当前正在运行的 8 个实验

状态快照：2026-08-12 03:50 UTC。8 个 Screen 会话仍为 Detached，8 张 A100 均有训练进程占用显存；快照时 GPU 5 利用率 100%，其余 GPU 多处于验证或数据等待阶段。

| GPU | Screen 会话 | 实验 | 已进入 | 最近 pixel F1 | 当前记录 best mIoU |
|---:|---|---|---:|---:|---:|
| 0 | `csig2a_g0_deeppro_validmask` | DeepPro + valid-frame mask | 61/100 | 0.493579 | 0.341305 |
| 1 | `csig2a_g1_brtd_full_validmask` | BRTD full + valid-frame mask | 55/100 | 0.460284 | 0.303538 |
| 2 | `csig2a_g2_brtdv2_full_validmask` | BRTD2 full + valid-frame mask | 52/100 | 0.478725 | 0.319403 |
| 3 | `csig2a_g3_brtdv2_no_background` | BRTD2 no background | 52/100 | 0.458758 | 0.309069 |
| 4 | `csig2a_g4_brtdv2_fixed_router` | BRTD2 fixed router | 52/100 | 0.469206 | 0.317583 |
| 5 | `csig2a_g5_brtdv2_no_gate` | BRTD2 no gate | 53/100 | 0.485302 | 0.324185 |
| 6 | `csig2a_g6_brtd_no_background` | BRTD no background | 56/100 | 0.464349 | 0.302379 |
| 7 | `csig2a_g7_brtd_fixed_router` | BRTD fixed router | 55/100 | 0.450231 | 0.296327 |

日志目录：

```text
log/sem_seg/2026-08-11/_screen_logs
log/sem_seg/2026-08-11/_pipeline_status
```

实验目录统一位于：

```text
log/sem_seg/2026-08-11/SatVideoIRSDT_v1__2026-08-11__ValidFrames-F1OHEM-*_E100
```

重要：这 8 个任务启动于早停代码加入之前，启动命令也没有传早停参数，因此它们不会使用新早停机制。不要因为源码已更新而误认为正在运行的进程会自动早停。

常用监控命令：

```bash
screen -ls
screen -r csig2a_g0_deeppro_validmask
tail -f log/sem_seg/2026-08-11/_screen_logs/csig2a_g0_deeppro_validmask.log
nvidia-smi
```

从 Screen 退出但保持任务运行：按 `Ctrl+A`，再按 `D`。

## 5. 已实现的代码能力

### 5.1 统一训练与损失

`train.py` 已支持：

- DeepPro-Plus、BRTD、BRTD2、BRTD3。
- 原始 DeepPro-Plus 预训练权重非严格加载；只允许新 adapter 键缺失。
- backbone 与 adapter 分层学习率。
- `f1_calibrated_ohem` 和多种历史损失。
- padded frame 掩码。
- DDP、断点恢复、SwanLab、定期 checkpoint。
- 可恢复且 DDP 同步的早停。

### 5.2 早停

默认 `--early_stopping_patience 0`，即保持旧行为并关闭早停。未来 BRTD3 runner 显式使用：

```bash
--early_stopping_patience 30 \
--early_stopping_min_delta 0.0001 \
--early_stopping_start_epoch 15 \
--early_stopping_metric eval_f1
```

早停状态写入 checkpoint，断点续训会恢复最佳值和未提升计数；支持 `eval_f1`、`eval_iou` 最大化和 `eval_loss` 最小化。F1-OHEM 存在 warm-up，不能用早期 eval loss 作为可靠停止依据。

### 5.3 BRTD2

`docs/brtd2_research.md` 记录了设计依据。核心变化：

- adapter 从浅层 8 通道 stem 后移到 `layer1` 后的 32 通道语义特征。
- 3/5/9 帧真实多尺度时域感受野。
- 显式 appearance 路径，局部对比只作为门控证据。
- GroupNorm、保守初始 gate、零初始化输出投影。
- 加载 DeepPro-Plus 后首步 logits 与原模型完全一致。

### 5.4 BRTD3 / 结构第二轮

最新提交已实现 8 个单变量候选：

| GPU | `structure_variant` | 目的 |
|---:|---|---|
| 0 | `second_order` | 一阶/二阶时序异常 |
| 1 | `tdc_dual_stream` | 普通 3D 与时差双流 |
| 2 | `lfp_shallow` | 浅层低频引导净化 |
| 3 | `lfp_deep` | TPro 前低频引导净化 |
| 4 | `global_align` | 背景主导全局平移对齐 |
| 5 | `local_align` | 稠密局部特征对齐 |
| 6 | `multiscale_head` | 全分辨率多尺度上下文头 |
| 7 | `bidirectional` | 前后向递归传播 |

关键文件：

- `networks/models/DeepPro-Plus_BRTD3.py`
- `networks/layers/structure_adapters.py`
- `tools/check_brtd3.py`
- `docs/structure_round2.md`
- `tools/launch_structure_round2_8gpu.sh`
- `tools/run_structure_candidate_experiment.sh`

历史会话中已完成形状、反向传播、预训练兼容和初始 logits 等价性检查。正式启动前仍建议重新运行 `tools/check_brtd3.py` 并做一次 `--dry-run`。

### 5.5 评估与提交工具链

结构实验 runner 会自动执行：

1. 训练并保存定期 checkpoint。
2. 选取验证 pixel F1 靠前的三个定期 checkpoint 与 `best_model`。
3. 导出验证概率图。
4. 扫描阈值 `0.15:0.70:0.01` 和最小面积 `1,2,3`。
5. 按质心代理 F1 保留最佳结果。
6. 轨迹关联。
7. 生成 ZIP、严格校验并写 SHA-256。

关键工具：

- `tools/select_eval_checkpoints.py`
- `tools/centroid_f1_sweep.py`
- `tools/probability_ensemble_sweep.py`
- `tools/build_single_submission.py`
- `tools/validate_submission_zip.py`
- `tools_forSatVideoIRSTD/seg2tracked_centroid_txt.py`

注意：当前 checkpoint 候选先按 pixel F1 排名，仍可能漏掉质心 F1 最优轮次。后续应扩大扫描范围，而不是只信 `best_model.pth`。

## 6. 已确认的技术结论

1. 比赛按质心检测计分，本地 pixel IoU/pixel F1 只能作为辅助指标。
2. 轨迹分数已满，保持现有轨迹参数，优化集中到检测 F1。
3. DeepPro 延长到 100 轮出现更多碎片和虚警；BRTD 延长训练反而有一定收益，不同模型不能统一取最后一轮。
4. F1-OHEM 是当前最可靠的损失方向，但历史日志中的部分配置字段与损失内部真实参数不完全一致；若继续做损失消融，应先暴露并核对 `dice_weight`、`hard_weight`、`negative_ratio`、`min_negatives`、`margin`、`warmup_epochs`、`ramp_epochs`。
5. 目标通常只有几像素，验证集中大量目标相对局部背景更暗且 CNR 低。模型不能只依赖亮点增强或激进高通差分。
6. 大量训练窗口含 padding：此前统计 44.1% 候选窗口不满 40 帧，约 29.7% 的采样仍含 padding；当前 ValidFrames 实验正验证掩码收益。
7. BRTD 全模型用 `0.005` 学习率会严重失稳；较可靠配置是 adapter `0.001`、backbone `0.0001`。
8. BRTD 单模型弱，但与 DeepPro 融合有真实互补性。概率尺度差异大，应优先做 logit 校准后融合。
9. 历史行列坐标颠倒问题已经修复；不要再次交换坐标。严格 ZIP 校验必须按每个序列真实图像尺寸检查，不能写死网络 patch 尺寸。

## 7. 下一步建议顺序

### A. 当前 8 个任务完成后

1. 确认 8 个 status 文件和 `COMPLETE` 标记，不只看 Screen 是否存在。
2. 收集每个实验 35–100 轮的多个 checkpoint；至少按每 5 轮扫描，而不是只取最终或 pixel best。
3. 对所有候选运行同一套质心阈值/最小面积扫描，比较 Precision、Recall、F1。
4. 先回答两个问题：valid-frame mask 是否提升 DeepPro；BRTD/BRTD2 哪个组件造成召回损失。
5. 对最优 1–2 个结果生成严格验证通过的 ZIP，再提交网站验证代理排序。

### B. 无需重新训练的优先优化

1. 扩大历史 checkpoint 扫描。
2. 分别拟合 temperature/bias，在校准后的 logit 上融合 DeepPro、BRTD、BRTD2。
3. 细扫融合权重、阈值和最小面积，并用序列级交叉验证降低后处理过拟合。
4. 尝试 EMA/SWA 或邻近 checkpoint 平均。
5. 评估局部峰值/NMS，减少相邻目标被单个连通域合并的问题。

### C. 需要训练的优先优化

1. 数据增强：水平/垂直翻转、90°旋转、时序反转、轻微 gamma/增益/噪声/模糊。
2. adapter-only warm-up 5–10 轮，再以 10 倍更小 LR 解冻 backbone。
3. 独立中心热图头 + 困难正样本损失；不要直接复用旧 `stc_f1`。
4. 当前实验全部完成且 8 张 GPU 真正空闲后，再启动 BRTD3 第二轮。
5. 对最优方案补至少两个随机种子；小于约 0.3–0.5 个 F1 百分点的单 seed 差异暂不视为确定收益。

## 8. BRTD3 启动方式与安全约束

只预览，不启动：

```bash
bash tools/launch_structure_round2_8gpu.sh --dry-run
```

仅当 8 张 GPU 都空闲后启动：

```bash
bash tools/launch_structure_round2_8gpu.sh
```

launcher 检测到任意 GPU 显存使用超过 1024 MiB 就会拒绝启动。不要绕过此保护，也不要与当前 8 个 ValidFrames 任务抢卡。

## 9. 不要做的事情

- 不要停止当前 Screen/GPU 进程，除非用户明确要求并已确认 checkpoint 可恢复。
- 不要根据 `nvidia-smi` 单次 0% 利用率判断进程已死；验证和数据等待期间可能为 0%。
- 不要删除旧 Screen socket、日志、checkpoint、提交 ZIP 或分析目录，除非先精确确认目标。
- 不要只用 `best_model.pth` 或最后一轮提交。
- 不要把 pixel F1 当成网站 F1。
- 不要在一次实验中同时改变结构、损失、采样、阈值和后处理。
- 不要重新引入行列坐标交换。
- 不要把运行中进程自动继承后来修改的源码或早停配置。

## 10. 历史资料入口

- 本交接：`docs/BRTD_CONVERSATION_HANDOFF_2026-08-12.md`
- 旧综合交接：`docs/CONVERSATION_HANDOFF.md`
- BRTD2 设计：`docs/brtd2_research.md`
- BRTD3 第二轮：`docs/structure_round2.md`
- 恢复索引：`/home/devbox/project/model/sjy/CSIG2026/recovered_codex_chat/BRTD_network/README.md`
- 可读历史对话：`/home/devbox/project/model/sjy/CSIG2026/recovered_codex_chat/BRTD_network/BRTD_network_readable_transcript.md`
- 原始事件流：`/home/devbox/project/model/sjy/CSIG2026/recovered_codex_chat/BRTD_network/*.jsonl`

仅在需要追溯某一具体决策时读取完整历史对话；日常续接以本文件、当前代码和当前日志为准。

## 11. 可直接粘贴到新 Codex 对话的开场提示

```text
请先完整阅读：
/home/devbox/project/model/sjy/CSIG2026/Deeppro_v2/DeepPro-main/docs/BRTD_CONVERSATION_HANDOFF_2026-08-12.md

这是 DeepPro/BRTD 比赛项目的最新交接。先不要启动、停止、删除或修改任何训练任务。
请先只读检查：
1. git status 和当前提交；
2. screen -ls 与 nvidia-smi；
3. log/sem_seg/2026-08-11/_screen_logs 下 8 个实验的最新 epoch、异常和指标；
4. _pipeline_status 与各实验 COMPLETE 标记。

先向我汇报当前状态，并说明与交接快照相比发生了什么变化。之后再根据我的指示继续评估、生成提交包或启动 BRTD3。
```

