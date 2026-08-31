# DeepPro / CSIG2026 文档索引

本目录同时保存当前研发说明和历史研究记录。新实验应优先服从当前代码中的安全检查，
再参考本索引标为“当前”的文档；历史文档中的旧服务器路径、预训练流程和 GPU 方案只
用于结果审计，不应直接复制执行。

## 当前主线

比赛提交阶段已经结束。最终 scratch-only Hybrid-RMS 提交 ID `907655` 得分
**91.30**；完整 checkpoint、源码快照、环境、最终 ZIP、阈值扫描和复现脚本见
[`release/2026-08-29_final_submission_score91.30_scratch/`](../release/2026-08-29_final_submission_score91.30_scratch/README.md)。

| 文档 | 内容 | 状态 |
|---|---|---|
| [F1 最大化跨领域研究与 PointCenter 决策](F1_MAXIMIZATION_RESEARCH_2026-08-27.md) | 数据结构、跨领域证据、PointCenter 网络/损失、候选队列与验收判据 | 当前优先实验 |
| [模型演进、当前框架与损失](MODEL_EVOLUTION_ARCHITECTURE_AND_LOSS_2026-08-26.md) | 从 DeepPro 到当前 BRTD3/Raw-APMD 的结构、框架图、模块增删、优缺点和损失公式 | 当前总览 |
| [Scratch-only 模型改进](../SCRATCH_MODEL_IMPROVEMENT_2026-08-25.md) | 非零投影、bandpass、detail 三候选的设计与验收 | 当前实验依据 |
| [网站结果分析](../WEBSITE_RESULTS_ANALYSIS_2026-08-25.md) | 八个历史配对结果、scratch-init 结果及 scratch-only 决策 | 当前策略依据 |
| [迁移验收](../MIGRATION_ACCEPTANCE_2026-08-25.md) | 新服务器路径、环境、数据和运行检查 | 当前运行基线 |

当前不可绕过的规则：

- 新训练只能随机初始化；`base_ckpt`、`spatial_ckpt`、`st_ckpt` 均被拒绝；
- 只允许物理 GPU 0、1、2，GPU3 不进入训练或后处理任务；
- 三张卡一次训练一个网络；
- 当前优先实验为 scratch PointCenter，使用逐目标中心热图与过滤前后信息一致性；
- 训练结束必须完成 Top-5、质心阈值扫描、轨迹生成、ZIP 校验和 SHA256；
- SwanLab 云端异常不能使已经完成的本地训练失效。

## 迁移与完整交接

| 文档 | 内容 | 使用方式 |
|---|---|---|
| [完整迁移交接](../MIGRATION_HANDOFF_2026-08-24.md) | 数据、模型、实验、环境和旧服务器完整记录 | 历史事实与故障排查；部分路径和预训练建议已过时 |
| [环境 YAML](environment_sjyPID_2026-08-24.yml) | Conda 环境导出 | 重建环境 |
| [Conda 显式包清单](conda_explicit_sjyPID_2026-08-24.txt) | 精确 Conda 包版本 | 环境审计 |
| [pip freeze](pip_freeze_sjyPID_2026-08-24.txt) | Python 包版本 | 环境审计 |

## 历史结构研究

| 文档 | 内容 | 当前结论 |
|---|---|---|
| [BRTD2 research](brtd2_research.md) | 深层语义适配、3/5/9 帧时域和门控设计 | no-gate 证据影响了 Raw-APMD；旧预训练命令不得执行 |
| [Raw-APMD](raw_apmd.md) | 原始外观、多尺度一/二阶运动与局部对比 | 已成为当前结构母体 |
| [结构优化分析](structure_optimization_2026-08-20.md) | RMS、Channel-RMS、去趋势、多尺度对比 | 促成 Hybrid-RMS；scratch 下扩展模块未成为默认 |
| [第二轮结构候选](structure_round2.md) | 对齐、双向传播、低频净化等八候选 | 保留为候选档案，不是当前执行队列 |
| [实验分析](experiment_analysis_2026-08-20.md) | Raw-APMD 与早期结构的实验复盘 | 历史对照 |

## 对话与决策归档

- [2026-08-12 BRTD 交接](BRTD_CONVERSATION_HANDOFF_2026-08-12.md)
- [2026-08-20 BRTD 交接](BRTD_CONVERSATION_HANDOFF_2026-08-20.md)
- [早期对话交接](CONVERSATION_HANDOFF.md)

这些文件用于追溯当时的假设和决策，不代表当前默认配置。

## 实验发布集

`release/2026-08-29_final_submission_score91.30_scratch/` 是最终提交的完整可复现
发布目录，也是提交阶段结束后的首选入口。它只使用随机初始化训练的 epoch-86
Hybrid-RMS 权重，包含网站 91.30 分结果记录、自适应阈值证据和端到端重新生成脚本。

`release/2026-08-22_pretrained_vs_scratch_seed47/` 是八组 Hybrid-RMS
pretrained/scratch 对照的可审计快照，包含选中 checkpoint、训练日志、阈值扫描、提交
ZIP 和校验和。它保留历史证据，但当前 scratch-only 训练不得加载其中权重。

## 代码入口

| 任务 | 文件 |
|---|---|
| 训练 | `train.py` |
| 推理/概率导出 | `test.py` |
| 当前模型 | `networks/models/DeepPro-Plus_BRTD3_PointCenter.py` |
| 结构适配器 | `networks/layers/structure_adapters.py` |
| 损失函数 | `networks/losses/segmentation_losses.py` |
| 运行环境和 GPU 白名单 | `tools/project_runtime_env.sh` |
| 完整结构实验流水线 | `tools/run_structure_candidate_experiment.sh` |
| PointCenter 三卡训练与自动 ZIP | `tools/run_pointcenter_f1_experiment.sh` |
| 后处理恢复 | `tools/resume_structure_candidate_postprocess.sh` |
