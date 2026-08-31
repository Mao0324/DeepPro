# CSIG2026 最终提交复现包：Scratch Hybrid-RMS，91.30

本目录冻结了 CSIG2026 赛道一最终提交（网站 ID `907655`）的完整提交阶段。
网站于 2026-08-29 22:26 完成评分，最终分数为 **91.30**。上传文件名在网站中
显示为 `submit_hrms_scratch_epoch86_adaptiv.zip`；本目录保留语义更完整的本地文件名。

## 最终结论

- 模型：`DeepPro-Plus_BRTD3`，结构 `raw_apmd_hybrid_rms`；
- checkpoint：epoch 86，seed 47；
- 初始化：完全随机初始化；
- 预训练权重：**未使用**；
- 普通序列：阈值 `0.16`、最短轨迹 `3` 帧；
- 1280×1024 序列 `000204`、`000205`：阈值 `0.96`、最短轨迹 `4` 帧；
- 最小连通区域：`2`；
- 全验证集自适应轨迹 F1：`0.780460890`；
- 网站最终分数：`91.30`。

最终提交文件：

```text
artifacts/submit_hrms_scratch_epoch86_adaptive_thr0p16_highres0p96.zip
SHA256 7348c804cf3f6e8f1142fdee0dccc8621fc8e681065e0abc05c37d9499e3437b
```

## 目录内容

```text
artifacts/       网站提交 ZIP 与解压后 TXT 内容哈希
checkpoint/      scratch-only epoch-86 checkpoint
environment/     sjyPID Conda 环境快照
evidence/        训练、推理、轨迹、格式审计和来源证据
scripts/         发布包验证与正式测试集重新生成脚本
source_snapshot/ 训练和提交关键代码的逐文件快照
validation/      全验证集 AMP 阈值扫描及大分辨率专项扫描
```

## 快速验证发布包

在仓库根目录执行：

```bash
bash release/2026-08-29_final_submission_score91.30_scratch/scripts/verify_release.sh
```

该命令验证所有发布文件哈希、checkpoint 元数据、无预训练训练证据、ZIP CRC、
220 个序列/21,285 帧格式，以及发布快照与本分支关键源码的一致性。

## 从 checkpoint 重新生成提交

准备数据目录，结构必须为：

```text
SatVideoIRSDT_v1/test/img/<sequence>/<frame image>
```

默认沿用 `tools/project_runtime_env.sh` 的数据路径。也可以显式设置：

```bash
DATA_ROOT=/path/to/SatVideoIRSDT_v1 \
FINAL_SUBMISSION_GPU=0 \
bash release/2026-08-29_final_submission_score91.30_scratch/scripts/reproduce_submission.sh
```

脚本只允许物理 GPU 0、1、2，并执行：

1. 校验 release 文件和 scratch checkpoint；
2. 复制冻结的模型/适配器源码到独立运行目录；
3. 以 40 帧、AMP、分块推理导出 220 个测试序列的概率图；
4. 应用普通/大分辨率自适应阈值和轨迹过滤；
5. 生成顶层 TXT ZIP，并运行比赛格式校验器；
6. 输出新 ZIP、SHA256、推理日志和轨迹日志。

运行产物写入被 Git 忽略的 `log/sem_seg/reproduction/`，不会污染发布目录。

2026-08-31 已在 RTX 3090 24GB 上实际执行完整脚本。重新生成包与发布包的 220 个
TXT 文件逐字节一致；详见 `evidence/reproduction_verification.md`。两次 ZIP 的整体
SHA256 不同仅由 ZIP 成员时间戳造成，不影响提交内容。

## 训练来源与可复现边界

`evidence/training.log` 保存了原始训练全过程。日志明确包含：

```text
base_ckpt=''
spatial_ckpt=''
st_ckpt=''
resume='never'
Initialized ... from random weights; no base checkpoint loaded.
Starting a new experiment from scratch.
```

checkpoint 内同时保存了模型配置、模型状态、优化器状态和 early-stopping 状态。
原训练设置 `deterministic=0`，因此重新训练可复现训练方法和配置，但不承诺不同 GPU
上的 checkpoint 逐字节一致；使用已发布 checkpoint 重新生成提交是本目录的主要复现路径。

## 最终 ZIP 不变量

- 220 个顶层 TXT，无子目录；
- 21,285 帧；
- 48,673 个检测点、1,766 条轨迹；
- 帧号连续，字段数合法；
- 坐标顺序为 `x y`，全部位于图像范围；
- 普通轨迹至少 3 帧，大分辨率轨迹至少 4 帧；
- ASCII、无 BOM、LF 换行；
- 单帧最大目标数 12；
- checkpoint SHA256：`63d620dedfeab5a58610b90f7a912176368d3e9e48f402364982a052f70373f4`。

网站分数来自比赛结束时保存的提交页面记录；隐藏测试标签和网站评分器不在本仓库中。
