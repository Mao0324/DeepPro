# DeepPro / BRTD 项目完整迁移与技术交接

> 生成日期：2026-08-24（UTC）
> 源项目：`/home/devbox/project/model/sjy/CSIG2026/Deeppro_v2/DeepPro-main`
> Git 分支：`BRTD-Adapter`
> Git HEAD：`e47cc6398e3c19ea0f6dceee16b10660eefd79e0`
> 数据集：`SatVideoIRSDT_v1`（位于项目目录外，**不在本项目压缩包中**）
> 当前状态：最后一轮 8 卡 Hybrid-RMS 预训练/scratch 对照实验已全部完成；GPU 空闲；无运行中的 Screen。

---

## 0. 交接摘要：新服务器接手人首先需要知道什么

本项目是以 DeepPro-Plus 为基础的红外卫星视频弱小运动目标分割/质心检测工程。
最终比赛输出不是像素掩码，而是每帧目标质心及轨迹 TXT，再打包为比赛 ZIP。

当前最值得继续使用的模型是：

```text
DeepPro-Plus_BRTD3
structure_variant = raw_apmd_hybrid_rms_motion_detrend_multiscale_contrast
initialization    = pretrained
seed              = 47
loss              = f1_calibrated_ohem
```

它在本地 23,087 帧验证集的质心匹配代理指标为：

| 指标 | 值 |
|---|---:|
| 最优 checkpoint | epoch 75 |
| 后处理阈值 | 0.17 |
| 最小连通域面积 | 2 |
| Precision | 0.933339 |
| Recall | 0.694785 |
| Proxy F1 | **0.796586** |
| TP / FP / FN | 61,130 / 4,366 / 26,854 |

对应实验相对同结构 scratch 的 Proxy F1 提升 `0.026110`，说明当前数据和训练
配置下，预训练权重是明确有效的；不要在没有新证据时改回随机初始化。

完整项目目录约 `5.4 GB`，包含代码、`.git`、预训练初始化权重、历史训练日志、
checkpoint、概率图、后处理结果、SwanLab 本地缓存和提交 ZIP。以下两项在目录外，
必须另行迁移或重建：

1. 数据集：约 `50 GB`；
2. Conda 环境 `sjyPID`：约 `8.4 GB`，推荐按本文提供的环境清单重建，而不是直接复制。

---

## 1. 项目目标、数据和评分口径

### 1.1 任务目标

输入是红外卫星视频序列，目标通常只有几像素，低信噪比、低对比度，并混有相机
运动、背景纹理和瞬态噪声。网络输出每帧分割 logits；后处理把二值连通域转为质心，
再关联为轨迹。

### 1.2 数据尺度

当前使用数据集：

```text
/home/devbox/project/model/sjy/CSIG2026/datasets/SatVideoIRSDT_v1
```

数据集约 `50 GB`，不位于 `DeepPro-main` 内，因此本次项目压缩包不会包含数据集。
验证集统计：

| 目标尺度 | 25% | 中位数 | 75% | 95% |
|---|---:|---:|---:|---:|
| 面积（像素） | 4 | 6 | 9 | 18 |
| 宽度（像素） | 2 | 3 | 3 | 5 |
| 高度（像素） | 2 | 3 | 3 | 5 |

79.2% 的目标面积不超过 9 像素，93.7% 不超过 16 像素。这是项目始终保留原始
空间分辨率、不增加下采样，并采用 3/5/7 小尺度上下文的直接依据。

### 1.3 三种指标不要混淆

1. `pixel IoU / pixel F1`：训练验证日志中的像素指标，用于早停和初筛 checkpoint；
2. `centroid proxy F1`：本地按质心匹配规则计算，决定最终本地候选；
3. 网站 Score：隐藏测试集总分，包含检测与轨迹评分，不能与本地 Proxy F1 等同。

历史上 pixel F1 最优 epoch 经常不是 centroid F1 最优 epoch，因此最终流程会先取
pixel F1 Top-5 checkpoint，再逐个导出概率并扫描质心阈值和面积。

---

## 2. 目录结构和关键文件

```text
DeepPro-main/
├── README.md
├── MIGRATION_HANDOFF_2026-08-24.md       # 本文件
├── train.py                              # 统一训练入口
├── test.py                               # 推理/概率导出入口
├── runtime_utils.py                      # DDP、随机种子、checkpoint 工具
├── sequence_utils.py                     # 序列/有效帧辅助逻辑
├── ShootingRules.py                      # 质心匹配评估
├── networks/
│   ├── models/DeepPro-Plus.py            # 基础网络
│   ├── models/DeepPro-Plus_BRTD.py       # BRTD1
│   ├── models/DeepPro-Plus_BRTD2.py      # BRTD2
│   ├── models/DeepPro-Plus_BRTD3.py      # 可选择结构适配器的当前网络
│   ├── layers/structure_adapters.py      # 当前全部 BRTD3/raw-APMD 结构
│   └── losses/segmentation_losses.py     # 全部可选分割损失
├── data_utils/                           # 训练/测试数据加载器
├── tools/                                # 训练、筛选、融合、打包、自检脚本
├── tools_forSatVideoIRSTD/               # 质心和轨迹 TXT 工具、比赛说明
├── pretrained/
│   └── SatVideoIRSDT_DeepPro-Plus_pretrained_init.pth
├── log/sem_seg/                          # 完整实验日志、checkpoint、后处理、ZIP
├── analysis/                             # 历史概率融合与阈值实验
├── docs/                                 # 历史研究、实验复盘、环境清单
├── paper/                                # DeepPro 论文 PDF
├── swanlog/                              # 历史 SwanLab 本地缓存
└── .git/                                 # 完整 Git 历史，约 751 MB
```

环境复现文件：

```text
docs/environment_sjyPID_2026-08-24.yml
docs/conda_explicit_sjyPID_2026-08-24.txt
docs/pip_freeze_sjyPID_2026-08-24.txt
```

---

## 3. Git 状态与修改历史

### 3.1 当前 Git 状态

分支为 `BRTD-Adapter`，HEAD 为：

```text
e47cc63 删除没用的脚本
```

远程：

```text
CSIG2026  https://github.com/Mao0324/DeepPro.git
origin    https://github.com/ShiJiayu-del/DeepPro-main.git
```

当前工作树并不干净。最终 Hybrid-RMS 实验依赖一批尚未提交的修改，因此迁移时
必须传整个工作树；只从远程 Git clone 会丢失最终结构和启动工具。

截至打包前的重要未提交内容：

- `networks/layers/structure_adapters.py`：RMS、Channel-RMS、Hybrid-RMS、运动去趋势、
  多尺度对比度及组合结构；
- `networks/models/DeepPro-Plus_BRTD3.py`：所有 `raw_apmd*` 变体接入 raw fusion；
- `train.py`：结构白名单、scratch 初始化审计日志；
- `tools/check_brtd3.py`：预训练恒等、梯度、padding、Hybrid-RMS 和尺度权重自检；
- `tools/run_structure_candidate_experiment.sh`：预训练/scratch 双模式、Top-5、后处理；
- `tools/launch_hybrid_rms_pretrain_ablation_8gpu.sh`：最终 8 卡矩阵；
- `tools/stream_training_log_to_swanlab.py` 和侧车启动器；
- `tools/centroid_f1_sweep.py`：限制数值库/OpenCV 线程，改善后处理并行效率；
- 多份 8 月 20 日之后的分析和交接文档；
- `tools/launch_raw_apmd_3gpu.sh` 已从工作树删除，避免误跑旧的三 seed 方案。

迁移后不要执行 `git reset --hard`、`git clean` 或 `git checkout -- .`，否则会删除
已经产生最终结果但尚未提交的代码。

### 3.2 从基础 DeepPro 到当前模型的提交时间线

| 日期 | Commit | 变化 |
|---|---|---|
| 2026-06-28 | `96b2ea0` | 项目初始化，形成基础训练工程 |
| 2026-07-01 | `ef9674e` | 训练加入 SwanLab |
| 2026-07-01 | `54b0a04` | 加入时空注意力卷积机制 |
| 2026-07-02 | `b6c9a83` | 三阶段训练策略 |
| 2026-07-06 | `74f1c0a` | 增加 TDCSTA |
| 2026-07-15 | `ddcb133` | 创建第一版 BRTD 网络 |
| 2026-07-21 | `d2cc7b3` | 正式加入 `DeepPro-Plus_BRTD.py` 和 `brtd_adapter.py` |
| 2026-07-22 | `22b4a1b` | 增加 Precision/Recall/F1 训练验证指标 |
| 2026-07-22 | `affc323` | 统一二值分割损失模块，多种 loss 可切换 |
| 2026-08-05 | `7d42c94` | 重构训练参数、日志和入口 |
| 2026-08-06 | `9dae019` | 整理 DeepPro-Plus 与分割损失实现 |
| 2026-08-07 | `9efed80` | 加入 `f1_calibrated_ohem` 和对应实验模型 |
| 2026-08-11 | `5052f64` | 加入 BRTD3 和 8 个结构候选工具链 |
| 2026-08-17 | `c1d8ba3` | 加入 raw-APMD 原始帧外观/运动/对比度结构 |
| 2026-08-20~23 | 未提交工作树 | RMS 系列、motion detrend、multiscale contrast、Hybrid-RMS、8 卡预训练对照、SwanLab 侧车和最终结果 |

---

## 4. 基础 DeepPro-Plus 网络

核心文件：`networks/models/DeepPro-Plus.py`。

### 4.1 前向结构

输入：`[B, 1, T, H, W]`，最终实验使用 `T=40`。

```text
raw frames
  │
  ├─ SDifferenceConv(1→8, kernel=5×7×7, padding=2×3×3)
  ├─ BatchNorm3d + ReLU
  ├─ STD_Resblock(8→16)
  ├─ STD_Resblock(16→32)
  ├─ permute: 每个空间位置形成一条 40 帧 temporal profile
  ├─ TPro(d_model=32, num_head=8, seqlen=40, out_len=40)
  ├─ 1×1×1 Conv(32→8) + BatchNorm3d + ReLU
  └─ 1×1×1 Conv(8→1) → squeeze → [B,T,H,W] logits
```

参数量约 `70,913`。

### 4.2 基础网络的优点

- SDifference/STD 特征对时空差异敏感，能抑制大面积背景，Precision 较高；
- TPro 对每个空间位置的长时间剖面建模，适合慢速、微弱、跨帧稳定目标；
- 不下采样，避免 2~5 像素目标被池化抹除；
- 网络极小，可以一张 A100 以较大 batch 训练。

### 4.3 基础网络的漏洞

- 差分主干会削弱静止/慢速且绝对强度微弱的目标；
- 固定像素时间差中混有目标运动与相机/背景相干运动；
- 训练 pixel 指标与比赛质心 F1 并不完全一致；
- 大量训练窗口包含补帧，如果不掩码会把 padding 当成真实负样本；
- 主要误差长期表现为 Recall 低，而不是 Precision 崩塌。

---

## 5. 网络结构演进过程

### 5.1 BRTD1：显式时域差分适配器

BRTD1 尝试在 DeepPro-Plus 上增加时域差分和背景/门控结构。其价值在于提供了
和 DeepPro 互补的假阳性过滤，但单模型召回下降，网站和本地 F1 都弱于基础模型。

历史结果：

| 方案 | 网站检测 F1（由总分扣除轨迹项近似） | 本地 Proxy F1 |
|---|---:|---:|
| DeepPro-Plus + F1-OHEM，约 epoch45 | 0.7491 | 0.75960 |
| BRTD + F1-OHEM，50 epoch | 0.7235 | 约 0.73819 |
| BRTD + F1-OHEM，100 epoch | 0.7323 | 约 0.73819 |
| DeepPro epoch45 + BRTD 融合 | **0.7559** | **0.76296** |

结论：BRTD1 单模型没有成功，但与 DeepPro 存在互补性。激进的时域高通和门控会
连同背景一起抑制弱目标，不能把差分分支当作唯一目标表征。

### 5.2 BRTD2：语义层适配、真实多尺度时域、保留外观

BRTD2 的主要修改：

1. adapter 从浅层 8 通道 stem 后移动到 `layer1` 后的 32 通道语义特征；
2. 时域卷积分支使用 dilation 1/2/4，形成真实 3/5/9 帧感受野；
3. 保留 appearance 分支，时域证据只做残差修正；
4. 局部对比度仅作为可靠性证据；
5. 使用 GroupNorm，避免小 batch 的统计漂移；
6. 输出 projection 零初始化，加载基础预训练权重时初始 logits 完全一致。

实验发现 no-gate 版本优于带输入依赖 gate 的版本。门控虽然理论上能抑制杂波，
但在弱目标任务中 gate 也会把低响应目标压掉。因此后续 raw-APMD 不再引入同类
Sigmoid gate。

8 月 13 日 BRTD2 no-gate 的学习率实验：

| Backbone LR | Proxy F1 |
|---:|---:|
| 0.0005 | 0.7597 |
| 0.0010 | 0.7702 |
| 0.0025 | 0.7723 |
| 0.0050 | 0.7760 |

同一轮 adapter LR 0.0005 的结果约 0.7764，但只比 0.001 高 0.0004，小于随机
波动，不能认为降低 adapter LR 已被证明有效。后续保持 adapter 0.001、backbone
0.005。

### 5.3 BRTD3：统一的零初始化结构候选框架

`DeepPro-Plus_BRTD3.py` 把基础 DeepPro-Plus 保持为主干，在不同位置插入
`brtd.*` 适配器。初始候选包括：

| Variant | 目的 |
|---|---|
| `second_order` | 显式一阶/二阶时序异常 |
| `tdc_dual_stream` | 普通 3D 与时间差分双流 |
| `lfp_shallow` / `lfp_deep` | 低频引导净化 |
| `global_align` | 背景主导全局平移对齐 |
| `local_align` | 局部稠密对齐 |
| `multiscale_head` | 原分辨率多尺度上下文头 |
| `bidirectional` | 正向/反向递归传播 |

统一设计规则是：适配器最后一层 projection 为 0，旧 DeepPro-Plus checkpoint
允许只缺失 `brtd.*` 键。因此插入适配器后第一步预测与基础模型严格一致，新结构
从残差 0 开始学习，减少破坏已有特征的风险。

### 5.4 Raw-APMD：原始外观 + 多尺度一/二阶运动 + 局部对比度

Raw-APMD 是结构搜索中最有证据的方向。它把原始帧直接送入一个独立外观旁路，
再和 backbone 的 32 通道特征融合，弥补差分主干遗漏绝对辐射信息的问题。

核心计算：

1. 原始外观编码 `A_t`；
2. 对时间步长 `s ∈ {1,2,4}` 计算邻域：

```text
D1_s(t) = 0.5 × (A_{t+s} - A_{t-s})
D2_s(t) = A_{t+s} - 2A_t + A_{t-s}
```

3. 每个 bottleneck 通道分别学习步长 softmax 权重；
4. 构造局部中心-周边对比；
5. 外观、运动、对比度融合；
6. 零初始化 1×1×1 projection 产生 residual delta，加到 backbone 特征。

padding 有效帧掩码贯穿外观/运动/融合，补帧区域不得产生响应。代码快照会被复制
到实验目录，推理时优先加载当时的 `structure_adapters.py`，防止后续源码变化导致
旧 checkpoint 无法复现。

Raw-APMD 相比 DeepPro 的配对提升：

| Seed | DeepPro Proxy F1 | Raw-APMD Proxy F1 | 提升 |
|---:|---:|---:|---:|
| 47 | 0.760839 | 0.785431 | +0.024592 |
| 49 | 0.770413 | 0.777190 | +0.006777 |

提升主要来自 Recall，同时保持约 0.93 的 Precision，证明独立原始外观旁路有效。

### 5.5 RMS 系列：解决 GroupNorm 移除绝对亮度的问题

原 Raw-APMD 的逐帧 GroupNorm 会减去帧内均值，与“保留绝对外观”的设计目标
冲突。因此依次实验了：

#### Framewise RMS

每帧跨通道与空间计算共同二阶矩，不减均值：

```text
r_t = sqrt(mean_{c,h,w}(x²) + eps)
y = x / r_t × gamma
```

#### Channel-RMS

每帧每通道独立计算空间二阶矩，避免强通道统一缩放弱通道：

```text
r_{c,t} = sqrt(mean_{h,w}(x²) + eps)
```

它取得历史单次网站最高 `88.33`，但两个 seed 网站分数为 `88.33 / 86.19`，波动
`2.14`；本地最优阈值也从 `0.39` 漂移到 `0.94`。说明纯逐通道能量估计可能
放大低能量通道噪声，校准和泛化不稳定。

#### Hybrid-RMS

最终采用共享帧二阶矩和逐通道二阶矩的可学习收缩：

```text
m_shared  = mean_{c,h,w}(x²)
m_channel = mean_{h,w}(x²)
lambda_c  = sigmoid(channel_mix_logit_c), 初始 0.5
m_blend   = (1-lambda_c) × m_shared + lambda_c × m_channel
y         = x / sqrt(m_blend + eps) × gamma
```

这让每个通道自行学习“共享稳定性”与“通道独立性”的比例，避免在设计阶段硬选
shared RMS 或 channel RMS。

### 5.6 Motion detrend：抑制相干背景运动

一、二阶上下文不仅包含目标运动，也包含相机和背景的低频一致运动。去趋势分支用
15×15 空间均值估计低频成分：

```text
M_local = M - AvgPool15x15(M)
```

15×15 明显大于 95% 目标宽高，目标更可能作为局部稀疏异常保留，相干背景运动
则被削弱。早期双 seed 网站结果的均值最高：`87.18 / 88.09`，平均 `87.64`。

### 5.7 Multiscale contrast：3/5/7 中心-周边尺度

固定 3×3/7×7 对比度被替换为每通道可学习的 3/5/7 softmax 混合：

```text
C_k = A - AvgPool_k(A), k ∈ {3,5,7}
C   = Σ softmax(w_k,c) × C_k
```

早期双 seed 网站结果 `87.31 / 87.32`，范围仅 `0.01`，稳定性最佳，但单独使用
没有提高最终上限。最终结果表明它与 motion detrend、预训练同时存在时有正交互，
不能仅凭单模块结果决定删除。

### 5.8 最终四结构 × 两初始化实验

最终采用固定 seed 47 的 2×2×2 因子实验：

- motion detrend：关/开；
- multiscale contrast：关/开；
- 初始化：pretrained/scratch。

每张 GPU 独立运行一个实验，共 8 个实验，不是一个 DDP 任务跨 8 卡。

参数量：

| 模型 | 总参数 | Adapter 参数 |
|---|---:|---:|
| DeepPro-Plus | 70,913 | 0 |
| Hybrid-RMS | 73,409 | 2,496 |
| Hybrid-RMS + Motion | 73,409 | 2,496 |
| Hybrid-RMS + Multiscale | 73,433 | 2,520 |
| Hybrid-RMS + Motion + Multiscale | 73,433 | 2,520 |

---

## 6. 损失函数演进

### 6.1 基础损失

早期模型文件内直接包含：

- HAM loss；
- HPM loss；
- BCEWithLogits；
- clip 级 SoftIoU。

原始 SoftIoU 对 `[B,T,H,W]` 在整个 clip 上求交并比，优化像素重叠，但不能直接
处理极端正负不平衡、困难背景和质心 F1 的阈值校准。

### 6.2 统一 segmentation_losses

`networks/losses/segmentation_losses.py` 后续统一支持：

```text
soft_iou, frame_soft_iou, bce, focal, dice, bce_dice,
tversky, focal_tversky, lovasz, sls_iou, tda_sls,
hard_focal, tversky_hard_focal, stc_f1, f1_calibrated_ohem
```

这一步把所有 loss 统一为接收 `[B,T,H,W]` logits/target，并加入形状、dtype 和参数
校验。部分探索性 loss（TDA-SLS、STC-F1）较慢或没有形成稳定正收益，不是当前
推荐配置。

### 6.3 Tversky + Hard Focal 阶段

Tversky 可以单独调节 FP/FN 权重；Hard Focal 只对正样本和 Top-K 困难背景优化，
避免海量容易负样本淹没目标。BRTD2 阶段曾使用该组合，但损失与结构同时变化时
难以归因，最终转向更直接面向 F1 和校准的 OHEM。

### 6.4 当前损失：F1CalibratedOHEMLoss

当前最终实验使用：

```text
loss = Tversky
     + 0.15 × Dice
     + current_hard_weight × HardMarginOHEM
```

默认参数：

| 参数 | 值 | 作用 |
|---|---:|---|
| fp_weight | 0.6 | Tversky 与 margin 中提高困难负样本权重 |
| fn_weight | 0.4 | 正样本权重 |
| dice_weight | 0.15 | 提供对称重叠约束 |
| hard_weight | 0.10 | OHEM 最大权重 |
| negative_ratio | 4.0 | 困难负样本数约为正样本数 4 倍 |
| min_negatives | 256 | 每 clip 至少困难负样本数 |
| max_negatives | 4096 | 每 clip 最多困难负样本数 |
| margin | 1.0 | 正负 logit 间隔 |
| warmup_epochs | 5 | 前 5 epoch 不启用 hard margin |
| ramp_epochs | 10 | 之后 10 epoch 线性增至 0.10 |

困难 margin：

```text
positive loss = softplus(margin - positive_logit)
negative loss = TopK[softplus(margin + negative_logit)]
```

有效帧 mask 同时作用于 Tversky、Dice 和 OHEM。当前代码规定
`--mask_padded_frames 1` 必须搭配 `f1_calibrated_ohem`，防止其他 loss 静默忽略
padding mask。

### 6.5 损失变化的实际效果与结论

- F1-OHEM 是当前整套实验中最可靠、使用最广的损失；
- 早期 DeepPro 约 epoch45 网站优于继续训练到 100 epoch，说明 loss 并不能替代
  checkpoint/阈值选择；
- 当前网络结构对照全部固定 F1-OHEM，结构结论没有被 loss 同时变化污染；
- 现有瓶颈仍是 Recall，后续若改 loss，应先做困难正样本/中心热图分支，而不是继续
  只增加负样本惩罚。

---

## 7. 训练工程和稳定性变化

### 7.1 有效帧 mask

历史统计显示 44.1% 候选窗口不足 40 帧，约 29.7% 的实际采样含 padding。后来
数据加载器返回有效帧信息，训练 loss 和 raw-APMD adapter 都对 padding 屏蔽。

自检验证：

- 同一有效 clip 放在张量前半或后半时，输出误差低于数值容差；
- padding 区 residual response 严格为 0；
- 一、二阶尺度权重在有效边界仍归一化。

### 7.2 分层学习率

最终实验：

```text
adapter LR  = 0.001
backbone LR = 0.001 × base_lr_mult(5.0) = 0.005
optimizer   = Adam
weight decay= 0.0001
```

这是基于 BRTD2 学习率实验保留的设置。虽然通常预训练 backbone 用更小 LR，但在
该小模型和数据上，更高 backbone LR 的单轮趋势更好。不要在比较新结构时同时
改变这两个 LR。

### 7.3 预训练和 scratch 审计

`train.py` 的行为：

- 指定 `--base_ckpt` 且不是 resume：加载基础权重，`strict=False`，只允许新增
  `brtd.*` 键缺失；
- 不指定 `--base_ckpt`：完整 PyTorch 随机初始化，并明确写日志；
- resume checkpoint 优先于 base checkpoint；
- `--resume never` 拒绝覆盖非空实验目录；
- `--resume auto` 优先加载 `latest_model.pth`，其次 `best_model.pth`。

### 7.4 早停

最终实验显式启用：

```text
metric      = eval_f1
patience    = 30
min_delta   = 0.0001
start_epoch = 15
```

早停状态随 checkpoint 保存，resume 会恢复 best value、best epoch 和 bad epoch
计数。最终 8 个实验中，部分预训练结构在 55 epoch 早停，scratch 多数跑满 100。
早停 epoch 不等于最终质心最佳 checkpoint。

### 7.5 推理分块

`eval_chunk_rows=64` 只在 eval 时按行分块执行 TPro，减小显存。TPro 不做空间混合，
因此在 `multiscale_head` 场景会先重组 TPro 结果再执行空间头，避免分块接缝。
严格预训练恒等自检时基线和 BRTD3 必须采用相同分块路径，否则会出现约 1e-5 的
浮点累加顺序差异。

---

## 8. 历史效果汇总

### 8.1 早期网站和融合结果

| 方案 | 网站总分/检测近似 | 结论 |
|---|---:|---|
| DeepPro-Plus + F1-OHEM，约 epoch45 | 84.91 / 0.7491 | 强单模型基线 |
| DeepPro-Plus + F1-OHEM，100 epoch | 84.13 / 0.7413 | 训练过久退化 |
| BRTD，50 epoch | 82.35 / 0.7235 | 单模型召回低 |
| BRTD，100 epoch | 83.23 / 0.7323 | 延长有改善但仍弱 |
| BRTD2，约 epoch45 | 80.89 / 0.7089 | 未成功 |
| DeepPro epoch45 + BRTD 融合 | **85.59 / 0.7559** | 证明互补性 |
| DeepPro valid-frame 历史提交 | 85.72 | mask 方向有效 |

### 8.2 Raw-APMD 本地结果

| Seed | Precision | Recall | Proxy F1 |
|---:|---:|---:|---:|
| 46 | 0.920209 | 0.677930 | 0.780705 |
| 47 | 0.935289 | 0.676964 | **0.785431** |
| 49 | 0.934005 | 0.665462 | 0.777190 |
| 三 seed 均值 | — | — | 0.781109 |

### 8.3 RMS/运动/对比度网站结果

下表来自 2026-08-22 网站提交记录；seed 对应按提交清单和上传顺序核对：

| Variant | seed47 | seed49 | 均值 | 范围 |
|---|---:|---:|---:|---:|
| RMS | 87.41 | 87.01 | 87.21 | 0.40 |
| Channel-RMS | **88.33** | 86.19 | 87.26 | **2.14** |
| Motion detrend | 87.18 | 88.09 | **87.64** | 0.91 |
| Multiscale contrast | 87.31 | 87.32 | 87.32 | **0.01** |

结论：Channel-RMS 有高上限但稳定性差；Motion 平均最好；Multiscale 最稳定但单独
上限不高。这些结果促成 Hybrid-RMS 和组合因子实验。

---

## 9. 最终 8 个实验的完整结果

所有实验固定 seed 47、相同 loss/学习率/采样和后处理，唯一变量是结构开关与
初始化。以下是 `selected_submission.json` 的本地质心结果：

| 结构 | 初始化 | 训练结束 | 质心最佳 epoch | 阈值 | 面积 | Precision | Recall | Proxy F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Hybrid-RMS | pretrained | 55 早停 | 24 | 0.10 | 2 | 0.916369 | 0.690557 | 0.787597 |
| Hybrid-RMS | scratch | 100 | 86 | 0.22 | 2 | 0.948090 | 0.654517 | 0.774414 |
| Hybrid-RMS + Motion | pretrained | 55 早停 | 55 | 0.17 | 2 | 0.929883 | 0.685829 | 0.789424 |
| Hybrid-RMS + Motion | scratch | 100 | 72 | 0.10 | 2 | 0.934858 | 0.658154 | 0.772474 |
| Hybrid-RMS + Multiscale | pretrained | 55 早停 | 24 | 0.10 | 2 | 0.923900 | 0.679033 | 0.782763 |
| Hybrid-RMS + Multiscale | scratch | 100 | 86 | 0.13 | 2 | 0.923623 | 0.657119 | 0.767906 |
| Hybrid-RMS + Motion + Multiscale | pretrained | 100 | **75** | 0.17 | 2 | 0.933339 | **0.694785** | **0.796586** |
| Hybrid-RMS + Motion + Multiscale | scratch | 69 早停 | 39 | 0.14 | 2 | 0.931505 | 0.656915 | 0.770476 |

### 9.1 预训练效果

| 结构 | Pretrained - Scratch Proxy F1 |
|---|---:|
| Hybrid-RMS | +0.013183 |
| + Motion | +0.016950 |
| + Multiscale | +0.014857 |
| + Motion + Multiscale | **+0.026110** |

四个结构全部是预训练更好，因此“是否加载预训练权重”已经得到同 seed 的明确答案。

### 9.2 模块主效应和交互

在 pretrained 组中：

- Motion 相对 Hybrid-RMS：`+0.001827`；
- Multiscale 单独：`-0.004834`；
- Motion + Multiscale：`+0.008989`。

在 scratch 组中，三个扩展都没有超过基础 Hybrid-RMS。说明多尺度对比不是独立
稳定增益，但与运动去趋势和预训练存在明显正交互。最合理的解释是：预训练主干先
提供可用的目标特征，motion 清除相干背景后，multiscale 才能强化剩余局部异常；
scratch 早期特征未成形时，多分支反而增加优化难度。

### 9.3 与旧模型比较

最终最佳 `0.796586`：

- 相对 Raw-APMD seed47 `0.785431`：提升 `0.011155`；
- 相对 Raw-APMD seed47/49 均值 `0.781311`：提升约 `0.015275`；
- Recall 从旧 Raw-APMD seed47 的 `0.676964` 提高到 `0.694785`；
- Precision 仍保持 `0.933339`。

这是当前本地证据最强的模型，但它还没有本文记录中的隐藏测试网站分数；迁移后
如果要正式决定最终提交，应先上传该 ZIP 验证本地排序是否延续到隐藏测试集。

---

## 10. 最终实验目录和提交包

实验根目录：

```text
log/sem_seg/2026-08-22/
```

所有 8 个 `_structure_pipeline_status/*.status` 均为 `COMPLETE`。最佳模型目录：

```text
log/sem_seg/2026-08-22/
SatVideoIRSDT_v1__2026-08-22_08-27-32__F1OHEM-
brtd3_raw_apmd_hybrid_rms_motion_detrend_multiscale_contrast_pretrained_seed47_E100/
```

关键文件：

```text
checkpoints/epoch_75_model.pth
postprocess/results/selected_submission.json
postprocess/results/epoch_75_centroid_f1.json
postprocess/probabilities/epoch_75/
submission/submit_brtd3_raw_apmd_hybrid_rms_motion_detrend_multiscale_contrast_pretrained_seed47_best_proxy_f1.zip
submission/submission_validation.txt
```

每个实验目录都保存了训练时的模型源码和 adapter 源码快照。测试旧 checkpoint 时
优先使用快照，而不是假设根目录最新源码一定兼容。

---

## 11. 最终训练配置

| 配置 | 值 |
|---|---|
| Model | `DeepPro-Plus_BRTD3` |
| Dataset | `SatVideoIRSDT_v1` |
| Seed | 47 |
| Deterministic | 0 |
| Sequence length | 40 |
| Patch size | 128 |
| Sample rate | 0.04 |
| Batch size | 20 |
| Epoch | 最多 100 |
| Optimizer | Adam |
| Adapter LR | 0.001 |
| Base LR multiplier | 5.0 |
| Weight decay | 0.0001 |
| LR step / decay | 10 / 0.7 |
| Loss | `f1_calibrated_ohem` |
| Tversky FP/FN | 0.6 / 0.4 |
| Hard negative top-k | 4096 |
| Valid-frame mask | 1 |
| Eval threshold | 0.5（只用于 pixel eval） |
| Eval chunk rows | 64 |
| Train/val workers | 4 / 2 |
| Prefetch factor | 2 |
| Early stopping | eval_f1, patience 30, start 15 |
| Checkpoint candidates | pixel F1 Top-5 |
| Centroid threshold grid | 0.10:0.95:0.01 |
| min_area | 后处理搜索，最终均为 2 |

启动矩阵在：

```text
tools/launch_hybrid_rms_pretrain_ablation_8gpu.sh
tools/run_structure_candidate_experiment.sh
```

注意：两个脚本含旧服务器绝对路径；迁移后必须先改路径再运行。

---

## 12. 自动后处理与提交链路

结构实验 runner 在训练结束/早停后自动执行：

1. 从模型日志解析 pixel F1；
2. 选择 Top-5 已保存 checkpoint；
3. `test.py` 导出各 checkpoint 的验证概率；
4. `centroid_f1_sweep.py` 扫描阈值和 `min_area`；
5. 以 centroid Proxy F1 选最优 checkpoint/阈值；
6. `seg2tracked_centroid_txt.py` 生成带轨迹 ID 的 TXT；
7. `build_single_submission.py` 生成 ZIP；
8. `validate_submission_zip.py` 检查 255 个序列、23,087 帧、坐标范围和尺寸；
9. 生成 SHA-256；
10. 状态文件写 `COMPLETE zip=...`。

关键经验：

- 行列坐标曾经出现过颠倒，已经修复；不要再次交换 x/y；
- 不能把网络 patch 尺寸当作原视频尺寸；
- `min_area=2` 在多数最优结果中稳定，孤立单像素是常见假阳性；
- 质心最佳 checkpoint 可能远离早停 pixel best，例如最终最佳使用 epoch75；
- 轨迹项历史已接近/达到满分，当前主要优化目标是检测 Recall/F1。

---

## 13. 自检和已完成验证

`tools/check_brtd3.py` 已验证最终四结构：

- 旧 DeepPro-Plus state dict 非严格加载只缺 `brtd.*`；
- 零初始化 residual projection 后 logits 与基础模型逐元素一致；
- projection weight 的反向梯度非零；
- padding 位置改变不影响有效片段；
- padding 区 residual 为 0；
- 一阶/二阶时间尺度权重 softmax 和为 1；
- multiscale contrast 权重 softmax 和为 1；
- Hybrid-RMS 初始 channel mix 为 0.5；
- Python 编译、Shell `bash -n`、`git diff --check` 均通过。

新服务器建议执行：

```bash
cd /NEW/PATH/DeepPro-main

python -m py_compile \
  train.py test.py \
  networks/models/DeepPro-Plus_BRTD3.py \
  networks/layers/structure_adapters.py \
  tools/check_brtd3.py

bash -n tools/run_structure_candidate_experiment.sh
bash -n tools/launch_hybrid_rms_pretrain_ablation_8gpu.sh

python tools/check_brtd3.py \
  --variant raw_apmd_hybrid_rms_motion_detrend_multiscale_contrast \
  --device cpu --seqlen 40 --height 32 --width 32
```

如果 CUDA/驱动匹配，再用 `--device cuda:0` 做一次 GPU 自检。

---

## 14. 原服务器环境

### 14.1 硬件

```text
8 × NVIDIA A100-SXM4-80GB
单卡显存：81920 MiB
Driver：550.90.07
```

### 14.2 软件

```text
Python  3.8.5
PyTorch 2.1.2
CUDA runtime 12.1
cuDNN 8902
OpenCV 4.10.0
NumPy 1.23.5
SciPy 1.9.3
```

原 Python：

```text
/home/devbox/project/model/miniconda3/envs/sjyPID/bin/python
```

环境约 8.4 GB，不在项目压缩包内。推荐重建：

```bash
conda env create -f docs/environment_sjyPID_2026-08-24.yml
conda activate sjyPID
```

若跨操作系统/驱动导致 YAML 解析冲突，先创建 Python 3.8 环境，再用
`pip_freeze_sjyPID_2026-08-24.txt` 逐项安装；PyTorch 应优先按新服务器 CUDA 驱动
选择官方匹配版本，不要盲目强装旧 CUDA 构建。

`conda_explicit_sjyPID_2026-08-24.txt` 只适合相同 Linux/架构和相同 channel 可用时
做精确复刻。

---

## 15. 新服务器迁移步骤

### 15.1 复制并校验项目包

源服务器会生成：

```text
DeepPro-main_full_transfer_2026-08-24.tar.gz
DeepPro-main_full_transfer_2026-08-24.tar.gz.sha256
```

新服务器：

```bash
sha256sum -c DeepPro-main_full_transfer_2026-08-24.tar.gz.sha256
tar -xzf DeepPro-main_full_transfer_2026-08-24.tar.gz
cd DeepPro-main
```

压缩包包含整个目录，包括 `.git` 和 `log`。解压预计至少需要 6 GB 空间；训练还
需要额外 checkpoint、概率和临时结果空间，建议项目所在磁盘至少预留 50 GB。

### 15.2 单独迁移数据集

数据集约 50 GB，需要另行复制：

```text
/home/devbox/project/model/sjy/CSIG2026/datasets/SatVideoIRSDT_v1
```

复制后至少确认存在训练/验证/测试序列和标注，并更新所有 `DATA_ROOT` / `--datapath`。
不要仅复制项目压缩包后直接启动训练，否则会报数据集路径不存在。

### 15.3 修改硬编码路径

以下脚本含旧服务器绝对路径：

```text
tools/run_structure_candidate_experiment.sh
tools/run_valid_frame_architecture_experiment.sh
tools/launch_hybrid_rms_pretrain_ablation_8gpu.sh
tools/launch_structure_round2_8gpu.sh
tools/launch_raw_apmd_optimizations_6gpu.sh
tools/launch_raw_apmd_rms_2gpu.sh
tools/launch_raw_apmd_experiment.sh
tools/resume_structure_candidate_postprocess.sh
tools/launch_swanlab_sidecars_8run.sh
```

至少替换：

```text
REPO_ROOT=/NEW/PATH/DeepPro-main
PYTHON_BIN=/NEW/CONDA/envs/sjyPID/bin/python
DATA_ROOT=/NEW/DATA/SatVideoIRSDT_v1
SWANLAB_CREDENTIAL_FILE=/NEW/PRIVATE/PATH/.netrc
```

建议后续把这些路径改为从脚本自身位置和环境变量推导，而不是继续硬编码。

实验 JSON 和历史日志中的旧绝对路径只是历史记录，不需要批量改写；如果读取
`selected_submission.json`，可根据相对实验目录重新定位文件。

### 15.4 预训练权重

基础初始化权重已包含在项目包：

```text
pretrained/SatVideoIRSDT_DeepPro-Plus_pretrained_init.pth
```

文件约 315 KB。迁移后运行：

```bash
sha256sum pretrained/SatVideoIRSDT_DeepPro-Plus_pretrained_init.pth
```

并用 `tools/check_brtd3.py` 验证 state dict 兼容。

### 15.5 SwanLab

SwanLab 项目：

```text
https://swanlab.cn/@SInt123/CSIG2026-DeepPro
```

API key 和 `/home/devbox/project/model/.swanlab/.netrc` **不在项目目录中，也不应放进
压缩包**。在新服务器单独登录/配置凭据。历史运行曾遇到 HTTP(S) 代理连接失败，
侧车启动器后来对 SwanLab 子进程显式取消代理变量并从 epoch 9 重放。

安全注意：旧 API key 曾在一次本地进程检查输出中出现，建议在 SwanLab 控制台
轮换 key，再在新服务器配置新 key。

### 15.6 第一次启动前

```bash
git status --short --branch
nvidia-smi
screen -ls
df -h .
```

先运行 Python/Shell 自检和 `tools/check_brtd3.py`，再用 launcher 的 `--dry-run`
查看 8 卡映射。任何 GPU 显存超过 1024 MiB 时不要强行启动 8 个独立训练。

---

## 16. Screen、恢复和运行规范

长任务必须使用 Screen：

```bash
screen -dmS experiment_name -L -Logfile /ABS/PATH/screen.log \
  bash /ABS/PATH/runner.sh ...

screen -ls
screen -r experiment_name
```

从 Screen detach：`Ctrl+A`，再按 `D`。

判断任务状态不能只看单次 GPU utilization：验证/数据等待阶段可能瞬时为 0%。应同时
检查：

1. `screen -ls`；
2. `ps` 中主训练进程；
3. GPU 显存；
4. Screen 日志在 20~60 秒内是否增长；
5. status 文件是 `RUNNING / FAILED / COMPLETE`；
6. `latest_model.pth` 修改时间。

恢复训练应使用同一实验目录和：

```text
--resume auto
```

不要再次使用 `--resume never` 指向已有目录，也不要新旧两个进程同时写同一
checkpoint。DataLoader worker 会继承 `train.py` 命令行，不要把它们误判为重复训练。

---

## 17. 已知漏洞、风险和下一步建议

### 17.1 仍存在的技术漏洞

1. 最终最好结果只有 seed 47；虽然 pretrained/scratch 是严格同 seed 对照，但最终
   full 模型的跨 seed 稳定性尚未验证；
2. 最终 Hybrid 8 组只记录本地 Proxy F1，尚缺网站隐藏测试分数；
3. Multiscale contrast 权重是每通道全局参数，不随具体帧/场景自适应；
4. Motion detrend 是固定 15×15 均值，只能抑制低频平移类相干背景，不能显式处理
   旋转、仿射、视差或局部非刚性背景运动；
5. BatchNorm 仍存在于基础 stem/head，跨设备或很小 batch 时可能有统计偏移；
6. 最终选择仍受本地质心阈值拟合影响，存在验证集后处理过拟合；
7. pixel F1 Top-5 仍可能漏掉 centroid F1 最优 checkpoint；
8. 脚本硬编码绝对路径和固定历史日期，不适合作为通用部署入口；
9. `deterministic=0` 提高速度但使完全比特级复现不可保证。

### 17.2 推荐下一步顺序

1. 先把最终 full pretrained ZIP 提交网站，确认隐藏测试分数；
2. 若网站排序保持，补一个固定 seed（例如 49）验证 full 模型稳定性；
3. 做 full pretrained 的邻近 checkpoint 概率平均/EMA，不要立即重训大结构；
4. 对 epoch 60~90 更密集做 centroid 扫描，检查 Top-5 是否漏掉更优轮次；
5. 将 15×15 fixed detrend 升级为轻量全局运动估计/对齐，但保持零初始化和单变量
   对照；
6. 把 multiscale 权重改为受限的实例自适应权重，同时防止重新引入压制弱目标的
   Sigmoid gate；
7. 若继续改 loss，优先增加中心热图/困难正样本监督，不要继续只加强负样本 OHEM；
8. 把 launcher 改为环境变量驱动并增加统一 `resume` 启动器。

---

## 18. 参考资料与设计来源

- DeepPro：`paper/Li 等 - 2026 - Probing Deep into Temporal Profile Makes the Infrared Small Target Detector Much Better.pdf`
- DeepPro arXiv：https://arxiv.org/abs/2506.12766
- TDCNet / 时间差分多尺度：https://ojs.aaai.org/index.php/AAAI/article/view/37385
- DMRL / 相干背景运动与局部异常：https://arxiv.org/abs/2606.15286
- IRDINO / 二阶运动与预训练：https://openaccess.thecvf.com/content/CVPR2026F/html/Xu_IRDINO_Adapting_DINOv3_with_Second-Order_Motion_Awareness_for_Moving_Infrared_CVPRF_2026_paper.html
- FRN / 逐通道二阶矩归一化：https://openaccess.thecvf.com/content_CVPR_2020/html/Singh_Filter_Response_Normalization_Layer_Eliminating_Batch_Dependence_in_the_Training_CVPR_2020_paper.html
- ACM / 小目标低层细节与上下文：https://openaccess.thecvf.com/content/WACV2021/html/Dai_Asymmetric_Contextual_Modulation_for_Infrared_Small_Target_Detection_WACV_2021_paper.html

历史说明文档：

```text
docs/CONVERSATION_HANDOFF.md
docs/BRTD_CONVERSATION_HANDOFF_2026-08-12.md
docs/BRTD_CONVERSATION_HANDOFF_2026-08-20.md
docs/brtd2_research.md
docs/structure_round2.md
docs/raw_apmd.md
docs/experiment_analysis_2026-08-20.md
docs/structure_optimization_2026-08-20.md
```

---

## 19. 迁移验收清单

### 文件完整性

- [ ] 压缩包 SHA-256 通过；
- [ ] 能看到 `.git`、`networks`、`tools`、`pretrained`、`log`；
- [ ] 本交接文件存在；
- [ ] 最终 8 个 status 都是 COMPLETE；
- [ ] 最终最佳 epoch75 checkpoint 和 submission ZIP 存在；
- [ ] 三份 Conda/Pip 环境清单存在。

### 数据和环境

- [ ] 50 GB 数据集已另行复制；
- [ ] `DATA_ROOT` 已修改；
- [ ] Conda 环境已重建；
- [ ] PyTorch 能识别新 GPU；
- [ ] 预训练 checkpoint 能加载；
- [ ] API key 已在新服务器单独配置且不提交到 Git。

### 代码自检

- [ ] `python -m py_compile` 通过；
- [ ] launcher/runner `bash -n` 通过；
- [ ] `tools/check_brtd3.py` CPU 通过；
- [ ] CUDA smoke test 通过；
- [ ] 数据加载一个 batch 成功；
- [ ] 用历史 checkpoint 导出少量概率成功；
- [ ] 历史 submission ZIP 校验通过。

### 运行安全

- [ ] 启动前确认 GPU 空闲；
- [ ] 使用 Screen；
- [ ] 新实验目录不与历史目录重名；
- [ ] 恢复任务使用 `--resume auto`；
- [ ] 不执行会清理未提交工作树的 Git 命令。

---

## 20. 最终结论

项目从基础 DeepPro-Plus 的“差分主干 + 每像素 temporal profile”出发，经历了
BRTD1 的激进差分与门控失败、BRTD2 的语义层多尺度残差、BRTD3 的统一零初始化
结构框架，再演化为 Raw-APMD 的原始外观旁路。随后通过 RMS/Channel-RMS 发现了
“绝对外观保留”和“逐通道噪声放大”的矛盾，最终用 Hybrid-RMS 可学习收缩解决；
再用 motion detrend 抑制相干背景运动，用 3/5/7 multiscale contrast 覆盖几像素
目标尺度。

最终 8 卡因子实验最重要的两条证据是：

1. 四个结构中 pretrained 全部显著优于 scratch；
2. motion 与 multiscale 单独收益有限，但在 pretrained full 组合中形成最强正交互，
   达到当前最佳 Proxy F1 `0.796586`。

接手人应把完整预训练 full 模型作为当前主线，把 scratch 组和单模块组作为消融
证据保留。迁移后的首要工作不是立刻重训，而是完成路径/环境自检、提交当前最佳
ZIP 验证网站分数，再决定是否补 seed 或继续结构优化。
