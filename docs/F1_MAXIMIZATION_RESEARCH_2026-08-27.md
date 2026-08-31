# SatVideoIRSDT_v1 F1 最大化：跨领域证据与 PointCenter 决策（2026-08-27）

## 结论先行

当前最值得优先验证的方案不是继续堆叠 FeedbackSTS 形变对齐，而是以网站已经验证的
scratch Hybrid-RMS 为母体，加入三个与当前误差直接对应的机制：

1. **逐连通目标中心热图**：直接监督每一个目标中心，而不是用整帧掩码的全局中心；
2. **全分辨率局部—时序门控恢复块**：在不下采样目标的情况下扩大局部/时序信息交换；
3. **背景调制过滤与过滤前后信息一致性**：降低虚警，同时用很弱的一致性项防止过滤掉
   只有数个像素的小目标。

实现名为 `DeepPro-Plus_BRTD3_PointCenter`，训练损失为
`center_consistency_f1`。它没有外部 teacher，也不加载任何预训练权重。所谓一致性是同一
网络、同一次前向中“过滤前预测”和“过滤后最终预测”的约束；过滤前头同时接受 Dice
深监督，KL 中只把它视为停止梯度的参考分布。

这仍然只是**高优先级候选**。有效改进的最终证据必须是 scratch 网站分数超过
`86.71`，不能由论文机制、参数量、训练验证 F1 或本地 Proxy F1 单独替代。

## 已有证据与误差结构

| Scratch 实验 | 网站分数 | Proxy F1 | Precision | Recall | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| Hybrid-RMS 基线 | **86.71** | **0.774414** | 0.948090 | 0.654517 | 3,153 | 30,397 |
| scratch-init | 86.45 | 0.771051 | 0.954120 | 0.646924 | - | - |
| scratch-bandpass | 86.47 | 0.772087 | 0.943955 | 0.653164 | 3,412 | 30,516 |

基线 Proxy 的 FN 是 FP 的 `9.64` 倍，但它的 Precision 已达 `0.9481`。因此目标不是
无条件提高 Recall，而是增加真实目标中心响应，同时继续限制新增 FP。此前将 Tversky
改成强 Recall 倾向的 FeedbackSTS run 在同 epoch 下出现“Recall 上升、Precision 大幅
下降、F1 反而下降”，构成反证；本轮恢复网站最佳基线的 `FP/FN=0.6/0.4`，把 Recall
改进交给逐目标中心监督，而不是继续降低负样本约束。

基线数据来自：
`log/sem_seg/2026-08-22/...hybrid_rms_scratch_seed47_E100/postprocess/results/selected_submission.json`。
其中 epoch 86、threshold 0.22、min area 2 得到 TP/FP/FN=
`57587/3153/30397`。

## 数据集结构决定了哪些损失可用

数据根目录：

```text
/home/user/4T_Storage/SJY/CSIG2026/datasets/SatVideoIRSDT_v1
```

对训练 mask 每 20 帧抽样一次，共检查 `4,990` 帧，并采用 8 连通域统计：

| 统计量 | 结果 |
|---|---:|
| 空帧 | 1.784% |
| 恰好一个目标 | 17.776% |
| 多目标帧 | **80.441%** |
| 单帧最多连通目标 | 10 |
| 连通域面积 p50 / p90 / p99 | **6 / 13 / 30 px** |
| 最大面积 | 159 px |

由此得到两个强约束：

- 约 80% 是多目标帧，所以原 SLS 风格“整帧预测中心对整帧 GT 中心”的位置项会把多个
  目标压成一个位于它们之间的虚构中心，不能直接使用；必须逐连通域生成中心峰，或使用
  一对一集合匹配。
- 目标面积中位数只有 6 像素，深层下采样后再上采样很容易彻底丢失目标；首个候选必须
  保留全分辨率表示。

复核命令的逻辑为：按排序后的训练 mask 每 20 个取一个，二值化后执行
`cv2.connectedComponentsWithStats(..., connectivity=8)`，记录每帧组件数及 `CC_STAT_AREA`。
这是一项本地数据诊断，不是论文结论。

## 跨领域证据矩阵

检索日期为 2026-08-27。纳入标准：论文原文或官方实现；机制能迁移到稀疏微小目标的
特征提取、跨帧融合或中心定位；能完全随机初始化。排除标准：核心能力依赖基础模型或
预训练 backbone；只报告与本任务无关的生成质量指标；会破坏全分辨率微小目标。

| 来源领域与原始资料 | 可迁移机制 | 对本任务的判断 | 证据级别 |
|---|---|---|---|
| 目标检测：[CenterNet / Objects as Points](https://arxiv.org/abs/1904.07850) | 每个对象生成中心高斯热图，用 focal 训练峰值 | 与最终质心匹配直接对齐；本轮采用逐组件中心头 | B：原始论文 |
| 人群点定位：[P2PNet, ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/html/Song_Rethinking_Counting_and_Localization_in_Crowds_A_Purely_Point-Based_Framework_ICCV_2021_paper.html) | 点集合与 GT 一对一 Hungarian 匹配 | 指标最直接，但 scratch 下集合查询优化风险较高，列为第二候选 | A：同行评审原文 |
| 姿态估计：[HRNet, CVPR 2019](https://openaccess.thecvf.com/content_CVPR_2019/html/Sun_Deep_High-Resolution_Representation_Learning_for_Human_Pose_Estimation_CVPR_2019_paper.html) | 始终保留高分辨率分支并反复融合 | p50=6 px 时高度相关；本轮先采用不下采样恢复块 | A |
| 图像恢复：[NAFNet](https://arxiv.org/abs/2204.04676) | 乘法 SimpleGate 与残差缩放 | 适合 scratch，迁移为轻量 3D 局部—时序门控块 | B |
| 图像恢复：[Restormer, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Zamir_Restormer_Efficient_Transformer_for_High-Resolution_Image_Restoration_CVPR_2022_paper.html) | 通道注意力和门控深度卷积 FFN | 证明局部卷积与全局/通道交互的价值；完整 Transformer scratch 风险高，未照搬 | A |
| 多帧红外：[MIST 官方实现, TIP 2026](https://github.com/ShuCvlab/MIST) | 多邻域运动补偿、调制过滤、渐进信息保留 | 采用背景调制过滤思想；完整 SNCB 依赖本环境没有的 NATTEN，且官方 sufficiency 实现的参考头缺少直接监督，因此未照搬 | 官方源码/论文仓库 |
| 视频恢复：[BasicVSR++, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Chan_BasicVSR_Improving_Video_Super-Resolution_With_Enhanced_Propagation_and_Alignment_CVPR_2022_paper.html) | 二阶双向传播、光流引导形变对齐 | 有理论匹配，但本地 FeedbackSTS 对齐实验已出现严重 Precision 损失，暂不作为首轮 | A |
| Burst 恢复：[BIPNet, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Dudhane_Burst_Image_Restoration_and_Enhancement_CVPR_2022_paper.html) | 对齐后交换互补帧信息、渐进融合 | 支持“先保细节再融合”，但参考帧重建目标与逐帧检测不同 | A |
| Burst 恢复：[Burstormer, CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Dudhane_Burstormer_Burst_Image_Restoration_and_Enhancement_Transformer_CVPR_2023_paper.html) | 多尺度形变对齐、参考帧富集、渐进融合 | 作为后续大模型候选；当前没有理由再次优先押注形变对齐 | A |
| 遥感去云：[UnCRtainTS, CVPRW 2023](https://openaccess.thecvf.com/content/CVPR2023W/EarthVision/html/Ebel_UnCRtainTS_Uncertainty_Quantification_for_Cloud_Removal_in_Optical_Satellite_Time_CVPRW_2023_paper.html) | 分辨率保持的空间块、跨时相注意聚合、不确定性 | 支持保留空间分辨率与选择性时序融合；云不确定性回归不直接迁移 | A |
| 遥感变化检测：[BIT](https://arxiv.org/abs/2103.00208) | 少量语义 token 建模双时相上下文，再反馈像素空间 | 长程上下文有价值，但目标过小、训练集有限，完整 token 化可能丢点 | B |
| 异常检测：[DRAEM, ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/html/Zavrtanik_DRAEM_-_A_Discriminatively_Trained_Reconstruction_Embedding_for_Surface_Anomaly_ICCV_2021_paper.html) | 正常背景重建与判别嵌入联合优化 | 可用于第二阶段背景模型；卫星运动会把配准误差当异常，首轮风险高 | A |
| 红外检测：[MSHNet/SLS, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Liu_Infrared_Small_Target_Detection_with_Scale_and_Location_Sensitivity_CVPR_2024_paper.html) | 多尺度头与尺度/位置敏感损失 | 尺度监督有价值；原位置项是整幅掩码中心，不适合本数据的多目标帧 | A |
| 卫星视频：[RFR](https://arxiv.org/abs/2409.12448) | 金字塔形变对齐、循环细化、时空频率调制 | 与任务直接相关，但再次引入对齐的边际证据弱于逐中心监督 | B |
| 医学小病灶：[Blob loss](https://arxiv.org/abs/2205.08209) | 对每个 GT 连通实例分别计算并平均分割损失，避免大实例主导梯度 | 与 p50=6 px、多目标帧直接匹配；若 PointCenter 后期 Recall 停滞，优先作为下一损失候选 | B：IPMI 2023 原文 |
| 医学分割：[ICI loss, MIDL 2023](https://github.com/BrainImageAnalysis/ICI-loss) | instance-wise 分割与归一化 center-of-instance 联合监督 | 官方实现和当前逐组件中心头高度互补；但预测连通域分析会增加训练开销 | 官方实现/同行评审论文 |
| 点监督计数：[Where are the Blobs, ECCV 2018](https://openaccess.thecvf.com/content_ECCV_2018/html/Issam_Hadj_Laradji_Where_are_the_ECCV_2018_paper.html) | 约束每个对象一个 blob，并分别惩罚合并 blob 和无 GT 点的假 blob | 与提交端“一连通域一个质心”最直接；适合在错误分析确认合并/重复峰后启用 | A |
| 人群计数：[DM-Count, NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/hash/118bd558033a1016fcc82560c65cca5f-Abstract.html) | 用最优传输匹配预测密度与点分布，避免固定高斯平滑 | 可替代中心高斯 focal，直接约束多点空间分布；实现和优化风险高于 Blob loss | A |

### 明确没有作为首轮的热门方向

- Mask2Former/P2P-DETR 一类 set prediction 与点级指标契合，但通常需要更长 scratch
  收敛周期和更复杂的匹配稳定化；当前先用 dense center heatmap 获得同样的逐对象归纳
  偏置。[Mask2Former 原文](https://openaccess.thecvf.com/content/CVPR2022/html/Cheng_Masked-Attention_Mask_Transformer_for_Universal_Image_Segmentation_CVPR_2022_paper.html)
- 生成式重建或扩散模型可能提供背景先验，但“重建视觉质量提高”不等于质心 Precision/
  Recall 提高；在没有独立异常残差证据前不让它取代监督检测主干。
- 完整 BasicVSR++、Burstormer 或 RFR 的主要变量仍是对齐。当前同类 FeedbackSTS 的本地
  反证要求我们先测试不同机制，而不是因论文结果更强就重复增加对齐复杂度。
- 任何依赖 ImageNet、DINO、冻结分支或外部 checkpoint 的实现均排除，即使历史网站
  分数更高。

## PointCenter 网络

```mermaid
flowchart LR
    X[40帧 Bx1xTxHxW] --> S[SDifferenceConv + 两个 STD ResBlock]
    S --> R[Hybrid-RMS Raw-APMD<br/>原始外观 + 一/二阶运动]
    R --> T[TPro 40帧投影]
    T --> P[32→16ch 全分辨率投影]
    P --> G1[NAF式 3D 门控恢复块 ×2]
    G1 --> V[过滤前 pre head]
    G1 --> M[通道/空间调制 + 过滤块 ×2]
    M --> H1[mask head]
    M --> H2[逐目标 center head]
    H1 --> F[mask logits + 0.25 × center logits]
    H2 --> F
    V -. 0.01 信息一致性 .-> F
    F --> Y[40帧最终 logits]
```

关键实现：

- `networks/models/DeepPro-Plus_BRTD3_PointCenter.py`
- `data_utils/TrainDataLoader.py`：裁剪与几何增强完成后，对每帧 8 连通域分别生成
  `sigma=1.25` 的高斯中心峰；一个连通目标恰好有一个值为 1 的峰。
- `networks/losses/segmentation_losses.py`：`center_consistency_f1`。

CenterNet 风格中心头偏置初始化为 `-2.19`，初始正类概率约 `0.10`，避免全分辨率负点
在训练初期淹没稀疏正点。最终融合仍保留 mask 形状，提交端继续使用统一连通域质心提取
和轨迹后处理。

2026-08-28 的真实 DDP 稳定性检查发现，四个乘法门控块使用 `beta/gamma=0.1` 时，第二个
过滤块会在个别 clip 的单个时空位置发生 FP16 溢出。最终实现按 NAF 式残差块改为
`beta/gamma=0` 的恒等初始化，并让恢复过滤分支使用 BF16；主干仍为 FP16，所有损失
归约保持 FP32。首次 epoch 2/4 全分辨率验证进一步发现过滤器之前的两个恢复块仍可能
产生极少量 FP16 溢出，因此代码把这两个块也纳入 BF16 边界，并从有限的 epoch 4
scratch checkpoint 恢复。epoch 5 全量验证损失恢复为有限值 `0.598160`，证明修正已
生效。BF16 与 FP16 同为两字节激活，但指数范围足以避免门控乘法
溢出，因此没有用全分辨率 FP32 恢复分支增加 `test.py` 显存。训练循环还会跨三个 rank
同步检查非有限 loss，异常时立即失败并输出各头的有限元素统计。

## 当前损失

总损失：

```text
L = L_tversky
  + 0.15 * L_dice
  + ramp(epoch) * 0.10 * L_hard_margin
  + 0.05 * L_component_center_focal
  + 0.01 * 0.5 * (L_pre_dice + L_pre_post_KL)
```

- `Tversky FP/FN=0.6/0.4`、OHEM ratio/min/max=`4/256/4096` 完全沿用网站最佳 scratch
  基线，避免再次因过度追 Recall 导致 Precision 崩塌。
- `component_center_focal` 对每个连通目标的高斯中心监督；这是与质心 F1 最直接的新项。
- `pre_dice` 使过滤前参考头本身学习目标，而不是随机参考。
- `pre_post_KL` 仅约束过滤后空间分布不要丢失过滤前已学到的信息；不存在第二模型、
  EMA teacher 或预训练 teacher。
- 验证/推理不需要中心标签；辅助项只在训练时启用。

## 验证结果与正式配置

已通过：

- Python 编译、shell `bash -n`、`git diff --check`；
- 合成双目标热图：两个组件得到两个中心峰；
- 真实 SatVideoIRSDT_v1 样本：40 帧 crop 中 `102` 个组件对应 `102` 个峰；
- CPU `B=1,T=40,16×16` 前向、损失、反向；pre/center/mask/restoration/Hybrid-RMS
  分支梯度有限；零初始化门控块在初始状态严格为恒等映射且 `beta/gamma` 梯度有限；
- 饱和 FP16 辅助 logits（`±20`）下，中心 focal、KL、总损失及梯度均为有限值；
- 单卡正式 batch 7 的混合 FP16/BF16 连续 5 次更新，loss 从 `3.230710` 降到
  `2.645057`，GradScaler 保持 `1024`，峰值显存 `19.269 GiB`；
- 三卡真实数据第 1 epoch 的 183 step 全部有限，mean loss `1.033792`，GradScaler
  仍为 `1024`；Tversky/Dice/center focal/pre-Dice/KL 分别为
  `0.745723/0.769528/3.244569/0.841787/1.240436`。
- epoch 2 训练 mean loss `0.518721`、训练 pixel F1 `0.453926`；首次验证得到
  P/R/F1=`0.439074/0.209471/0.283630`。相同 scratch-init HRMS 的 epoch 2 验证 F1
  为 `0.005731`，说明当前候选早期收敛显著更快，但尚不能外推为最终 Proxy/网站提升。
  该轮 `Eval mean loss` 因上述全分辨率 FP16 溢出为 NaN，有限的阈值化 P/R/F1 与
  checkpoint 选优未受影响；随后已由 epoch 5 的扩展 BF16 全量验证完成复验。
- epoch 4 训练 P/R/F1=`0.730264/0.383649/0.503028`，旧精度边界下验证
  P/R/F1=`0.654075/0.276139/0.388332`，但验证损失仍为 NaN；这是执行可恢复重启的
  直接证据。
- epoch 5 使用扩展 BF16 恢复分支从同一 scratch 轨迹续训，训练
  P/R/F1=`0.745844/0.385507/0.508291`；255 个序列的验证损失为有限值 `0.598160`，
  验证 P/R/F1=`0.757001/0.298064/0.427717`。验证 fail-fast 未触发，latest、epoch 5、
  best checkpoint 均成功保存。

截至 epoch 100，与网站 86.71 对应的原 Hybrid-RMS scratch run 使用相同验证阈值 0.5
进行同 epoch 比较：

| Epoch | PointCenter pixel F1 | Hybrid-RMS scratch pixel F1 | 差值 |
|---:|---:|---:|---:|
| 20 | 0.501711 | 0.488467 | +0.013244 |
| 30 | 0.503191 | 0.510571 | -0.007380 |
| 40 | 0.497786 | 0.508286 | -0.010500 |
| 50 | 0.512574 | 0.496691 | +0.015883 |
| 60 | **0.520415** | 0.509729 | **+0.010686** |
| 65 | 0.516743 | 0.511622 | +0.005121 |
| 70 | 0.518505 | 0.516374 | +0.002131 |
| 75 | 0.514074 | 0.517475 | -0.003401 |
| 80 | 0.509955 | 0.509633 | +0.000322 |
| 85 | 0.511719 | 0.510194 | +0.001525 |
| 90 | **0.535337** | 0.515696 | **+0.019641** |
| 95 | 0.524629 | 0.518108 | +0.006521 |
| 100 | 0.524148 | 0.516158 | +0.007990 |

PointCenter 最佳更新为 epoch 90：P/R/F1=`0.755385/0.414570/0.535337`。相对原基线
同 epoch 的 F1 提高 `0.019641`；相对原基线全程最佳 epoch 86 的 `0.526607` 仍提高
`0.008730`。增益主要来自 Recall，但 Precision 相对基线 epoch 86 的 `0.781793`
下降 `0.026408`。epoch 95/100 仍比同 epoch 基线高 `0.006521/0.007990`，说明后段
并非只有单点尖峰；但该结果仍只支持完成 Top-3 centroid Proxy 扫描，不能替代网站验证。

正式启动器：`tools/run_pointcenter_f1_experiment.sh`。

| 项目 | 配置 |
|---|---|
| GPU | 物理 `0,1,2`，一个网络、三进程 DDP |
| 初始化 | `base_ckpt/spatial_ckpt/st_ckpt` 全空；日志再次审计 random init |
| 序列 / patch | `40 / 128` |
| batch | global `21`，每卡 `7`；接近历史最佳 global 20 并留显存余量 |
| 优化 | Adam；主干保持历史有效 LR `0.005`，BRTD LR `0.001`；主干 FP16、恢复过滤分支 BF16、loss FP32；GradScaler 初值 `1024` |
| 加载 | 总 worker `12`，DDP 每 rank `4`；相对总 worker 6 的早期同窗口吞吐约提高 15% |
| 训练 | 100 epoch；早期每 2 epoch 验证，完成数值验收后每 5 epoch 三卡分片验证；不提前停止 |
| 记录 | SwanLab cloud，含总损失、各损失分量、P/R/F1 |
| 交付 | Top-3 checkpoint 概率导出、阈值/面积扫描、轨迹化、ZIP 校验、SHA256 |

## 候选队列与证伪规则

1. **PointCenter（当前）**：若 Proxy F1 超过 `0.774414`，生成 ZIP 并提交网站；网站
   超过 `86.71` 才升级基线。
2. **逐实例等权分割**：若 PointCenter Precision 保持较高但 Recall/Proxy FN 停滞，先把
   Blob/ICI 的 per-component Dice/Tversky 作为独立 loss 候选；它复用现有 GT 连通域，
   比重建新网络或 Hungarian 集合优化更低风险。当前中心 focal 已让每个中心峰等权，
   因此该候选只补偿 mask 分支仍按像素面积加权的部分，不能预设一定增益。
3. **一 blob 一点约束**：若错误分析出现相邻目标合并或一个目标重复峰，再迁移 LC-FCN
   的 split/false-positive 项；若主要问题是整帧漏计，再测试 DM-Count 式分布/计数约束。
4. **点集合一对一预测**：只有 dense center 仍产生无法由 blob 约束解决的重复/漏配，
   才把 P2PNet 的 Hungarian 一对一匹配迁移为独立候选，不与当前 run 混改。
5. **高分辨率多分支**：若中心头学得稳定但弱目标仍漏检，增加 HRNet 风格 1×/2×/4×
   并行表示；保持逐目标中心损失。
6. **显式背景重建残差**：只有在误检可由背景可预测性分开时才引入 DRAEM/重建耦合；
   必须先验证卫星运动不会制造大面积伪异常。
7. **完整对齐传播**：只有 PointCenter 失败且误差按运动速度显著分层时，才重新考虑
   BasicVSR++/Burstormer/RFR；否则现有 FeedbackSTS 反证优先。

任何一轮都执行同一判据：Proxy 只负责筛选，网站分数负责确认；单 seed 胜出后补 seed
49；只提高 Recall、只提高 pixel F1、只生成 ZIP 或只完成梯度测试都不能称为模型提升。

## 限制与披露

- 跨领域论文主要在各自数据和指标上证明机制有效，不能直接证明 SatVideoIRSDT_v1 F1
  会提升；本文件的排序是基于机制—误差匹配的可证伪推断。
- 每 20 帧抽样统计足以暴露多目标与微小面积结构，但不是全训练集精确普查。
- 网站评分函数不是完全公开；本地 Proxy 与网站方向在最近 scratch 实验上一致，但数值
  不等价。
- 本研究与文档由 AI 辅助完成；实验数值来自仓库日志、JSON、真实数据统计或用户提供的
  网站结果，论文机制通过上述原文/官方仓库核验。
