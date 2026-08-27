# F1 最大化研究与 FeedbackSTS 优先实验（2026-08-27）

## 决策摘要

最新 bandpass 网站分数为 `86.47`（ID `903589`），只比 scratch-init 的 `86.45`
高 `0.02`，仍比当前 scratch 基线 `86.71` 低 `0.24`。本地 Proxy F1 的相对方向一致：
`0.772087 > 0.771051`，但仍低于基线 `0.774414`。因此停止把轻量 Raw-APMD 增量作为
优先路线，首选独立的 `DeepPro-FeedbackSTS` 网络。

该选择不是因为参数更少，而是因为它同时满足四个条件：直接针对卫星红外视频；显式
处理跨帧位移；能利用 13 帧双向语义；原论文在 IRSatVideo 上相对 RFR 报告了更高 F1。
新实现不加载任何预训练权重。

## 当前误差诊断

| 模型 | 网站 | Proxy F1 | Precision | Recall | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| Hybrid-RMS scratch 基线 | 86.71 | 0.774414 | 0.948090 | 0.654517 | 3,153 | 30,397 |
| scratch-init | 86.45 | 0.771051 | 0.954120 | 0.646924 | - | - |
| scratch-bandpass | 86.47 | 0.772087 | 0.943955 | 0.653164 | 3,412 | 30,516 |

bandpass 的 FN 是 FP 的 `8.94` 倍，占两类检测错误的 `89.95%`。训练目标却长期使用
Tversky `FP=0.6 / FN=0.4`，并保留最多 4 倍正样本数的困难负样本，方向上继续强化了
已经很高的 Precision。验证集目标面积中位数仅 6 像素，`79.2%` 不超过 9 像素；
模型还缺少跨位置对齐和多尺度解码。这三点共同解释了低 Recall，而不是单纯训练时长
不足。

## 文献路线比较

| 路线 | 与本任务匹配 | Scratch-only | 主要风险 | 决定 |
|---|---|---|---|---|
| FeedbackSTS-Det | 卫星视频、mask、长时序、形变对齐、双向反馈 | 官方入口可随机初始化 | DCN 计算较重 | **首选** |
| RFR | 卫星视频、PDA、循环细化 | 可随机初始化 | IRSatVideo 报告 F1 低于 FeedbackSTS | 备选机制 |
| MSHNet/SLS | 小目标尺度与位置敏感、多尺度监督 | 可随机初始化 | 主要是单帧检测 | 第二阶段损失/解码器候选 |
| MI-DETR | 原始外观与运动双流 | 可随机初始化 | bbox/DETR 与当前逐帧 mask、质心提交不一致 | 暂缓 |
| TDCNet/MOCID | 强时空表示 | 高分配置依赖冻结或预训练分支 | 违反项目 scratch-only 约束 | 排除首轮 |

关键原始资料：

- [FeedbackSTS-Det, Remote Sensing 2026](https://www.mdpi.com/2072-4292/18/12/2042)
  与[官方实现](https://github.com/IDIP-Lab/FeedbackSTS-Det)；
- [RFR, arXiv 2409.12448](https://arxiv.org/abs/2409.12448) 与
  [官方实现](https://github.com/XinyiYing/RFR)；
- [MSHNet/SLS, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Liu_Infrared_Small_Target_Detection_with_Scale_and_Location_Sensitivity_CVPR_2024_paper.html)
  与[官方实现](https://github.com/ying-fu/MSHNet)；
- [CenterNet: Objects as Points](https://arxiv.org/abs/1904.07850)，作为后续直接中心监督的依据。

## 新网络

```mermaid
flowchart LR
    X[13帧原始序列 Bx1xTxHxW] --> E1[8ch 3D块 + 前向SSM]
    E1 --> P1[空间下采样]
    P1 --> E2[16ch + 前向SSM]
    E2 --> E3[32ch + 前向SSM]
    E3 --> E4[64ch + 前向SSM]
    E4 --> E5[128ch + 前向SSM]
    E5 --> D4[上采样 + skip + 后向SSM]
    D4 --> D3[上采样 + skip + 后向SSM]
    D3 --> D2[上采样 + skip + 后向SSM]
    D2 --> D1[上采样 + skip + 后向SSM]
    D1 --> H[1x1x1 logits头]
    H --> Y[13帧分割logits]
```

每个 SSM 按固定间隔 `T=2` 把帧分组，在组内用两级金字塔形变卷积把当前特征对齐到
传播语义。编码器从过去向未来传播，解码器从未来向过去传播；3D 卷积分支保留局部
时空上下文，残差反馈分支提供长程对齐。五级 U-Net 恢复了旧模型缺少的多尺度语义与
细节融合。实现使用 torchvision 维护的 `DeformConv2d`，并保持全随机初始化。

## 数据与损失改动

训练启用与官方数据策略一致的标签保持变换：水平/垂直翻转、转置和时间反转。损失仍
为可审计的 `f1_calibrated_ohem`，但根据本地错误结构改为：

| 参数 | 旧值 | 新值 | 目的 |
|---|---:|---:|---|
| Tversky FP/FN | 0.60 / 0.40 | 0.35 / 0.65 | 提高漏检代价 |
| hard negative ratio | 4.0 | 1.5 | 减少负样本主导 |
| max/min negatives | 4096 / 256 | 2048 / 128 | 保留校准但降低抑制强度 |
| Dice weight | 0.15 | 0.20 | 强化整体重叠 |
| hard-margin weight | 0.10 | 0.05 | 降低背景排序项权重 |

结构与损失同时改变会降低单变量归因能力，但本轮目标是最大化最终 F1，而不是做机制
消融。若首轮失败，下一轮将固定结构分别恢复旧损失、加入中心高斯辅助头，以定位失败
来自网络还是 Recall 权重。

## 运行与验收

- 物理 GPU：仅 `0,1,2`，一个三进程 DDP 网络；
- 序列/patch：`13 / 128`；全局 batch `6`，每卡 `2`；
- Adam，初始学习率 `5e-4`，最多 100 epoch，5 epoch 一次三卡分片验证；
- SwanLab cloud 全程记录；Top-5 checkpoint 逐个导出、质心阈值扫描、轨迹化；
- 最终 ZIP 必须通过结构/帧数校验并生成 SHA256。

正式 run 于 2026-08-27 10:46（Asia/Shanghai）启动：

- 实验目录：`2026-08-27/SatVideoIRSDT_v1__2026-08-27_02-46-06__FeedbackSTS-F1-feedbacksts_t2_recallaug_ddp3_seed47_E100`；
- [SwanLab run](https://swanlab.cn/@SInt123/CSIG2026-DeepPro/runs/xpep23qnp7y93bnu9pulp)；
- 预期提交包：`submission/submit_feedbacksts_t2_recallaug_ddp3_seed47_best_proxy_f1.zip`。

10:41 的短暂 run `92bnlg69j3m4qcyokd0a3` 只用于启动前复核，在首个 epoch 内主动
终止；它缺少 modulation mask，不进入性能对比或 checkpoint 选择。

已完成的代码验收：Python 与 shell 语法、奇数空间尺寸补边、CPU 前反向、真实
`T=13,128x128` CUDA 前反向、三卡 NCCL/DDP 同步。最终调制形变版本每卡 batch 2
的三卡 DDP 峰值约 `1.59 GiB`；`1024x1024` AMP 全帧推理峰值约
`2.54 GiB`，因此使用全帧验证而不是 384 小块拼接。

## 晋级与证伪标准

1. 训练结束自动选择本地质心 Proxy F1 Top-5；
2. Proxy F1 必须先超过 `0.774414` 才值得网站提交；
3. 网站必须超过 `86.71` 才能替换基线；
4. 若只提升 Recall 但 FP 激增导致 F1 不升，先做阈值/最小面积校准，不把 Recall 单项
   写成模型成功；
5. 单 seed 胜出后补 seed 49；没有网站结果前只称“候选”，不称“有效改进”。
