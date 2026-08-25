# Scratch-only 模型改进与并行实验记录（2026-08-25）

## 结论先行

本轮不再叠加已经在 scratch 条件下失败的 motion detrend 或 multiscale
contrast，而是以网站 scratch 最优的 `raw_apmd_hybrid_rms` 为唯一母体，解决两个更
直接的问题：

1. 历史适配器末端投影全零，使第一个反向步骤只有投影层得到梯度，上游外观、运动和
   对比分支在第一步无法学习；
2. 本地 Proxy 指标呈现高精度、低召回，主要矛盾是漏检微弱目标，而不是虚警过多。

据此生成三个轻量候选。为尽快验证最高置信度假设，当前策略改为先将 GPU 0/1/2
组成一个 DDP 任务，只训练候选 1；候选 2、3 保留代码但暂不占用 GPU：

| GPU | 结构变量 | 变体 | 适配器参数量 |
|---:|---|---|---:|
| 0 | 仅将零投影改成 0.05 倍 Kaiming 非零初始化 | `raw_apmd_hybrid_rms_scratch_init` | 2,496 |
| 1 | 候选 1 + 有效帧掩码的 3/9 帧时域带通 | `raw_apmd_hybrid_rms_scratch_bandpass` | 3,072 |
| 2 | 候选 1 + 1×1 主干细节直连融合 | `raw_apmd_hybrid_rms_scratch_detail` | 3,344 |

候选 1 与历史 Hybrid-RMS 对照可归因于初始化；候选 2、3 分别只比候选 1 多一个模块。

## 本地与网站证据

网站八个严格配对结果显示，四种结构在 pretrained 条件下均优于 scratch，平均差值
为 `+1.7275`；但项目约束要求以后全部 scratch-only，因此性能证据与研发约束需要
分开表述。

在 scratch 条件下，以 Hybrid-RMS 的 `86.71` 为基线：

- 加 motion detrend：`86.33`，下降 `0.38`；
- 加 multiscale contrast：`86.05`，下降 `0.66`；
- 两者同时加入：`86.34`，下降 `0.37`。

对应本地 Proxy F1 为 `0.774414 / 0.772474 / 0.767906 / 0.770476`，排序与网站一致。
Hybrid-RMS scratch 的 Proxy precision 为 `0.948090`，recall 仅 `0.654517`，说明改进
应优先恢复弱目标证据。验证标注扫描还显示目标面积中位数仅 6 像素，79.2% 不超过
9 像素，细粒度信息很容易被过强的空间抑制吞没。

## 文献核验与取舍

- DeepPro 将问题重述为逐像素时域轮廓异常检测，并强调全局时域显著性和相关性；这
  支持保留 TPro 主干，并用轻量时域带通补充固定一、二阶差分，而不是替换主干。
  来源：[DeepPro / arXiv 2506.12766](https://arxiv.org/abs/2506.12766)。
- TDCNet 指出纯时域差分空间表征不足，而普通 3D 卷积缺少显式运动感知；本文采用
  低成本融合而非完整复制其双主干和注意力结构。
  来源：[AAAI 2026 TDCNet](https://ojs.aaai.org/index.php/AAAI/article/view/37385)。
- ACM 的 bottom-up point-wise 路径用于交换高层语义与细微低层细节，支持候选 3 的
  无门控 1×1 细节融合。历史 BRTD2 的 sigmoid reliability gate 已在本项目中负收益，
  所以这里保持加法残差而不恢复门控。
  来源：[WACV 2021 ACM](https://openaccess.thecvf.com/content/WACV2021/html/Dai_Asymmetric_Contextual_Modulation_for_Infrared_Small_Target_Detection_WACV_2021_paper.html)。
- FRN 证明按样本、按通道的非中心二阶矩归一化适合小 batch；当前 Hybrid-RMS 已采用
  该类能量归一化思想。由于本地 Channel-RMS 跨 seed 波动明显，本轮不再整体替换
  Hybrid-RMS。
  来源：[CVPR 2020 FRN](https://openaccess.thecvf.com/content_CVPR_2020/html/Singh_Filter_Response_Normalization_Layer_Eliminating_Batch_Dependence_in_the_Training_CVPR_2020_paper.html)。
- Fixup 说明残差网络的初始化尺度会显著影响 scratch 优化稳定性。论文并不直接证明
  “零投影一定更差”；本项目的判断来自本地计算图：当整个适配器只通过全零投影接入
  主干时，第一步上游梯度严格为零。因此采用“小幅非零”而非普通尺度初始化。
  来源：[ICLR 2019 Fixup](https://openreview.net/pdf?id=H1gsz30cKX)。
- SLS 强调小目标尺度和位置敏感损失，但位置项可能改变精度/召回权衡；当前本地问题
  已是低召回且训练使用 F1-calibrated OHEM，本轮为保持结构归因不同时更改损失。
  来源：[CVPR 2024 SLS](https://openaccess.thecvf.com/content/CVPR2024/html/Liu_Infrared_Small_Target_Detection_with_Scale_and_Location_Sensitivity_CVPR_2024_paper.html)。

RFR 的形变对齐和循环细化适合卫星视频长程运动建模，但其内存和变量数量均明显更高，
在当前 24GB 单卡推理约束下暂不作为首轮候选。
来源：[RFR / arXiv 2409.12448](https://arxiv.org/abs/2409.12448)。

## 实现与安全约束

- 三个候选均从随机权重开始；训练入口仍会拒绝 `base_ckpt`、`spatial_ckpt` 和
  `st_ckpt`。
- 只允许物理 GPU `0,1,2`，GPU3 被运行时白名单阻止。
- 时域带通的平均值按有效帧计数，零 padding 不参与均值，也不会产生伪运动。
- 末端投影初始化为 Kaiming 权重乘 `0.05`，既让上游从第一步获得梯度，又使初始
  残差相对主干保持很小。
- 训练和验证使用现有 40 帧、128×128 patch 配置；验证启用 AMP，完整推理沿用
  `eval_chunk_rows=32`，适配 RTX 3090 24GB。

## 验收结果

三个候选均通过：

- Python 编译与 shell 语法检查；
- CPU 与 CUDA 前向/反向；
- CUDA 实际训练尺寸 `T=40, H=W=128`；
- 有效片段位于序列首尾时结果一致；
- padding 区残差严格为 0；
- 第一个反向步骤投影层及上游适配器梯度均非零；
- 输出尺寸和原模型接口不变。

## 正在运行的实验

- 活跃批次：`2026-08-25_09-30-18`
- 结构：`raw_apmd_hybrid_rms_scratch_init`
- GPU：`0,1,2`，DDP world size 3；GPU3 未使用
- SwanLab project：`CSIG2026-DeepPro`
- SwanLab group：`priority_scratch_ddp3_seed47_2026-08-25_09-30-18`
- [SwanLab 运行页面](https://swanlab.cn/@SInt123/CSIG2026-DeepPro/runs/7gmpov6o8108a87j92omw)
- seed：47
- 训练：100 epoch，Adam，adapter LR `0.001`，backbone LR `0.005`；全局 batch
  `18`，每卡 batch `6`，梯度累积 `1`
- 加速策略：epoch 1–5 完成后从同一 scratch checkpoint 续跑；完整验证改为每 5
  个 epoch 一次，并把 255 个验证序列分片到三个 DDP rank 并行处理。未减少训练样本、
  epoch、损失或模型容量。
- 损失：`f1_calibrated_ohem`
- 数据：3,850 个训练采样，255 个验证序列

训练结束后会自动执行：候选 checkpoint 选择、AMP/chunk 完整验证导出、质心 F1
阈值与最小面积扫描、跟踪文本生成、提交 ZIP 校验和 SHA256 记录。状态文件位于
`log/sem_seg/2026-08-25/_structure_pipeline_status/`。

运行审计：首次批次 `2026-08-25_08-44-26` 沿用迁移前的物理 batch 20，三个单卡
任务均在首个 batch OOM；第二批次 `2026-08-25_08-57-26` 以单卡 batch 10 稳定运行，
随后按用户要求停止并标记 `CANCELLED`，未完成 epoch、未产生有效 checkpoint。当前
三卡 DDP 批次从随机权重重新开始，不继承上述任务的任何权重。

加速审计：初始 DDP 实测每轮训练约 5 分钟、单卡完整验证约 9–10 分钟，验证占总耗时
约三分之二且 GPU1/2 空闲。epoch 5 完整 checkpoint 保存后，任务以同一 optimizer、
模型和早停状态从 epoch 6 恢复；SwanLab 继续使用同一 run ID。预计剩余训练与定期
验证约 8–10 小时，训练结束后的候选复评与 ZIP 流程保持不变。

## 局限与下一步判据

网站分数目前每个新候选尚无结果，不能把结构动机写成性能提升结论。历史网站八次
实验也主要是单 seed 证据。首轮选择规则应为：先比较同 seed 本地 Proxy F1 与 recall，
只有显著优于历史 `0.774414` 且 precision 没有不可接受下降的候选才提交网站；胜者再补
seed 49 复验。若三个候选都不胜，下一步应优先做学习率/训练时长控制实验，而不是把
已负收益的去趋势和多尺度空间对比重新叠加。

## AI 使用披露

本轮文献检索、实验数据归纳、候选设计、代码修改和验收由 AI 辅助完成；所有性能数字
均来自项目本地日志或用户提供的网站截图，文献主张链接到论文原始页面。新候选性能需
以本轮实际日志和网站提交结果为准。
