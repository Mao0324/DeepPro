# DeepPro / Raw-APMD 结构优化分析（2026-08-20）

## 1. 8 月 17 日实验核查

8 月 17 日实际只保留了 Raw-APMD 的 seed 46、47、49 三个正式目录。
三者均已完成训练和后处理，并同时具备：

- `COMPLETE`；
- 状态文件中的 `COMPLETE zip=...`；
- `selected_submission.json`；
- `submission_validation.txt` 中的 `VALID`；
- 255 个序列、23,087 帧；
- ZIP 和匹配的 SHA-256。

2026-08-20 再次执行 `unzip -t` 和 `sha256sum -c`，三项均通过。
因此没有需要恢复的 8 月 17 日训练，也不应重跑 seed 46。

双 seed 47/49 的 Raw-APMD 配对基线为：

| seed | Precision | Recall | Proxy F1 |
|---:|---:|---:|---:|
| 47 | 0.935289 | 0.676964 | 0.785431 |
| 49 | 0.934005 | 0.665462 | 0.777190 |
| 均值 | 0.934647 | 0.671213 | 0.781311 |

## 2. 现有模型的优点

1. DeepPro 的 TPro 把每个像素的 40 帧时间剖面视为一维异常信号，通过多组
   可学习时间相关矩阵建模长时信息。原论文的长度消融表明超过 40 帧后收益
   已很有限，因此当前 `seqlen=40` 合理。
2. SDifference/STD 主干对空间和时间差分敏感，能有效压制大面积背景和杂波，
   是当前高 Precision 的主要来源。
3. Raw-APMD 增加独立的原始帧外观旁路，并在 TPro 前加入一阶、二阶、多时间
   尺度和局部对比度信息，直接针对差分主干遗漏的弱目标；现有配对结果证明
   Recall 有稳定提升。
4. adapter 最后使用零初始化投影，加载 DeepPro-Plus 权重时初始 logits 完全
   一致，新增实验可以做严格的预训练配对。
5. 有效帧掩码、双 seed、早停、SwanLab、Top-5 像素检查点再做质心 F1 搜索，
   以及最终 ZIP 严格验证，已经形成完整而可复现的闭环。

## 3. 主要不足

1. 原始 GroupNorm 会逐帧减均值；当前 RMS 虽不减均值，但把所有通道共同
   除以一个帧级 RMS。少数强响应通道可能压低其他通道中的弱目标响应。
2. 一、二阶运动是在固定像素上计算的，其中既包含稀疏目标异常，也包含相机
   与背景的低频一致运动。后者容易形成稳定假警。
3. 当前局部对比度固定混合 3×3 与 7×7 背景。对不同目标尺寸只有一个固定
   比例，且所有通道共享比例。
4. 外观、运动和对比度直接卷积融合，分支没有单独监督。依赖输入的 Sigmoid
   gate 在既有 BRTD2 中曾压制弱目标，因此本轮仍不重新引入 gate。
5. 训练早停看 pixel F1，而最终选择看质心 F1。Top-5 能降低错过风险，但仍须
   以最终质心代理指标而不是最后 epoch 判断。

## 4. 本地目标尺度证据

对验证集全部 23,087 帧、87,984 个 8 连通目标区域做了只读统计：

| 指标 | 25% | 中位数 | 75% | 95% |
|---|---:|---:|---:|---:|
| 面积（像素） | 4 | 6 | 9 | 18 |
| 宽度（像素） | 2 | 3 | 3 | 5 |
| 高度（像素） | 2 | 3 | 3 | 5 |

79.2% 的目标面积不超过 9 像素，93.7% 不超过 16 像素。这支持保留原分辨率，
并用 3/5/7 的小尺度中心—周边上下文，而不是增加下采样层。

## 5. 三个并行单变量候选

所有候选都以 `raw_apmd_rms` 为共同基线，保持 adapter LR 0.001、
backbone LR 0.005、loss、采样、双 seed 和后处理完全相同。

### 5.1 raw_apmd_channel_rms

只改变 RMS 统计粒度：由每帧跨通道/空间的共同 RMS 改为每帧、每通道独立
空间 RMS。不减均值，不引入 batch 依赖，也避免强通道缩放所有弱通道。
该统计方式受 FRN 的逐样本逐通道二阶矩归一化启发，但保留现有 SiLU。

### 5.2 raw_apmd_motion_detrend

只在多尺度一、二阶运动上下文之后减去 15×15 空间低频分量，再送入原
motion fusion。15×15 明显大于 95% 目标宽高，目标应作为稀疏局部异常保留，
而相机/背景的一致运动会被削弱。该设计对应“背景运动全局一致、目标运动
局部稀疏”的解耦观察。

### 5.3 raw_apmd_multiscale_contrast

把固定的 3×3/7×7 等权中心—周边对比度，改为 3×3、5×5、7×7 三尺度，
并为每个 bottleneck 通道学习 softmax 权重。仅对比度构造变化；其余外观、
运动和投影均不变。尺度覆盖本地目标宽高的主要分布。

## 6. 实验与接受规则

启动器 `tools/launch_raw_apmd_optimizations_6gpu.sh` 固定映射：

| GPU | seed | variant |
|---:|---:|---|
| 2 | 47 | raw_apmd_channel_rms |
| 3 | 49 | raw_apmd_channel_rms |
| 4 | 47 | raw_apmd_motion_detrend |
| 5 | 49 | raw_apmd_motion_detrend |
| 6 | 47 | raw_apmd_multiscale_contrast |
| 7 | 49 | raw_apmd_multiscale_contrast |

每项最多 100 epoch，早停后自动对 pixel F1 Top-5 检查点做阈值和
`min_area` 搜索，生成轨迹 TXT、ZIP、SHA-256、`VALID` 和 `COMPLETE`。

结构接受条件沿用交接规则：双 seed 平均 proxy F1 至少约 0.7843，seed 47
不低于约 0.7824，seed 49 不低于约 0.7742，且 Precision 不明显崩塌。
如果多个候选通过，优先比较双 seed 均值、较差 seed、Precision 稳定性；
模型确定后才做双 seed 概率平均和统一阈值搜索。

## 7. 一手资料

- DeepPro: https://arxiv.org/abs/2506.12766
- TDCNet（AAAI 2026）:
  https://ojs.aaai.org/index.php/AAAI/article/download/37385/41347
- IRDINO（CVPR Findings 2026）:
  https://openaccess.thecvf.com/content/CVPR2026F/html/Xu_IRDINO_Adapting_DINOv3_with_Second-Order_Motion_Awareness_for_Moving_Infrared_CVPRF_2026_paper.html
- Decoupled Motion Representation Learning:
  https://arxiv.org/abs/2606.15286
- Filter Response Normalization（CVPR 2020）:
  https://openaccess.thecvf.com/content_CVPR_2020/html/Singh_Filter_Response_Normalization_Layer_Eliminating_Batch_Dependence_in_the_Training_CVPR_2020_paper.html
- Asymmetric Contextual Modulation（WACV 2021）:
  https://openaccess.thecvf.com/content/WACV2021/html/Dai_Asymmetric_Contextual_Modulation_for_Infrared_Small_Target_Detection_WACV_2021_paper.html
- IRSatVideo-LEO / RFR:
  https://arxiv.org/abs/2409.12448
