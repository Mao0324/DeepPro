# 2026-08-24 网站结果分析与 Scratch-only 决策

> 分析日期：2026-08-25
> 数据来源：比赛网站提交列表截图
> 说明：截图最上方 `apmd_multiscale_contrast_seed49.zip`（87.32）是额外历史对照，
> 不属于本轮 Hybrid-RMS 四结构 × 两初始化的 8 个严格配对实验。

## 1. 八个配对提交

| 结构 | Scratch ID / 分数 | Pretrained ID / 分数 | Pretrained - Scratch |
|---|---:|---:|---:|
| Hybrid-RMS | 898904 / 86.71 | 898903 / 88.07 | +1.36 |
| Hybrid-RMS + Motion detrend | 898893 / 86.33 | 898890 / 87.97 | +1.64 |
| Hybrid-RMS + Multiscale contrast | 898902 / 86.05 | 898900 / 87.52 | +1.47 |
| Hybrid-RMS + Motion + Multiscale | 898889 / 86.34 | 898888 / **88.78** | **+2.44** |

两个同名 ZIP 对的初始化身份依据本轮上传顺序、完整实验清单和本地
`selected_submission.json` 对应关系确定。

汇总：

- Scratch 平均分：`86.3575`；
- Pretrained 平均分：`88.0850`；
- 平均预训练增益：`+1.7275`；
- 四个结构中预训练胜率：`4/4`；
- 网站全局最佳：full pretrained，`88.78`；
- 网站 scratch 最佳：纯 Hybrid-RMS，`86.71`。

## 2. 结构效应

### Scratch 条件

以纯 Hybrid-RMS 的 86.71 为基线：

- 加 Motion：`-0.38`；
- 加 Multiscale：`-0.66`；
- 同时加 Motion + Multiscale：`-0.37`。

因此在 scratch 条件下没有证据保留两个扩展模块作为默认结构。当前最合理的
scratch 主线是纯 `raw_apmd_hybrid_rms`，full 结构只能保留为消融证据。

### Pretrained 条件

以纯 Hybrid-RMS 的 88.07 为基线：

- 加 Motion：`-0.10`；
- 加 Multiscale：`-0.55`；
- 同时加 Motion + Multiscale：`+0.71`。

两个扩展只有共同存在时才形成明显正交互，这与本地 Proxy F1 的结论一致。

## 3. 本地指标与网站指标的一致性

本地四个配对中 pretrained 全部优于 scratch，网站同样为 4/4。full pretrained
在本地 Proxy F1 和网站 Score 上都是第一，说明这轮实验没有出现“本地预训练占优、
隐藏测试反转”的现象。

额外历史提交 `apmd_multiscale_contrast_seed49.zip` 为 87.32：高于本轮所有 scratch
结果，但低于本轮四个 pretrained 结果，也没有提供弃用预训练的性能证据。

## 4. 决策解释

从实验性能证据出发，应继续使用预训练；但项目负责人已明确决定以后只研究从随机
初始化训练的模型。因此本项目从 2026-08-25 起执行 **scratch-only 研发策略**。

必须区分：

1. **证据结论**：预训练在本轮网站结果中显著更好；
2. **研发约束**：未来代码和实验不得加载预训练初始化权重。

历史 pretrained checkpoint、日志、ZIP 和结果文档继续保留，用于审计和结果复核；
推理历史 checkpoint 不属于“用预训练权重初始化新训练”。

## 5. Scratch-only 后续建议

1. 以纯 Hybrid-RMS scratch 86.71 作为新基线；
2. 先补 seed 49，确认 scratch 基线稳定性；
3. 纯 Hybrid-RMS scratch 的本地最佳 checkpoint 在 epoch 86，靠近 100 epoch 上限，
   可优先测试 130–150 epoch、较长 warmup 或后段更平滑的学习率衰减；
4. 新结构一律与同 seed、同训练预算的纯 Hybrid-RMS scratch 对照；
5. 在没有 scratch 正收益前，不把 motion detrend 或 multiscale contrast 加回默认主线；
6. 禁止通过 `base_ckpt`、TDCSTA branch checkpoint 或 launcher 初始化模式绕过策略。
