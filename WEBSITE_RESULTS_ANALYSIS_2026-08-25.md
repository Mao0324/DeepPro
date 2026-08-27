# 网站结果分析与 Scratch-only 决策（更新至 2026-08-27）

> 初次分析日期：2026-08-25；最近更新：2026-08-27
> 数据来源：比赛网站提交列表截图和对应本地实验产物
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

## 6. 2026-08-26 Scratch-init 新结果

| 网站 ID | 本地对应提交文件 | 网站分数 | 本地选中结果 |
|---:|---|---:|---|
| 902114 | `submit_hrms_scratch_init_ddp3_seed47_best_proxy_f1.zip` | **86.45** | epoch 80，threshold 0.16，min_area 2，Proxy F1 0.771051 |

本地选中结果的 Precision 为 `0.954120`，Recall 为 `0.646924`。与同 seed 的历史纯
Hybrid-RMS scratch 基线相比：

- 网站分数：`86.45 - 86.71 = -0.26`；
- 本地 Proxy F1：`0.771051 - 0.774414 = -0.003363`。

本地和网站方向一致，均不支持“只把全零 projection 改为 `0.05×Kaiming` 就能提升
最终性能”的假设。该改动仍然解决了首个反向步骤中 adapter 上游梯度为零的机制问题，
但机制正确不等于最终泛化更好，因此 `scratch_init` 不升级为新的成绩基线。当前网站
scratch 基线仍为纯 Hybrid-RMS 的 `86.71`。

`scratch_bandpass` 是在 scratch-init 上增加独立时域带通变量。它应继续完成，以
`scratch_init` 判断 bandpass 的增量效应，并以 `86.71` 判断整个组合是否值得替换原
Hybrid-RMS；只有同时看这两个对照，才不会把初始化的 `-0.26` 影响错误归因给带通模块。

## 7. 2026-08-27 Scratch-bandpass 最终结果

| 网站 ID | 本地对应提交文件 | 网站分数 | 本地选中结果 |
|---:|---|---:|---|
| 903589 | `submit_hrms_scratch_bandpass_ddp3_seed47_best_proxy_f1.zip` | **86.47** | epoch 80，threshold 0.10，min_area 2，Proxy F1 0.772087 |

本地选中结果的 Precision 为 `0.943955`，Recall 为 `0.653164`。其相对关系为：

- 相比 scratch-init：网站 `+0.02`，本地 Proxy F1 `+0.001036`；
- 相比历史 Hybrid-RMS scratch：网站 `-0.24`，本地 Proxy F1 `-0.002327`；
- 误差组成：FP `3,412`，FN `30,516`，漏检占 FP+FN 的 `89.95%`。

因此 bandpass 有很弱的同母体正增量，但不足以抵消 scratch-init 的退化，也不能升级为
新基线。Raw-APMD 系列继续做小模块叠加的边际收益已经很低，下一轮改为独立的
FeedbackSTS 风格时空反馈网络；新网络仍从随机权重训练，并以 `86.71` 为网站晋级线。
