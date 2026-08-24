# DeepPro / BRTD 新对话交接文档

> 更新时间：2026-08-20 02:58 UTC
> 仓库：/home/devbox/project/model/sjy/CSIG2026/Deeppro_v2/DeepPro-main
> 分支：BRTD-Adapter
> 数据集：/home/devbox/project/model/sjy/CSIG2026/datasets/SatVideoIRSDT_v1

## 0. 新对话的第一条指令

先完整阅读本文档，再做只读实时检查。不要因为本文记录了 RUNNING 就假设任务
仍在运行，也不要因为单次 GPU 利用率为 0 就判断训练卡死。

~~~bash
cd /home/devbox/project/model/sjy/CSIG2026/Deeppro_v2/DeepPro-main
git status --short --branch
git log -3 --oneline --decorate
screen -ls | grep csig_apmd_rms
nvidia-smi
ps -eo pid,ppid,etimes,stat,args | grep -E 'train\.py|run_structure_candidate_experiment|centroid_f1_sweep|seg2tracked' | grep -v grep
cat log/sem_seg/2026-08-20/_structure_pipeline_status/*.status
~~~

先向用户汇报实时状态以及与本快照的差异，再执行后续操作。用户此前明确表示
不需要持续监控训练；只有用户询问状态时再检查。

## 1. 当前实时状态

生成本文时，两个 Raw-APMD-RMS 实验正在训练，状态文件均为 RUNNING：

| GPU | seed | Screen | 实验 |
|---:|---:|---|---|
| 0 | 47 | csig_apmd_rms_g0_seed47_2026-08-20_02-49-07 | brtd3_raw_apmd_rms_seed47 |
| 1 | 49 | csig_apmd_rms_g1_seed49_2026-08-20_02-49-07 | brtd3_raw_apmd_rms_seed49 |

Screen：

~~~bash
screen -r csig_apmd_rms_g0_seed47_2026-08-20_02-49-07
screen -r csig_apmd_rms_g1_seed49_2026-08-20_02-49-07
~~~

状态文件：

~~~text
log/sem_seg/2026-08-20/_structure_pipeline_status/brtd3_raw_apmd_rms_seed47.status
log/sem_seg/2026-08-20/_structure_pipeline_status/brtd3_raw_apmd_rms_seed49.status
~~~

实验目录：

~~~text
log/sem_seg/2026-08-20/SatVideoIRSDT_v1__2026-08-20_02-49-07__F1OHEM-brtd3_raw_apmd_rms_seed47_E100
log/sem_seg/2026-08-20/SatVideoIRSDT_v1__2026-08-20_02-49-07__F1OHEM-brtd3_raw_apmd_rms_seed49_E100
~~~

本快照中两张卡各占约 54865 MiB，训练进程和 DataLoader 子进程存在。
两个日志均刚进入 epoch 1。日志已经确认：

- DeepPro-Plus 预训练权重成功加载；
- 新 RMS adapter 有 11 个参数键；
- 使用 f1_calibrated_ohem；
- backbone 学习率是 adapter 的 5 倍；
- 早停已经启用。

不要把 DataLoader 子进程继承的 train.py 命令行误认为重复启动。

## 2. 当前训练配置

| 项目 | 值 |
|---|---|
| 模型 | DeepPro-Plus_BRTD3 |
| 结构变体 | raw_apmd_rms |
| seeds | 47、49；以后正式实验只跑两个 seed |
| 最大 epoch | 100 |
| 早停 | eval_f1，patience=30，min_delta=1e-4，start_epoch=15 |
| adapter LR | 0.001 |
| backbone LR | 0.005，base_lr_mult=5.0 |
| loss | f1_calibrated_ohem |
| batch size | 20 |
| sequence length | 40 |
| patch size | 128 |
| valid frame mask | 开启 |
| SwanLab group | f1_raw_apmd_rms_2seed_2026-08-20_02-49-07 |
| Python | /home/devbox/project/model/miniconda3/envs/sjyPID/bin/python |
| 启动器 | tools/launch_raw_apmd_rms_2gpu.sh |
| runner | tools/run_structure_candidate_experiment.sh |

训练结束后 runner 会自动：

1. 从像素 F1 中选择 Top-5 已保存检查点；
2. 对每个候选扫描质心阈值与 min_area；
3. 保留验证质心代理 F1 最优结果；
4. 生成轨迹 TXT；
5. 生成比赛 ZIP；
6. 严格校验 255 个序列和真实帧数；
7. 写入 SHA-256、COMPLETE 和状态文件。

预计 ZIP：

~~~text
...raw_apmd_rms_seed47_E100/submission/submit_brtd3_raw_apmd_rms_seed47_best_proxy_f1.zip
...raw_apmd_rms_seed49_E100/submission/submit_brtd3_raw_apmd_rms_seed49_best_proxy_f1.zip
~~~

只有状态为 COMPLETE、ZIP 存在且 submission_validation.txt 为 VALID 时，
才向用户报告“可提交”。

## 3. 为什么做 Raw-APMD-RMS

8 月 17 日 Raw-APMD 三个 seed 的本地质心代理结果：

| seed | Precision | Recall | F1 |
|---:|---:|---:|---:|
| 46 | 0.920209 | 0.677930 | 0.780705 |
| 47 | 0.935289 | 0.676964 | 0.785431 |
| 49 | 0.934005 | 0.665462 | 0.777190 |

均值 F1 约 0.7811，优于 8 月 13 日最佳 BRTD2 no-gate 的 0.776350。
相同 seed 配对时，Raw-APMD 相对 DeepPro：

- seed 47：0.760839 -> 0.785431，提升 0.024592；
- seed 49：0.770413 -> 0.777190，提升 0.006777。

提升主要来自 Recall，说明独立原始帧外观旁路能补回差分主干遗漏的弱目标。
训练检查点中的时间尺度权重也已发生有意义的学习：一阶动态总体偏向 step=4，
二阶动态在通道间明显分化。

原 Raw-APMD 使用逐帧 GroupNorm，会逐帧减去均值，与保留绝对辐射外观的
目标存在冲突。数据加载器已经使用固定数据集均值和方差，因此新增
raw_apmd_rms，以不减均值的逐帧 RMSNorm 代替 GroupNorm。当前实验只改
归一化，损失、学习率和主结构保持不变，是单变量对照。

完整分析：

~~~text
docs/experiment_analysis_2026-08-20.md
docs/raw_apmd.md
~~~

## 4. 两随机种子规则与接受标准

以后每个正式配置只使用 seed 47 和 49，不再跑三个随机种子。选择这两个不是
因为挑中了最高结果，而是二者同时有 8 月 13 日 DeepPro 与 8 月 17 日
Raw-APMD 配对基线。

Raw-APMD 在 seed 47/49 上的基线均值为 0.781311。接受 RMS 改进的条件：

1. 两 seed 平均质心代理 F1 至少达到约 0.7843，即提升至少 0.003；
2. seed 47 不低于约 0.7824；
3. seed 49 不低于约 0.7742；
4. Precision 不出现明显崩塌；
5. 最终仍需 CodaBench 网站结果确认，本地代理不等于隐藏测试集 F1。

若通过，下一阶段才把 adapter LR 单独降到 0.0005，并仍跑 seed 47/49。
不要同时修改 loss、backbone LR 或采样。模型确定后再做双 seed 概率平均
与统一阈值搜索；集成结果不用于结构归因。

## 5. 当前最可靠的历史结论

1. 比赛轨迹准确度和完整度已经能满分，主要目标是检测 F1。
2. 行列坐标颠倒问题已经修复；不要再次交换 x/y。
3. DeepPro validmask 的网站总分曾达到 85.72，本地代理 F1 为 0.763120。
4. BRTD2 no-gate 优于带门控变体；输入依赖门控容易压制弱目标。
5. 8 月 13 日 BRTD2 的 backbone LR 从 0.0005 提高到 0.005 时本地结果
   总体改善，当前保留 0.005。
6. adapter LR 0.0005 相对 0.001 的单 seed 收益只有约 0.0004，小于
   seed 波动，尚不能认定有效。
7. pixel F1 最佳 epoch 不一定是质心 F1 最佳 epoch。Raw-APMD seed 47
   的质心最佳来自 pixel 候选第三名，因此 runner 已从 Top-3 改为 Top-5。
8. min_area=2 在 Raw-APMD 三个 seed 上一致，孤立单像素假阳性仍值得抑制。

## 6. 代码与 Git 状态

当前 HEAD：

~~~text
e47cc63 删除没用的脚本
~~~

HEAD 与 CSIG2026/BRTD-Adapter 一致，但以下改动尚未提交或推送：

- networks/layers/structure_adapters.py
  - 新增 raw_apmd_rms 和 FramewiseRMSNorm；
- networks/models/DeepPro-Plus_BRTD3.py
  - 将 raw_apmd_rms 接到 raw_fusion 位置；
- train.py
  - 增加结构参数白名单；
- tools/check_brtd3.py
  - 增加 RMS 变体的恒等性、梯度与补帧检查；
- tools/run_structure_candidate_experiment.sh
  - checkpoint 候选 Top-3 -> Top-5；
- tools/launch_raw_apmd_rms_2gpu.sh
  - 新增双 GPU、双 seed Screen 启动器；
- tools/launch_raw_apmd_3gpu.sh
  - 已从工作区删除，避免以后误跑三 seed；
- docs/raw_apmd.md
  - 更新双 seed 流程；
- docs/experiment_analysis_2026-08-20.md
  - 新增实验复盘；
- 本交接文档。

已完成验证：

- Python 语法检查通过；
- shell bash -n 通过；
- raw_apmd_rms 加载原预训练权重后初始输出完全一致；
- residual projection 梯度非零；
- 有效帧前后补帧误差为 0；
- 补帧区域响应为 0；
- 原 raw_apmd 回归检查通过；
- Top-5 选择器已用 8 月 17 日 seed 47 日志验证。

新对话不要运行 git reset、checkout 或清理未跟踪文件。训练目录会保存源码快照，
但正式提交 Git 前仍应先复查 diff。

## 7. 本次日志清理

用户明确要求删除 SatVideoIRSDT_v1 的无用训练日志。本次只删除了 6 个能够
明确判定为失败、无提交 ZIP 的目录，约释放 19 MB：

~~~text
log/sem_seg/2026-07-22/SatVideoIRSDT_v1__2026-07-22_03-36__SoftLoUloss_DeepPro-Plus_DataL40
log/sem_seg/2026-07-22/SatVideoIRSDT_v1__2026-07-22_06-19__SoftLoUloss_DeepPro-Plus_DataL40
log/sem_seg/2026-07-28/SatVideoIRSDT_v1__2026-07-28_11-28-38__Loss-frame-soft-iou-Pretrained_DeepPro-Plus_DataL40
log/sem_seg/2026-08-17/SatVideoIRSDT_v1__2026-08-17_03-11-14__F1OHEM-brtd3_raw_apmd_seed46_E100
log/sem_seg/2026-08-17/SatVideoIRSDT_v1__2026-08-17_03-11-14__F1OHEM-brtd3_raw_apmd_seed47_E100
log/sem_seg/2026-08-17/SatVideoIRSDT_v1__2026-08-17_03-11-14__F1OHEM-brtd3_raw_apmd_seed49_E100
~~~

这些目录已永久删除，不能从日志目录直接恢复，但实验可重新运行。所有已有 ZIP
的实验、8 月 13/17 的核心对照、8 月 20 日当前训练都保留。不要把
brtd_fixed_router 因缺少 COMPLETE 就当作失败：它已有网站结果和有效 ZIP。

部分 8 月 13 日完成实验仍保存 postprocess/probabilities，单目录约 24 MB。
这些概率可用于后续 ensemble，因此本次没有删除。

## 8. 新对话中的下一步

1. 先执行第 0 节只读命令。
2. 如果用户问训练是否完成：
   - 检查两个状态文件；
   - 检查 Screen/进程；
   - 查看模型日志最后 epoch；
   - 检查 COMPLETE、ZIP、SHA-256 和 VALID。
3. 两个实验都完成后，从 selected_submission.json 提取 threshold、
   min_area、precision、recall、proxy_f1，按第 4 节判据比较。
4. 若 RMS 通过，向用户报告并建议下一阶段 adapter LR 0.0005 双 seed；
   若未通过，保留原 raw_apmd，不要擅自堆叠更多模块。
5. 未经用户再次授权，不要删除更多有效日志、停止训练、提交 Git 或推送远程。
6. 用户要求提交代码时，先排除 log/、检查未跟踪文件，再提交到
   BRTD-Adapter 分支。

## 9. 可复制到新对话的提示

~~~text
请先完整阅读：
/home/devbox/project/model/sjy/CSIG2026/Deeppro_v2/DeepPro-main/docs/BRTD_CONVERSATION_HANDOFF_2026-08-20.md

这是 DeepPro/BRTD 比赛项目的最新交接。先只读检查 git status、Screen、
GPU、训练进程以及 2026-08-20 两个状态文件，并汇报它们与交接快照的差异。
当前阶段是 Raw-APMD-RMS 双 seed 47/49 单变量实验，最多 100 epoch、早停、
SwanLab、Top-5 质心后处理，完成后自动生成轨迹 ZIP。不要恢复三 seed 策略，
不要再次交换坐标，也不要擅自删除有 ZIP 的历史实验。
~~~
