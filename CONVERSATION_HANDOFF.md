# DeepPro / CSIG2026 对话交接总结

更新时间：2026-07-22（Asia/Shanghai）

## 1. 当前任务背景

目标是在远程服务器上使用 4 张 A100 训练和测试 DeepPro，并针对以下问题持续优化：

- 训练阶段 GPU 长时间等待数据、利用率呈 `0% → 100% → 0%` 波动。
- train/eval 切换时显存没有及时下降。
- 训练与验证指标缺少 Precision、Recall 和 F1。
- 需要多种适合红外小目标/极端类别不平衡的损失函数，并通过命令行选择。
- 使用预训练权重生成验证集质心 TXT 和随机 100 张可视化图片。
- `test.py` 最终完整评估很慢。
- 优化必须尽量不改变模型权重、预测结果和最终指标。

## 2. 关键路径和运行环境

### 本地代码

```text
/home/user/4T_Storage/SJY/CSIG2026/DeepPro-main
```

### 远程代码

```text
/home/devbox/project/model/sjy/CSIG2026/Deeppro_v2/DeepPro-main
```

### 远程数据集

```text
/home/devbox/project/model/sjy/CSIG2026/datasets/SatVideoIRSDT_v1
```

### Conda 环境

```bash
conda activate sjyPID
```

### GPU

- 服务器有 8 张 A100。
- 当前训练计划使用 4 张 A100。
- 常用参数：`--gpu 0,1,2,3 --gpu_num 4`。

### 预训练权重

```text
/home/devbox/project/model/sjy/CSIG2026/Deeppro_v2/DeepPro-main/log/sem_seg/SatVideoIRSDT_v1__2026-07-21_08-48__SoftLoUloss_DeepPro-Plus_DataL40/checkpoints/best_model.pth
```

## 3. Git 状态

### 仓库和分支

```text
分支：BRTD-Adapter
最新提交：895e821
跟踪分支：CSIG2026/BRTD-Adapter
```

在生成本文件前：

```text
BRTD-Adapter 与本地记录的 CSIG2026/BRTD-Adapter 左右提交差为 0 / 0
工作区干净
```

远程配置：

```text
CSIG2026  git@github.com:Mao0324/DeepPro.git
origin    git@github.com:ShiJiayu-del/DeepPro-main.git
```

本交接文件创建后会作为新的未跟踪文件显示，除非后续提交。

### 关键提交

```text
895e821  优化 test.py、ShootingRules 和测试数据读取
affc323  添加统一二值分割损失库和 --loss 选择参数
22b4a1b  添加 IoU、Precision、Recall、F1 指标
dc218fd  重构 DataLoader 和训练/验证数据路径，提高训练速度
1a5d687  优化 train/eval 阶段显存释放
c22fcd2  添加随机可视化和质心 TXT 输出功能
```

## 4. 已完成的训练速度优化

主要修改集中在：

```text
train.py
data_utils/TrainDataLoader.py
data_utils/TestDataLoader.py
```

已完成内容：

- 训练和验证 DataLoader 支持独立 worker 数量。
- 新增参数：

  ```text
  --train_workers
  --val_workers
  --prefetch_factor
  ```

- worker 大于 0 时启用持久化 worker 和预取。
- 使用 `pin_memory=True` 和 GPU `non_blocking=True` 传输。
- 将验证窗口整理为长期存在的 DataLoader，避免每个序列重复启动 worker。
- 训练指标在 GPU 上累计，只在 epoch 结束同步一次，移除了每 batch 的大张量 `.cpu().numpy()`。
- `optimizer.zero_grad(set_to_none=True)` 降低梯度显存占用。
- 已关闭每 batch 的：

  ```python
  torch.autograd.set_detect_anomaly(True)
  ```

  异常检测只应在定位 NaN/非法梯度时临时打开，否则会显著拖慢训练。

### 推荐初始参数

```text
--train_workers 8
--val_workers 4
--prefetch_factor 2
```

worker 不是越多越好。若磁盘随机读取或共享内存成为瓶颈，应测试 `4/8/12`，比较每秒处理窗口数。

## 5. 显存处理

`train.py` 已加入阶段边界显存清理：

- checkpoint 加载后清理。
- 训练结束、验证开始前删除最后一个 batch 的张量和梯度。
- 验证结束后清理。
- 整个训练结束并启动独立 `test.py` 前删除 model、criterion、optimizer 和 DataLoader。
- 日志打印 CUDA `allocated` 和 `reserved`。

重要理解：

- `nvidia-smi` 显示的是进程占用，不等于当前活跃张量大小。
- `torch.cuda.empty_cache()` 只能释放没有活跃引用的缓存块。
- 模型参数、Adam 状态和 CUDA context 在训练进程存活时仍会常驻。
- `train.py` 训练结束后会自动调用 `test.py`，所以显存重新升高可能是测试进程已经开始，不一定是训练泄漏。

## 6. 新增训练和验证指标

当前训练和验证均记录微平均像素级：

```text
IoU
Precision
Recall
F1
```

计算在 GPU 上累计 TP、预测正像素和真实正像素，epoch 结束只同步一次。

注意：这里是像素级指标，不等同于比赛平台按质心/目标匹配计算的目标级 F1。

## 7. 损失函数系统

实现位置：

```text
networks/losses/__init__.py
networks/losses/segmentation_losses.py
```

训练入口：

```bash
--loss <名称>
```

支持的损失：

```text
soft_iou
frame_soft_iou
bce
focal
dice
bce_dice
tversky
focal_tversky
lovasz
sls_iou
tda_sls
hard_focal
tversky_hard_focal
stc_f1
```

### 重要说明

- 默认 `--loss soft_iou`。
- 新的 `soft_iou` 已与模型原始 `MODEL.SoftLoUloss()` 做过损失值和每个 logits 梯度的逐元素一致性检查。
- `tversky_hard_focal` 推荐作为第一个新损失对照实验。
- `stc_f1` 是中心响应和时序一致性组合的实验性损失，需要做消融实验。
- `tda_sls` 使用 CPU 连通域和逐目标局部裁剪，会引入 GPU/CPU 同步，速度明显慢于纯 GPU 损失。
- `tda_mean_size=0` 和 `tda_mean_contrast=0` 时使用当前 batch 统计量；严格复现论文式 TDA 时应先计算训练集全局均值。

常用调节：

```bash
# 更压制虚警
--tversky_fp_weight 0.7 --tversky_fn_weight 0.3

# 更重视召回率
--tversky_fp_weight 0.4 --tversky_fn_weight 0.6
```

## 8. sample_rate 和归一化结论

### sample_rate

`sample_rate` 不是视频逐帧抽帧率，而是每个 epoch 从候选训练窗口集合中抽取多少比例。

例如：

```text
候选窗口 10000 个
sample_rate = 0.04
每个 epoch 使用约 400 个训练样本
```

增大 `sample_rate` 会增加每个 epoch 的数据量和耗时，但不保证模型效果单调变好。更合理的是保持总优化步数可比后做实验。

### 均值和标准差

数据集均值/标准差用于：

```python
normalized = (image - train_mean) / train_std
```

其作用是稳定数值范围和优化过程。验证/测试必须使用训练集统计量，不能使用验证集自身统计量。

SatVideoIRSDT_v1 的统计量已集中在 loader 工具中使用。

## 9. 质心 TXT 和随机可视化

目标：

- 完整验证集质心 TXT。
- 随机 100 张预测可视化。

相关参数：

```text
--centroid_txt
--centroid_threshold
--centroid_dir
--visual
--visual_count
--visual_seed
--output_only
```

推荐命令：

```bash
cd /home/devbox/project/model/sjy/CSIG2026/Deeppro_v2/DeepPro-main
conda activate sjyPID

python -u test.py \
  --gpu 7 \
  --batch_size 1 \
  --test_workers 2 \
  --prefetch_factor 1 \
  --seqlen 40 \
  --dataset SatVideoIRSDT_v1 \
  --datapath /home/devbox/project/model/sjy/CSIG2026/datasets/SatVideoIRSDT_v1 \
  --logpath /home/devbox/project/model/sjy/CSIG2026/Deeppro_v2/DeepPro-main/log/ \
  --log_dir SatVideoIRSDT_v1__2026-07-21_08-48__SoftLoUloss_DeepPro-Plus_DataL40 \
  --centroid_txt \
  --centroid_threshold 0.5 \
  --visual \
  --visual_count 100 \
  --visual_seed 46 \
  --output_only
```

输出目录默认：

```text
<experiment>/out_centroid
<experiment>/visual_random_100_seed46
```

### 重要约束

`test.py` 当前必须使用：

```text
--batch_size 1
```

旧代码虽然暴露 `batch_size`，但序列窗口拼接不支持批量窗口；此前使用 `batch_size=20` 触发过 squeeze/拼接错误。现在会明确拒绝非 1 的测试 batch，避免产生错误结果。

### 关于平台 F1=0 和目标过多

此前生成的质心结果出现大量目标且平台 F1=0。可能原因包括：

- 阈值 `0.5` 下分割图有大量孤立噪点，每个连通域都被转换成目标。
- 平台使用目标级匹配，而本地主要观察像素级指标。
- 提交格式、帧编号、行列坐标顺序或序列文件命名需要严格对应比赛 PDF。
- 当前质心提取不会自动过滤小连通域，避免在未确认比赛规则时擅自改变结果。

该问题尚需使用少量人工核对帧，把预测 mask、质心 TXT、真实目标位置和 PDF 格式逐项比对。

## 10. 最新 test.py 无损加速

最新提交：

```text
895e821
```

修改文件：

```text
ShootingRules.py
data_utils/TestDataLoader.py
test.py
```

### ShootingRules

旧逻辑对每一帧的 27 个阈值重复执行：

```text
measure.label
measure.regionprops
整图复制
逐目标扫描
```

新逻辑：

- 每帧只解析一次目标连通域。
- 对有限 sigmoid 概率，预先计算每个目标邻域最大响应。
- 对虚警区域预测值排序，使用 `np.searchsorted()` 一次得到全部阈值计数。
- 如果输出包含 NaN/Inf，自动回退到旧逐阈值逻辑。

512×512 合成测试：

```text
旧逻辑约 0.2466 秒/帧
新逻辑约 0.0059 秒/帧
该后处理部分约加速 41.93 倍
```

这是 ShootingRules 后处理的局部基准，不是整个验证集端到端加速倍数。

### test.py

- 使用一个全局 `ConcatDataset + DataLoader`，避免每个序列重新启动 worker。
- 新增：

  ```text
  --test_workers（默认 2）
  --prefetch_factor（默认 1）
  --profile_flops
  ```

- 使用 `torch.inference_mode()`。
- 图片使用 pinned memory 和非阻塞 GPU 传输。
- 标签不再无意义地传到 GPU。
- 立即删除未使用的 `seq_features`。
- 序列结果预分配，重叠窗口仍保持原始逐像素 `torch.maximum` 规则。
- checkpoint 使用 `map_location='cpu'` 加载，写入模型后立即删除 checkpoint，避免测试时保留 Adam 状态。
- FLOPs 分析不再默认额外执行一次模型 forward；只有传入 `--profile_flops` 才执行。
- 日志打印完整测试耗时和窗口数。

### output_only

`--output_only` 时：

- 不再读取或解码验证标签/质心标注。
- 仍读取并归一化输入图像。
- 模型预测、质心 TXT 和可视化逻辑不变。

## 11. 已完成检查

均在 `sjyPID` 环境执行：

### 损失函数

- 14 个损失正常目标前向/反向。
- 空目标前向/反向。
- 极端 logits 前向/反向。
- `[1,40,128,128]` 生产形状检查。
- 原始和新 SoftIoU 损失/梯度逐元素一致。

### test.py 加速

- 30 组随机和边界目标测试。
- 27 个阈值的 FalseNum、TrueNum、TgtNum 与旧代码逐项完全一致。
- NaN 回退路径与旧逻辑一致。
- 新旧长序列拼接逐元素完全一致。
- `torch.no_grad()` 和 `torch.inference_mode()` 模型输出逐元素一致。
- 仓库内真实 DeepPro-Plus checkpoint 使用 CPU 加载，所有 state_dict key 匹配。
- `output_only` 无标注读取路径检查通过。
- `python -m py_compile` 通过。
- `git diff --check` 通过。

限制：本地检查环境没有可用 CUDA，因此尚未在本机做完整 A100 端到端测试；需要远程拉取后测量实际总耗时和 GPU 利用率。

## 12. 四卡训练示例

```bash
cd /home/devbox/project/model/sjy/CSIG2026/Deeppro_v2/DeepPro-main
conda activate sjyPID

python -u train.py \
  --gpu 0,1,2,3 \
  --gpu_num 4 \
  --model DeepPro-Plus \
  --dataset SatVideoIRSDT_v1 \
  --datapath /home/devbox/project/model/sjy/CSIG2026/datasets/SatVideoIRSDT_v1 \
  --savepath /home/devbox/project/model/sjy/CSIG2026/Deeppro_v2/DeepPro-main/log/ \
  --batch_size 4 \
  --seqlen 40 \
  --patch_size 128 \
  --sample_rate 0.04 \
  --train_workers 8 \
  --val_workers 4 \
  --prefetch_factor 2 \
  --loss tversky_hard_focal
```

若需要完全延续原损失基线：

```bash
--loss soft_iou
```

## 13. 服务器拉取命令

先检查服务器本地修改：

```bash
cd /home/devbox/project/model/sjy/CSIG2026/Deeppro_v2/DeepPro-main
conda activate sjyPID
git status --short
```

如果 `train.py/test.py/ShootingRules.py/data_utils` 有本地修改，先保存：

```bash
git stash push -m "server-before-895e821" -- \
  train.py \
  test.py \
  ShootingRules.py \
  data_utils/TrainDataLoader.py \
  data_utils/TestDataLoader.py
```

拉取：

```bash
git remote set-url CSIG2026 https://github.com/Mao0324/DeepPro.git
git fetch CSIG2026 BRTD-Adapter
git switch BRTD-Adapter
git pull --ff-only CSIG2026 BRTD-Adapter

git log -1 --oneline
git status --short
```

期望最新提交：

```text
895e821
```

服务器没有 `rg` 时使用：

```bash
grep -nE 'test_workers|evaluate_thresholds|profile_flops' \
  test.py ShootingRules.py
```

语法检查：

```bash
python -m py_compile \
  train.py \
  test.py \
  ShootingRules.py \
  data_utils/TrainDataLoader.py \
  data_utils/TestDataLoader.py \
  networks/losses/segmentation_losses.py
```

## 14. 已知风险和后续建议

1. 在远程 A100 上分别运行一次完整评估和 `--output_only`，记录新日志中的总时间。
2. 若 DataLoader worker 报共享内存不足或 bus error，改为：

   ```bash
   --test_workers 0
   ```

   结果不会变化，但读取可能变慢。

3. `test.py` 不要设置 `batch_size>1`，当前会明确报错。
4. 完整测试默认不执行 FLOPs profiling；需要时显式加 `--profile_flops`。
5. `tda_sls` 会明显拖慢训练和验证损失计算，不建议作为第一组速度基线。
6. 平台 F1=0 和目标过多尚未完全解决，应优先核查质心 TXT 格式和预测连通域数量，再决定是否增加最小面积过滤或时序跟踪；这类后处理会改变提交结果，不能作为“无损速度优化”直接加入。
7. 如果服务器存在旧 stash：

   ```bash
   git stash list
   ```

   不要直接 `stash pop`，先用 `git diff HEAD 'stash@{n}' -- <file>` 检查，避免把旧版训练代码重新覆盖到新版本。

## 15. 新对话建议开场

可以在新对话中直接发送：

```text
请先阅读：
/home/user/4T_Storage/SJY/CSIG2026/DeepPro-main/CONVERSATION_HANDOFF.md

继续维护 BRTD-Adapter 分支。当前最新提交应为 895e821，远程服务器使用 sjyPID 环境和4张A100。请先检查 git 状态，再继续后续任务，不要覆盖服务器日志或未提交修改。
```
