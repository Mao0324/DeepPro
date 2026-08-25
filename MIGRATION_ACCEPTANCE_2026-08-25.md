# DeepPro / BRTD 新服务器迁移验收

> 验收日期：2026-08-25（Asia/Shanghai）
> 项目：`/home/user/4T_Storage/SJY/CSIG2026/DeepPro-main`
> 分支：`migration-2026-08-24`
> 基线 HEAD：`268838b chore: package DeepPro migration artifacts`

## 1. 验收结论

代码、训练/验证数据、`sjyPID` 环境、预训练权重、最终 epoch 75 checkpoint、发布集和
最佳提交 ZIP 均可用。CPU 与 CUDA 结构自检、真实训练 batch 加载以及历史 checkpoint
真实验证窗口推理均通过。

新服务器当前的固定 GPU 安全策略为：**只允许物理 GPU 0、1、2，禁止 GPU 3**。三卡
scratch launcher 对 4 个实验采用三个顺序队列，任何时刻最多运行三个训练进程。本次验收没有
启动训练。

## 2. 已通过项目

### 文件和结果

- 最终 8 个 `_structure_pipeline_status/*.status` 全部为 `COMPLETE`；
- 基础预训练权重存在并能兼容加载；
- 最佳 checkpoint：epoch 75；
- 最佳本地配置：threshold `0.17`、min area `2`、Proxy F1 `0.796586`；
- 发布集 `release/2026-08-22_pretrained_vs_scratch_seed47/SHA256SUMS` 全量校验通过；
- 完整实验目录和发布集中的 epoch 75 checkpoint SHA-256 一致：
  `74baa70fe04833570c713f808b100f7fddcb34c6f5781b1ae57ce3369c83ce84`；
- 两处最佳提交 ZIP SHA-256 一致：
  `d5254a191cc73522abe64267da9ff986b09fad94d336f857e0a8b885edefc331`；
- 提交 ZIP 重新校验通过：255 个序列、23,087 帧、65,496 个检测。

### 数据

- 数据根目录：`/home/user/4T_Storage/SJY/CSIG2026/datasets/SatVideoIRSDT_v1`；
- 数据体积约 50 GB；
- train：1,178 个序列，99,800 张图像和 99,800 张 mask；
- val：255 个序列，23,087 张图像和 23,087 张 mask；
- 真实训练 batch 加载通过：image `[2,1,40,128,128]`，label `[2,40,128,128]`，
  dtype 为 float32，数值有限，标签为 0/1。

### 环境和代码

- Python 3.8.5；
- PyTorch 2.1.2，CUDA build 12.1，cuDNN 8902；
- NumPy 1.24.4，SciPy 1.9.3；
- 服务器驱动 560.35.03，驱动报告 CUDA 12.6；
- 4 张 NVIDIA GeForce RTX 3090，每张 24 GB；
- 文档指定的 Python `py_compile` 检查通过；
- 迁移相关 Shell 脚本 `bash -n` 通过；
- `git diff --check` 通过；
- 完整 Hybrid-RMS + motion detrend + multiscale contrast CPU 自检通过；
- 同结构 CUDA 自检通过；
- GPU 1 上加载 epoch 75 checkpoint 并推理验证序列 `000001` 的首个 40 帧窗口成功：
  输入 `[1,1,40,256,256]`，输出 `[1,40,256,256]`，概率有限且范围为
  `[1.60e-9, 0.999846]`；
- 三张 smoke-test 概率图保存在临时目录
  `/tmp/deeppro_migration_smoke_gpu1.g9v7n1/`。
- 验证集同时包含 256×256 到 1024×1024 的多种尺寸；1024×1024×40 的 FP16
  autocast 实测峰值为 allocated 22.66 GiB、reserved 22.68 GiB，可在 GPU 1 的
  24 GiB 显存内运行；
- `test.py --amp` 已加入，模型前向使用 FP16 autocast，Sigmoid/概率保存保持 FP32；
  256×256 smoke test 上相对 FP32 的平均概率差为 `1.61e-7`，阈值 0.17 的二值结果
  无像素变化；新服务器 runner 默认设置 `TEST_USE_AMP=1`。
- 修改后的完整 `test.py --amp` 已在 GPU 1 上跑完验证序列 `000182`：1024×1024、
  374 帧、11 个窗口，耗时 38.27 秒，峰值 allocated 22.665 GiB、reserved
  22.697 GiB，并成功导出 3 张抽样概率图到
  `/tmp/deeppro_test_amp_gpu1.B54dsr/probabilities/`。
- `train.py --eval_amp 1` 已同步加入：训练期验证使用 AMP，并把重叠窗口的完整序列
  预测/标签放到 CPU 拼接，避免 GPU 累积结果与下一个 1024 窗口同时占用显存；同一
  374 帧序列的训练验证 smoke test 通过，峰值 allocated 22.821 GiB、reserved
  22.873 GiB；
- 新服务器统一默认 `TEST_EVAL_CHUNK_ROWS=32`、`TEST_USE_AMP=1` 和
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。

## 3. 迁移适配改动

- 新增 `tools/project_runtime_env.sh`：统一推导仓库、Python 和数据路径；
- 旧服务器绝对路径已从交接文档列出的运行脚本中移除；
- SwanLab 凭据不再使用旧服务器默认文件，必须通过 `SWANLAB_API_KEY` 或
  `SWANLAB_CREDENTIAL_FILE` 显式提供；
- runner 会拒绝 GPU 3 以及 GPU 0/1/2 之外的设备；
- 单卡 Raw-APMD launcher 默认使用 GPU 0；
- 双卡 RMS launcher 使用 GPU 0/1；
- 新增 `tools/run_structure_candidate_queue.sh`；
- 新增 `tools/launch_hybrid_rms_pretrain_ablation_3gpu.sh`，任务在 GPU 0/1/2
  上按 3/3/2 顺序排队；其 `--dry-run` 已通过；
- 原 6/8 卡 launcher 保留为历史入口，但因安全允许列表会在接触禁用 GPU 前终止，
  不应在本服务器用于启动实验。

推荐启动检查：

```bash
bash tools/launch_hybrid_rms_pretrain_ablation_3gpu.sh --dry-run
```

只有在确认 GPU 0/1/2 空闲、Screen 可用、磁盘充足并决定是否配置 SwanLab 后，才能
去掉 `--dry-run`。

## 4. 未完成或需人工/外部权限的事项

1. 未向比赛网站上传最佳 ZIP；这是外部提交操作，需要比赛账号和明确授权；
2. SwanLab API key 尚未在新服务器配置；旧 key 的控制台轮换需要账号权限；
3. 完整迁移压缩包存在，但源服务器生成的配套 `.sha256` 文件未迁移，因此无法对
   3.17 GB 压缩包执行源端到目的端的严格哈希比对；关键发布集已通过独立 SHA256SUMS；
4. 数据目录包含 train 和 val，没有独立 test 目录；当前本地评估/ZIP 校验不受影响，
   若正式测试需要隐藏测试数据，应另行取得；
5. 环境同时安装了 `opencv-python 4.9.0.80` 和
   `opencv-python-headless 4.10.0.84`，实际 `import cv2` 报告 4.9.0。现有全部验收通过，
   未在未授权情况下改动共享 Conda 环境；
6. 项目磁盘使用率约 97%，尚余约 120 GB。足够当前验收，但开展新一轮全量概率导出
   前需要持续关注空间；
7. Screen 在宿主环境可访问，验收时无运行会话；沙箱内 `/run/screen` 只读，正式启动
   应在正常宿主终端执行。

## 5. 当前运行状态

- 无 `train.py`、结构 runner 或三卡队列进程；
- Screen：无会话；
- 本次没有启动训练、上传提交包或写入任何密钥。

## 6. 2026-08-25 网站结果后的策略变更

网站四结构 × 两初始化的 8 个配对结果显示 pretrained 全部优于 scratch，平均增益
`+1.7275`，最高分为 full pretrained 的 `88.78`。完整分析见
`WEBSITE_RESULTS_ANALYSIS_2026-08-25.md`。

尽管性能证据支持 pretrained，项目负责人已明确将后续研发约束改为 scratch-only。
代码现已拒绝所有 `base_ckpt`、spatial branch checkpoint 和 spatio-temporal branch
checkpoint 初始化；历史 pretrained 结果仅保留用于审计和推理复核。

## 7. 重新解压后的代码恢复复验

2026-08-25 项目压缩包被再次解压到工作目录，覆盖了尚未提交的受 Git 跟踪代码。
随后按本验收清单恢复了路径迁移、三卡限制、24 GB 显存优化和 scratch-only
策略。数据集解压完成后再次核对为 train 1,178 个序列、val 255 个序列；验证序列
`000182` 仍为 374 张图像和 374 张 mask。

恢复后重新执行的高风险回归结果：

- `test.py --amp`：1024×1024、374 帧、11 个窗口，24.30 秒，峰值 allocated
  `22.665 GiB`、reserved `22.697 GiB`；
- `train.py` 训练期验证函数：同一完整序列，峰值 allocated `22.821 GiB`、reserved
  `22.873 GiB`；
- `train.py`、`train_BRTD.py`、结构 runner、队列和 TDCSTA 分支预训练入口均再次通过
  预训练拒绝测试；
- BRTD3 Hybrid-RMS 随机初始化自检和三卡 scratch launcher dry-run 再次通过；
- 恢复期间没有启动正式训练，也没有修改 checkpoint、历史实验或比赛提交包。
