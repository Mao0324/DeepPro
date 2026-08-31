# 发布清单

## 必需产物

- `checkpoint/epoch_86_model.pth`：最终 scratch checkpoint；
- `artifacts/submit_hrms_scratch_epoch86_adaptive_thr0p16_highres0p96.zip`：最终提交；
- `artifacts/TXT_SHA256SUMS`：ZIP 内 220 个 TXT 的内容哈希；
- `SHA256SUMS`：本发布目录全部冻结文件的哈希。

## 关键源码

- `source_snapshot/DeepPro-Plus_BRTD3.py`
- `source_snapshot/structure_adapters.py`
- `source_snapshot/test.py`
- `source_snapshot/TestDataLoader.py`
- `source_snapshot/seg2tracked_centroid_txt.py`
- `source_snapshot/validate_submission_zip.py`

`source_snapshot/segmentation_losses.py` 是 epoch-86 模型训练时的实验快照；当前分支的
损失代码在后续实验中继续演进，因此两者有意不同。提交复现不调用损失文件。
`source_snapshot/train.py` 和 `TrainDataLoader.py` 保存赛季结束时的训练入口与数据流程；
实际提交复现由 `scripts/reproduce_submission.sh` 调用本分支中与快照一致的推理代码。

## 验证证据

- `validation/hrms_epoch_86.json`：同一 checkpoint 的完整 AMP 阈值扫描；
- `validation/highres.json`：10 个 1024×1024 标注序列专项阈值扫描；
- `evidence/independent_audit.txt`：最终 ZIP 独立语义审计；
- `evidence/provenance.txt`：模型、阈值、轨迹和内存优化来源；
- `evidence/training.log`：完整 scratch 训练日志；
- `evidence/scratch_training_evidence.txt`：随机初始化关键日志摘录。
