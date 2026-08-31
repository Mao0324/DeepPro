# 端到端复现验证

验证日期：2026-08-31（Asia/Shanghai）

执行命令：

```bash
FINAL_SUBMISSION_GPU=0 \
FINAL_SUBMISSION_STAMP=release_verification_2026-08-31 \
bash release/2026-08-29_final_submission_score91.30_scratch/scripts/reproduce_submission.sh
```

结果：

```text
sequences=220
frames=21285
detections=48673
released_zip_sha256=7348c804cf3f6e8f1142fdee0dccc8621fc8e681065e0abc05c37d9499e3437b
reproduced_zip_sha256=ef94e15737673d465c9bcbf1e27695d0cc644a2d26eaf96aef851a5614b17e74
TXT_CONTENT_COMPARISON=PASS members=220 changed=0
```

两个 ZIP 的文件哈希不同，是因为 ZIP 成员时间戳来自各自生成时间；解压后的 220 个
TXT 文件逐字节一致，因此提交语义完全复现。

推理设备为 RTX 3090 24GB。两个 1280×1024 序列的 CUDA 峰值为：

```text
allocated=22.079 GiB
reserved=23.068 GiB
```
