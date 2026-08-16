# RekaDaily 双训练视图

RekaDaily raw 只有视频和粗粒度元数据，不能直接伪装成满足 EgoScale / LeRobot 3.0
手部动作标准的数据。本管线将同一来源拆成两个用途，并且从不修改 raw：

1. `video-pretrain`：用于视频表征、时序理解或弱监督预训练。只做廉价元数据门检，
   检查帧率、短边、时长和本地是否已经物化。MOV 不直接拒绝，而是标为需要派生转码。
2. `mano-silver`：严格的阶段状态机。视频门检通过后，还必须依次通过手部检测预筛、
   MANO 拟合产物完整性检查和人工对齐复检，才会进入 Silver ready。

## 最小运行方式

只针对当前已经下载到开发机的 loose video 和 tar shard 建立视图：

```bash
egoqc build-rekadaily-views /path/to/RekaDaily-10k-raw \
  --output /path/to/quality/rekadaily-views \
  --materialized-only
```

如果已经有手部预筛结果：

```bash
egoqc build-rekadaily-views /path/to/RekaDaily-10k-raw \
  --output /path/to/quality/rekadaily-views \
  --materialized-only \
  --hand-screen-root /path/to/hand-screen
```

许可证或内部用途审批未记录时，技术合格样本只进入
`video-pretrain-candidates.jsonl`，不会进入 `video-pretrain-ready.jsonl`。审批完成后应传递可追溯编号：

```bash
egoqc build-rekadaily-views /path/to/RekaDaily-10k-raw \
  --output /path/to/quality/rekadaily-views \
  --materialized-only \
  --license-id 'legal-ticket-or-license-version'
```

## MANO Silver 输入契约

手部预筛复用 `screen-rekadaily-hands` 的输出：

```text
<hand-screen-root>/<video_id>/hand-screen.json
```

MANO 拟合器应写入：

```text
<mano-root>/<video_id>/mano-fit.json
```

最小内容：

```json
{
  "status": "succeeded",
  "capabilities": {
    "wrist_pose": true,
    "mano_pose": true,
    "betas": true,
    "state_mask": true
  },
  "output_uri": "s3://derived-bucket/mano/video-id/"
}
```

人工复检写入：

```text
<alignment-root>/<video_id>/alignment-qc.json
```

最小内容：

```json
{
  "decision": "accept",
  "human_reviewed": true,
  "reviewer_id": "reviewer-001"
}
```

然后重新执行相同命令并加入 `--mano-root`、`--alignment-root`。构建是幂等的；
原始视频保持只读，输出 manifest 可以随新下载 shard 和新审核结果滚动重建。
`materialized-inventory-cache.json` 按 tar 的路径、大小和 mtime 复用成员索引；未变化的
大 shard 不会在每次更新时重新遍历，只有新增或变化的 shard 会被扫描。

## 输出

- `all-records.jsonl`：完整、嵌套且带 provenance 的权威记录。
- `all-records.parquet`：方便 DuckDB / Spark / PostgreSQL 导入的扁平查询视图。
- `video-pretrain-candidates.jsonl`：技术合格候选。
- `video-pretrain-ready.jsonl`：技术与许可证治理均通过。
- `video-blocked.jsonl`：不合格原因明确的记录。
- `hand-screen-queue.jsonl`：待 GPU 低频手检测。
- `hand-review-queue.jsonl`：待人工判断多人手、长时离画等边界样本。
- `mano-fit-queue.jsonl`：待 MANO / HaWoR 拟合。
- `alignment-review-queue.jsonl`：拟合完成，待骨骼/mesh 叠加复检。
- `mano-silver-ready.jsonl`：所有阶段均通过且治理已批准。
- `vla-pretrain-candidates.jsonl`：带 objective、loss mask 和稳定数据划分的技术候选。
- `vla-pretrain-ready.jsonl`：许可证治理通过后可被训练作业直接消费的 VLA 预训练视图。
- `summary.json`：数量、小时、失败原因、各阶段队列规模与代码/索引版本。

对于数百 TB 数据，关键成本控制是：先读约数 MB 的 Parquet 元数据，再只对通过门检的
已下载视频运行低帧率手检测；MANO 和人工复检只消费进一步缩小后的队列。tar 只扫描成员
头部建立索引，不解包、不复制视频负载。

## 无 MANO 视频的 VLA 训练契约

每条技术合格视频都会获得 `vla_pretraining` 字段。无 MANO 的样本允许进入
`video_representation`、`temporal_prediction`；存在粗文本时再进入
`video_text_alignment`。所有缺失目标都用显式 mask 隔离：

```json
{
  "loss_masks": {
    "video_representation": 1,
    "temporal_prediction": 1,
    "video_text_alignment": 1,
    "mano_motion": 0,
    "robot_action": 0,
    "camera_pose": 0,
    "tactile": 0
  }
}
```

数据划分由 `sha256(video_id)` 固定为 95%/2.5%/2.5%，新增 shard 不会导致旧样本
重新分组。由于上游没有 session id，记录会保留 `source_session_id_missing` 警告；
后续若取得采集 session 标识，应改为按 session 分组，避免相邻片段跨 train/test 泄漏。

## 数据加载器

`VLAPretrainDataset` 可直接读取 JSONL，支持 loose video 与 `tar://...!/member`，按契约
抽取 4 秒、8 FPS 窗口并输出 `[T,H,W,C]` uint8 视频、文本、objective 和 loss mask。
正式使用默认只读取 `training_ready=true`。许可证审核前做管线 smoke test 时，必须显式写：

```bash
egoqc smoke-vla-loader \
  --manifest /path/to/vla-pretrain-candidates.jsonl \
  --output /path/to/vla-loader-smoke \
  --batch-size 2 \
  --allow-technical-candidates
```

该开关只允许验证解码与 batch，不会把样本状态改成已授权。输出 contact sheet 可用于快速
检查裁剪、时间窗口和文本对应关系。训练框架只需将 numpy batch 转换成 Tensor；所有 loss
必须逐项乘以 `loss_masks`。

GPU 训练链路可以用轻量模型验证真实前向、反向、梯度和 checkpoint 写入：

```bash
egoqc smoke-vla-train \
  --manifest /path/to/vla-pretrain-candidates.jsonl \
  --output /path/to/vla-train-smoke \
  --batch-size 2 --steps 5 --device cuda \
  --allow-technical-candidates
```

该命令训练视频增强对比、前半段到后半段的时序预测、视频—文本 InfoNCE。模型故意保持
很小，只用于验证数据、loss mask、GPU backward 和 checkpoint，不是生产预训练模型，
也没有 MANO 或 robot action head。
