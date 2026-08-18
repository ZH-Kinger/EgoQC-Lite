# 自动候选 Clip 选择

`plan-qc-clips` 读取已有 `scan` 产出的 `episodes.jsonl` 和 `bad_frames.jsonl`，自动完成：

- 相邻坏帧合并；
- 生成默认 4–8 秒、带上下文的固定 clip；
- 正常随机对照采样；
- 聚合 MP4 episode offset 转换；
- 纯数值/格式问题不进入视觉模型队列；
- 视觉候选进行固定标签 + 开放世界质量审查；
- 输出可恢复、可审计的 JSONL 队列，不复制原视频、不保存 API 密钥。

```bash
egoqc plan-qc-clips /mnt/datasets/example \
  --quality-root /mnt/workspace/quality/example \
  --task-config config/visual_model_tasks.json \
  --output /mnt/workspace/clip-plans/example \
  --source-dataset example \
  --supplier-id supplier-a
```

输出：

- `clip-candidates.jsonl`：可直接进入训练/人工复检的数据视图；
- `teacher-api-queue.jsonl`：视觉教师请求任务；
- `summary.json`：候选数、模型请求数、来源和任务分布。

`trigger_tasks` 只说明规则为什么召回该 clip。教师实际需要检查配置中的所有固定任务、
`assessment_dimensions` 和开放式 `findings`，因此不会被基础规则限制。开放发现经人工归并后，
再决定是否升级为新的稳定 taxonomy 和学生模型 head。

接入具体 API 前需要确定：兼容协议、base URL、模型名、视频或多帧图片输入限制、并发/QPS、
数据留存策略，以及 API key 的环境变量名。密钥只能由开发机环境或 secret manager 注入。
