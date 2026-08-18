# EgoQC 视觉质量学生模型

格式、schema、时间戳、SO(3)、帧数、FPS、分辨率等确定性失败始终由规则判定，模型不能
覆盖规则失败。学生模型仅处理难以公式化的视觉语义，并且在 Gold Set 校准前只能路由到
人工复检，不能自动拒收。

## 标签优先级

同一个 `video_id/task` 按以下优先级覆盖：

1. 人工 Gold Set，权重 1.0；
2. 本地 VLM 教师 soft label，权重 `0.5 × confidence`；
3. 程序化检测器弱标签，权重 `0.25 × confidence`。

教师标签固定为 `<teacher-root>/<video_id>/teacher-label.json`，schema：

```json
{
  "schema_version": "egoqc-visual-teacher-v1",
  "teacher_model": "Qwen3-VL-8B-Instruct",
  "prompt_version": "egoqc-teacher-v1",
  "tasks": {
    "severe_occlusion": {"probability": 0.82, "confidence": 0.74},
    "semantic_camera_shake": {"probability": 0.11, "confidence": 0.91}
  }
}
```

教师输出必须缓存并携带模型和 prompt 版本。自由文本理由只能作为证据，不能直接作为
训练 target 或采购验收结论。
完整的 clip-level Gold 字段、数量和切分要求见
`docs/qc-training-data-contract.md`。

## 构建与工程 smoke

```bash
egoqc build-qc-distillation \
  --records /path/to/all-records.jsonl \
  --task-config config/visual_model_tasks.json \
  --hand-screen-root /path/to/hand-screen \
  --teacher-root /path/to/teacher-labels \
  --gold-labels /path/to/human-gold.jsonl \
  --output /path/to/qc-distillation

egoqc audit-qc-training \
  --manifest /path/to/qc-distillation/qc-distillation.jsonl \
  --task-config config/visual_model_tasks.json \
  --output /path/to/qc-training-audit

egoqc smoke-qc-student \
  --manifest /path/to/qc-distillation/qc-distillation.jsonl \
  --output /path/to/qc-student-smoke \
  --steps 20 --batch-size 4 --device cuda
```

当前 smoke student 是轻量 CNN + 双向 GRU + 多标签 head，只验证视频读取、soft target、
标签 mask、加权 BCE、GPU backward 和 checkpoint。只有 Gold validation 达到每类规定的
precision、完成概率校准和跨供应商测试后，才能为某个 task 单独开放自动拒收。

## Gold Set 门禁

学生批量推理输出 JSONL：`{"video_id":"...","probabilities":{"severe_occlusion":0.8}}`。
然后运行：

```bash
egoqc evaluate-qc-student \
  --predictions /path/to/predictions.jsonl \
  --gold-labels /path/to/human-gold.jsonl \
  --task-config config/visual_model_tasks.json \
  --output /path/to/evaluation
```

每个任务先检查 Gold 正负例覆盖，再搜索满足 taxonomy precision 的阈值。
经验 precision 达标仍不足够；其 95% Wilson 下界也必须达标，否则该任务
`auto_reject_enabled=false`。报告同时输出 Brier score 和 10-bin ECE。生产上线还应
分供应商、场景、人物和相机型号做切片验证。
