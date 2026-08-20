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
  --steps 20 --batch-size 4 --device cuda \
  --image-size 192 --temporal-stride 4
```

当前 smoke student 是轻量 CNN + 双向 GRU + 多标签 head，只验证视频读取、soft target、
标签 mask、加权 BCE、GPU backward 和 checkpoint。只有 Gold validation 达到每类规定的
precision、完成概率校准和跨供应商测试后，才能为某个 task 单独开放自动拒收。

生产部署只维护一套 student 权重，机器可读约束见
[`config/qc_student_deployment_v1.json`](../config/qc_student_deployment_v1.json)。CPU 使用 INT8
ONNX Runtime/OpenVINO，GPU 使用 FP16 PyTorch/ONNX Runtime/TensorRT；二者必须在同一 Gold
集合上做输出一致性和决策一致性测试。CPU 默认 16×160 全局帧和 16×128 手部 ROI，GPU
默认 32×224 全局帧和 32×160 手部 ROI。输入分辨率和帧数可以不同，但 taxonomy、权重、
概率校准口径和拒答策略必须一致。

QC student 不训练 8B 级通用大模型。默认输入从原始视频在线缩放为 192×192，并采用
letterbox 保留完整第一视角画面，避免方形中心裁剪删除画面边缘或底部的手；每 4 个已解码帧
取一帧进入时序层。原始 720p/1080p 始终只读保存，不额外存一份低分辨率视频。生产模型优先
采用不超过 8M 参数的 MobileNetV3-Large-0.75 + Temporal Shift + depthwise TCN 作为首选，
MoViNet-A0-Stream 作为 challenger；先冻结编码器训练 head，效果不足再
逐层解冻。MANO 数值指标、速度、相机运动和规则事件作为低维特征融合，不让视觉网络重复
学习代码已经能精确计算的内容。

模型“小”不以 checkpoint 文件名判断。每次发布必须同时记录参数量、INT8/FP16 体积、
峰值 RSS/显存、含视频解码的 P50/P95 延迟和 video-hours/wall-hour。若 INT8 相比 FP32 在
同一阈值口径下 recall 下降超过 1 个百分点，CPU 自动决策保持关闭；若 CPU/GPU 决策分歧
超过 0.1%，先修复预处理、量化或算子差异，不能分别调阈值掩盖不一致。

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
