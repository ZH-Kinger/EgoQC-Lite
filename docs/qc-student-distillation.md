# EgoQC few-B 视觉质量模型与高速级联

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
标签 mask、加权 BCE、GPU backward 和 checkpoint；它不是最终几 B 模型。只有 Gold validation 达到每类规定的
precision、完成概率校准和跨供应商测试后，才能为某个 task 单独开放自动拒收。

机器可读约束见
[`config/qc_student_deployment_v1.json`](../config/qc_student_deployment_v1.json)。最终语义质检模型
以 Qwen3-VL-4B 为首选，2B 是更低延迟候选，8B 是效果上限 challenger；2B/4B 优先做全参数
SFT，LoRA/QLoRA 只进入适配成本对照。三者使用同一冻结 split、同一 8 帧×448 最大边输入、
同一结构化输出和同一评估协议，避免用不同采样预算伪造参数规模收益。

16M–24M 的 MobileNetV4 + Temporal Shift + depthwise TCN 仍保留，但只作为全量高速 scout：
它在每个 clip 上运行并输出风险/不确定度，将预计 1%–10% 的高风险、分歧和随机抽检样本送给
2B/4B VLM。最终模型输出多标签概率、问题时间段、严重度、证据帧与 `abstain`，自由文本解释
只作为人工证据，不能成为验收真值。百万小时规模默认禁止让几 B 模型扫描所有 clip。

CPU 路径采用 INT8 scout，并测量 INT4 2B/4B 作为低频复核器；GPU 路径测量 BF16/FP16
2B/4B/8B。CPU INT4 是否达到可接受速度、内存和精度目前没有实测，配置中的相关数字保持空值，
不能写入论文结论。原始 720p/1080p 始终只读保存，在线 letterbox/抽帧，不额外复制低分辨率视频。

模型规模不以 checkpoint 文件名判断。每次发布必须同时记录参数量、INT4/INT8/BF16 体积、
峰值 RSS/显存、含视频解码的 P50/P95 延迟和 video-hours/wall-hour。若 INT8 相比 FP32 在
同一阈值口径下 recall 下降超过 1 个百分点，CPU 自动决策保持关闭；若 CPU/GPU 决策分歧
在相同 canonical 输入上超过 0.1%，先修复预处理、量化或算子差异，不能分别调阈值掩盖
运行时不一致。CPU 低分辨率 profile 与 GPU 高分辨率 profile 的效果差异则单独报告。

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
