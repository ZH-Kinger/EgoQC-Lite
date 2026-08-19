# QC 小模型训练数据合同

本合同面向“视觉语义质量评估学生模型”，不是 VLA action policy。格式、时间戳、SO(3)、
帧数和数值越界仍由确定性规则全量处理；模型只学习第二人手、严重遮挡、无意义抖动、
MANO overlay 漂移、任务文本错配和子任务边界等视觉语义。

## 1. 每条训练样本

一条样本是一个经过审核的短 clip，而不是一个随机视频窗口：

- 原视频只读 URI、对象版本/ETag 或 source revision；
- 唯一 `video_id`，以及 `clip_start_s/clip_end_s`；
- supplier、person/operator、collection session、scene、camera、task 标识；
- 4–8 秒 RGB clip，建议训练解码 8 FPS，保留原视频用于复核；
- 十三项多标签布尔 Gold label，允许只标本次明确判断的任务；
- reviewer、审核时间、标签规范版本；争议样本保存第二审核人和仲裁人；
- 可选 MANO mesh/skeleton overlay。`mano_overlay_drift` 没有 overlay 时必须缺失，不能填负例。

机器可读字段规范见 `config/qc_gold_label.schema.json`。当前 manifest 版本要求每个
`video_id` 对应一个审核窗口；需要从同一个长视频取多个窗口时，先产生带唯一 ID 的派生
clip 记录，仍不修改 raw。

```json
{
  "video_id": "supplier-a/session-07/episode-0042/clip-001",
  "source_revision": "etag-or-dataset-revision",
  "source_dataset": "supplier-a-batch-2026-08",
  "supplier_id": "supplier-a",
  "person_id": "person-019",
  "collection_session_id": "session-07",
  "scene_id": "kitchen-03",
  "camera_id": "headcam-model-a-02",
  "task_id": "put-cup-into-cabinet",
  "clip_start_s": 12.4,
  "clip_end_s": 18.4,
  "reviewer_id": "reviewer-12",
  "reviewed_at": "2026-08-18T08:00:00Z",
  "label_version": "egoqc-visual-gold-v1",
  "labels": {
    "persistent_extra_hands": false,
    "semantic_camera_shake": true,
    "severe_occlusion": false
  }
}
```

## 2. 标签来源

优先级固定为人工 Gold > 本地 VLM 教师 soft label > 程序化弱标签。教师和规则弱标签只
允许进入 train，validation/test 只允许人工 Gold。大模型离线批量跑一次并缓存模型、prompt、
输入 hash 与概率，不需要在生产 QC 时持续调用 API。

建议先做两阶段数据量：

| 阶段 | 数据要求 | 用途 |
|---|---:|---|
| 工程验证 | 每类至少 50 正例 + 50 负例 | 验证读取、训练、推理和基本可分性，不能宣称 99% |
| 生产训练 | train 每类 ≥500 有标签，其中 ≥100 人工 Gold | 训练 student，其余可来自教师/弱标签 |
| 阈值选择 | validation 每类 ≥100 Gold 正例 + ≥300 Gold 负例 | 选阈值、做概率校准 |
| 最终盲测 | test 每类 ≥200 Gold 正例 + ≥1000 Gold 负例 | 未参与训练和调参，报告 95% 置信区间 |

十三项是多标签，样本可同时覆盖多类，不能简单把各行数量相加。稀有错误应由规则候选、教师
高分样本和供应商失败案例主动挖掘；负例要包含“看起来相似但实际合格”的 hard negatives。

## 3. 划分和防泄漏

优先按 `person_id` 分组，其次 operator，再其次 collection session。相同 group 的所有片段
只能出现在一个 split。QC manifest 使用稳定 hash 分为 80% train / 10% validation /
10% test。`video_id`、原始 URI、相邻切片和派生版本不得跨 split。最终还应保留
至少一个完全未参与训练的供应商/相机组合做外部测试。

运行：

```bash
egoqc audit-qc-training \
  --manifest /path/to/qc-distillation.jsonl \
  --task-config config/visual_model_tasks.json \
  --output /path/to/qc-training-audit
```

输出 `qc-training-readiness.json` 和 `qc-training-blockers.jsonl`，检查类别覆盖、许可证治理、
人物/session 元数据、教师标签污染评估集以及 group/video/URI 跨 split 泄漏。

## 4. 99% 的正确口径

经验 precision=100% 不等于真实 precision≥99%。例如 60 个预测正例零误报，样本仍不足以
支持 99% 结论。`evaluate-qc-student` 同时计算经验 precision 和 95% Wilson 下界；只有下界
达到任务阈值才允许该任务自动拒收。

最终还需在完全独立的 test 上计算整条 pipeline 的 false accept、false reject、Micro/Macro-F1，
并按供应商、场景、人物、相机、遮挡强度切片。规则失败仍有最高权威，模型不得覆盖规则拒收。

## 5. 与 VLA 训练数据的区别

公开视频没有 MANO 也能用于 VLA 的视频表征、时序预测和视频—文本预训练；但不能用于机器人
action head。要训练动作策略，还需要同步 RGB、相机内外参、手腕/MANO 或目标机器人关节动作、
state mask、task/subtask、统一 timestamp 和来源许可。缺少的监督必须用 loss mask 关闭，不能
由默认值伪造成 Ground Truth。
