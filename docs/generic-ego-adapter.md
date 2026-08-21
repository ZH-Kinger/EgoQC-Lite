# 通用 Ego 原始数据适配

EgoQC-Lite 不要求输入已经是 LeRobot、MANO 或完成 SLAM。最小输入是一段可读取的视频；缺少
MANO、内参或轨迹只会关闭依赖这些模态的指标，不会把视频本身判为不合格。

## 能力分层

| 输入能力 | 可以执行 | 必须留空或等待补齐 |
|---|---|---|
| 只有视频 | 解码、FPS/分辨率、稀疏画质、冻结、视觉抖动、手可见性模型、视频自监督训练 | MPJPE、ATE、MANO 投影 |
| 视频 + task | 上述全部、任务文本匹配、视频文本训练 | subtask 边界、几何指标 |
| 视频 + 内参/畸变 | 投影模型检查、按标定去畸变 | ATE、MANO 精度仍不可声明 |
| 视频 + 相机轨迹 | 相机运动、轨迹抖动；有 GT 时才能算 ATE | 无 GT 时 ATE 留空 |
| 视频 + MANO/手部轨迹 | 手部运动学、抖动、mesh 可视化；有内参可回投影 | 无人工/独立 GT 时 MPJPE 留空 |
| 完整同步数据 | 全部适用指标 | — |

通过模型补出的内参、VSLAM、关键点和 MANO 都标为 Silver prediction，不能冒充供应商交付的
Ground Truth。通用 sidecar 中即使存在数组，正式训练前也必须经过针对该来源的 canonical
normalizer 验证形状、单位、坐标系和时间戳；未经规范化的数值目标会保持 loss mask=0。

每条输出记录会自动增加三组分类字段：

- `capability_class`：区分 `rgb_only`、`rgb_text`、`rgb_mano`、
  `rgb_calibrated_mano` 等输入能力；
- `task_taxonomy`：从 task、activity、description 等已有文本提取交互原语、物体可供性、
  粗细操作、单双手倾向和时序复杂度；无法可靠归类时写 `unknown` 并进入语义复核；
- `annotation_provenance`：逐项区分来源观测、来源标注、模型派生预测、人工批准的 Silver
  预测和真正 Ground Truth。模型补算的 MANO 永远不会被写成 Ground Truth。

## 目录与可选 sidecar

```text
raw_ego/
├── kitchen/
│   ├── episode-001.mp4
│   └── episode-001.json   # 可选，与视频同名
└── warehouse/
    └── episode-002.mov
```

sidecar 可以只提供已有字段：

```json
{
  "task_label": "place the cup on the table",
  "supplier_id": "supplier-a",
  "person_id": "person-019",
  "collection_session_id": "session-07",
  "scene_id": "kitchen-03",
  "camera_id": "headcam-a",
  "intrinsics": [1000, 0, 640, 0, 1000, 360, 0, 0, 1]
}
```

建议格式见 `config/generic_ego_sidecar.schema.json`。额外字段会被保留，便于后续添加来源专用
adapter。

## 单视频检查

```bash
egoqc inspect-adapter /path/to/episode-001.mp4 --video-check header
```

输出 `capabilities`、`capability_route`、当前可运行阶段、不可用指标和安全补齐候选。

同时输出 `use_case_eligibility`，对以下用途分别给出 `ready/partial/blocked`：

- 视频自监督、视频文本和视觉 QC 训练；
- 手部重建推理与有监督训练；
- VLA observation 预训练和机器人模仿学习；
- 多相机、双目深度、眼动、VSLAM；
- 遥操作手套与触觉学习；
- 供应商验收和隐私审查后的公开发布。

扩展质检任务还包括非 ego 视角、相机佩戴失效、隐私暴露、视频剪辑/回放、近重复 episode、
无有效手物交互、多相机不同步、机器人 action 不同步、手套掉信号和 IMU/视觉不一致。它们定义在
`config/qc_extension_tasks.json`，在拥有独立 Gold 和校准前只允许进入人工复检，不能自动拒收。

## 批量生成只读 manifest

```bash
egoqc build-generic-ego-views /mnt/data/raw-ego \
  --source-dataset supplier-a-2026-08 \
  --source-class supplier_dataset \
  --license-id internal-approval-001 \
  --workers 32 \
  --video-check header \
  --output /mnt/workspace/egoqc/generic/supplier-a-2026-08
```

产物：

- `generic-ego.jsonl`：canonical 训练与质检 manifest；
- `generic-ego.parquet`：供大规模 SQL/Spark/Ray 扫描；
- `errors.jsonl`：无法探测的文件；
- `summary.json`：能力、训练目标和错误统计。

输出目录禁止位于输入树内部；`/mnt/data` 源数据不会被修改。未提供 `license-id` 时仍生成
technical candidate，但 `training_ready=false`。

百万小时场景默认使用 `header`，不解码像素；随后只对候选运行 `sample-quality` 或模型阶段。
