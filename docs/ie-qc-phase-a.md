# IE-QC Phase A：可控干预与证据响应

Phase A 只回答一个问题：**对已知的数据/标注通道破坏，现有证据源是否产生稳定、可定位、
随强度不减弱的响应，并且这种响应能否预测真实错误上的专家可靠性？**

它不是新的供应商验收入口，也不把合成错误当成人工 Gold。

## 当前实现范围

首版面向具有完整数值标注的 LeRobot v3 episode，覆盖七个干预族：

| 干预族 | 被改变的通道 | 主要预期证据 |
|---|---|---|
| `timestamp_offset` | timestamp | 名义时间轴误差、帧间隔 jitter |
| `wrist_position_offset` | `observation.state` wrist | 世界系→相机系位置一致性 |
| `camera_translation_spike` | `extrinsics_w2c` | 相机 jitter、手/相机一致性 |
| `pose_representation_offset` | state 中的关节 Euler | rotmat↔Euler 表示一致性 |
| `state_mask_dropout` | `state_mask` | kept/mask 一致性、短时丢失 |
| `world_translation_scale` | 世界系 wrist translation | 坐标尺度/位置一致性 |
| `beta_drift` | state 中的 MANO betas | episode 内 shape 恒定性 |

每个干预默认包含 `low/high` 两个强度。参数全部位于
[`config/qc_interventions_phase_a.json`](../config/qc_interventions_phase_a.json)，不硬编码进实验脚本。

视频模糊、遮挡、镜头畸变、冻结帧等视觉干预已有在线 augmentation，但本阶段暂不与数值
干预混合。下一步应让视觉教师在同一 evidence-delta 契约上运行，再做跨模态实验。

## 只读与可复现保证

- 输出目录如果位于原始 dataset 内，程序直接拒绝运行；
- plan 只记录相对 shard 路径，不写入数据根绝对路径；
- 干预在内存中的 Arrow table 上执行，不生成或覆盖原始 Parquet；
- plan 保存完整 `meta/` 指纹以及源 shard 的 size/mtime；运行前不一致即停止；
- “恢复”虚拟视图只需丢弃内存对象并重新读取源数据；
- 合成标签只允许研究训练或校准，不得改变供应商 `accept/reject/rework`。

这套机制不会读取或写回 OSS/CPFS 源对象的内容；在挂载目录上运行时仍建议把 raw 挂成只读，
把 plan/run 输出放在 `/mnt/workspace`。

## 使用

先在少量 episode 上规划。默认按 seed 稳定选择最多 32 条；重复运行不会因目录新增而随机漂移：

```bash
egoqc plan-qc-interventions /mnt/data/readonly/lerobot-v3 \
  --intervention-config config/qc_interventions_phase_a.json \
  --maximum-episodes 16 \
  --output /mnt/workspace/ie-qc/phase-a/plan
```

也可以固定 episode 或只运行部分错误族：

```bash
egoqc plan-qc-interventions /mnt/data/readonly/lerobot-v3 \
  --episode 12 --episode 48 \
  --family timestamp_offset \
  --family camera_translation_spike \
  --output /mnt/workspace/ie-qc/phase-a/plan-fixed
```

随后运行现有 QC 专家并记录干预前后的响应：

```bash
egoqc run-qc-interventions /mnt/data/readonly/lerobot-v3 \
  --manifest /mnt/workspace/ie-qc/phase-a/plan/interventions.jsonl \
  --config config/default.json \
  --output /mnt/workspace/ie-qc/phase-a/evidence
```

小规模冒烟可以增加 `--maximum-interventions 10`。

## 输出契约

| 文件 | 粒度 | 用途 |
|---|---|---|
| `baseline-evidence.jsonl` | episode | 保存未干预基线，不把原本存在的问题误算为干预响应 |
| `sample-plan.jsonl` | episode | 原始基线的抽检帧，可直接交给现有 evidence/VLC 工作台 |
| `intervention-runs.jsonl` | episode × family × level | 目标命中、区间定位、tier 和新增 issue |
| `evidence-deltas.jsonl` | intervention × expert | 训练条件可靠性模型的长表 |
| `monotonicity.jsonl` | episode × family | 比较 high 响应是否不低于 low |
| `summary.json` | run | 按错误族汇总，仅用于实验诊断 |

`evidence-deltas.jsonl` 的关键字段为：

```json
{
  "intervention_id": "...",
  "episode_index": 12,
  "family": "timestamp_offset",
  "level": "high",
  "expert": "metric:timestamp_max_error_s",
  "expected_target": true,
  "baseline": 0.0,
  "intervened": 0.025,
  "delta": 0.025,
  "changed": true
}
```

这里的 `target_hit_rate` 只说明专家对合成破坏有反应，**不等于真实错误 recall/precision**。
`target_event_interval_precision` 也只衡量坏帧是否落在已知干预区间，不是通用定位精度。

需要人工检查原始基线时，直接复用现有可视化，不对合成干预作伪影像：

```bash
egoqc extract-samples /mnt/data/readonly/lerobot-v3 \
  --plan /mnt/workspace/ie-qc/phase-a/evidence/sample-plan.jsonl \
  --output /mnt/workspace/ie-qc/phase-a/review
```

输出包含 contact sheet、`review.html` 和 `episodes-vlc.xspf`。这里展示的是原始 episode，
用于判断 baseline 的规则报警是否真实；干预区间和 evidence delta 仍通过 JSONL 对齐。

## Phase A 的真实停止条件

完成代码冒烟后，还需要建立真实 Gold 对照：

1. 从供应商数据、公开原始 ego 数据和已有清洗数据中分别抽取真实错误；
2. 两名审核员独立标注错误类型与时间区间，冲突由第三人裁决；
3. 按 person/session/supplier 切分，禁止同源相邻片段跨 train/test；
4. 在 train/calibration 上用 intervention delta 训练简单 reliability 模型；
5. 在完全隔离的真实 test 上比较 fixed vote、stacking 和 Snorkel-style label model；
6. 报告 AUROC/AUPRC、segment IoU、ECE、worst-group，以及 bootstrap 置信区间。

如果 intervention response 对真实专家正确性的预测不能稳定超过普通 stacking，停止 IE-QC
复杂图模型路线。此时保留虚拟干预作为回归测试和规则覆盖率工具，但不把它写成论文核心贡献。
