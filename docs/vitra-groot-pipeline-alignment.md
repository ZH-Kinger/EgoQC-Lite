# VITRA → EgoQC → GR00T 数据管线对齐规范

状态：`draft-0.1`
更新日期：2026-08-05

## 1. 结论

采用双格式、单一事实源架构：

```text
对象存储原始视频 + HaWoR .pth（只读）
                  │
                  ▼
       Canonical LeRobot v3 大 shard
       多 episode / Parquet、MP4
                  │
                  ├── EgoQC 结构、几何、时序、MANO 质量门禁
                  ├── derived 原子动作/清洗视图
                  └── GR00T export：LeRobot v2 + modality.json
```

- LeRobot v3 是百 TB 数据管理、增量 QC 和长期归档格式。
- GR00T 当前主线使用 LeRobot v2 变体，因此按训练 run 导出 v2 视图，不把主存降级为每 episode 一个文件。
- 原始视频、`.pth`、Canonical v3 都不可原地修改。
- 平滑、插值、切段、retargeting 和语言标签都是带版本的 derived artifact。
- 在目标手 URDF、20 个关节顺序、旋转轴、零位和限位冻结前，不生成伪 20-DOF 标签。

## 2. 已发现的格式冲突与决策

| 项目 | 现有 EgoQC v3 标准 | 输入文档/训练设想 | 决策 |
|---|---|---|---|
| 手部姿态 | 每手 15×SO(3)，另有 Euler state 和 betas | `.pth` 每手 45 axis-angle | ingest 时保留原值并转成 canonical rotmat；禁止只保留一种有损表示 |
| state | 122：每手 wrist 6 + pose 45 + betas 10 | 108：每手 xyz 3 + rot6d 6 + pose 45 | 122 留在 canonical QC；108 只作为 `mano108` 训练导出 profile |
| action | `info.json` 声明 102，数据未落盘 | `action[t] = state[t+1]`，rot6d 后应为 108 | canonical 不伪造 action；训练导出写真实 108 action |
| Parquet/MP4 | v3 多 episode 聚合大文件 | GR00T v2 每 episode 一个文件 | 主存 v3；导出 v2，不冲突 |
| invalid frame | mask 保留时间轴 | 过滤 invalid | 禁止直接删散点帧；按连续有效区间切为新 episode |
| relative action | 未定义 | 绝对值落盘、processor 转 relative | 保留绝对 action；relative 由训练配置完成，并保存配置 hash |
| 20-DOF | 当前没有该字段 | 需接入目标手 retargeting | 通用 URDF/关节/mesh 工具已实现；作为独立 `robot20` profile |

`action=102` 仅适用于 wrist Euler 3 维旋转；一旦 wrist 改为 rot6d，双手维度为：

```text
2 × (xyz 3 + rot6d 6 + MANO pose 45) = 108
```

因此不能继续声明 102。

## 3. 分阶段数据契约

### 3.1 Raw source

```text
video.mp4
pose.pth
├── pred_trans      float[2,T,3]   # world, metre
├── pred_rot        float[2,T,3]   # axis-angle, radian
├── pred_hand_pose  float[2,T,45]  # 15 local axis-angle
├── pred_betas      float[2,T,10]
└── pred_valid      bool[2,T]
```

必须额外记录：source、原始 URI、对象版本/ETag、SHA-256（可选）、FPS、视频帧数、HaWoR commit、相机/深度/SLAM 模型版本和命令行参数。

### 3.2 Canonical LeRobot v3

继续使用 EgoQC 当前 122 维 state 和独立 world rotmat 字段。原因：它能够无歧义执行 SO(3)、左右手镜像、world↔camera 和 MANO 重投影检查。

Canonical v3 不承担特定训练框架的 action 语义。训练 action 由 export profile 生成。

### 3.3 `mano108` GR00T export

每个有效区间有 `K` 个连续原始状态时：

```text
rows/video frames = K - 1
observation.state[t] = state[source_start + t]
action[t]            = state[source_start + t + 1]
next.done[-1]        = true
```

即导出视频也只使用当前观测帧 `[start, end)`，最后一个源状态只作为最后一行动作目标，避免 video/data length 不一致。

建议 `modality.json` 切片：

```json
{
  "state": {
    "left_wrist_xyz": {"start": 0, "end": 3},
    "left_wrist_rot6d": {"start": 3, "end": 9},
    "left_mano_pose": {"start": 9, "end": 54},
    "right_wrist_xyz": {"start": 54, "end": 57},
    "right_wrist_rot6d": {"start": 57, "end": 63},
    "right_mano_pose": {"start": 63, "end": 108}
  },
  "action": {
    "left_wrist_xyz": {"start": 0, "end": 3},
    "left_wrist_rot6d": {"start": 3, "end": 9},
    "left_mano_pose": {"start": 9, "end": 54},
    "right_wrist_xyz": {"start": 54, "end": 57},
    "right_wrist_rot6d": {"start": 57, "end": 63},
    "right_mano_pose": {"start": 63, "end": 108}
  },
  "video": {
    "ego_view": {"original_key": "observation.images.ego"}
  },
  "annotation": {"human.task_description": {}}
}
```

注意：rot6d 是表示，不是六个独立关节角。必须在 profile 中固定“取旋转矩阵前两列还是前两行、展平顺序、逆变换方法”，并用单位阵和 90° 绕轴测试向量做 round-trip。没有通过 round-trip 的导出不得训练。

`relative` 不允许对 rot6d 六个数直接做减法；必须由 GR00T data config 的旋转变换计算相对旋转。训练 run 保存 `modality.json`、data config 和 processor commit。

### 3.4 `robot20` export

参考 Robot20 profile 按每手 20 个独立 revolute joint 设计。按人手和目标 URDF 掌根几何冻结：

```text
finger1 = thumb
finger2 = index
finger3 = middle
finger4 = ring
finger5 = pinky
```

MANO 使用 OpenPose 21-joint 顺序：wrist 0，thumb 1–4，index 5–8，middle
9–12，ring 13–16，pinky 17–20。retargeting 使用 MANO21 关节点/骨段目标优化
Robot20 FK，不把 MANO 的 15×SO(3) 旋转直接复制成 20 个标量角。

目标格式在以下其余材料齐全后冻结：

- 左右手 mesh（已接入官方 STL，并与当前 URDF 哈希匹配）；
- 目标手 retargeting 实现的 commit 和配置；

输出必须同时保存 `q[20]` 和 FK 校验指标：指尖误差、骨段方向误差、限位违反、自碰撞、时间连续性。不能只保存优化器的 q。

## 4. 清洗与切段规则

### 4.1 Invalid mask

- 不允许从一条 episode 中直接删除零散 invalid 行，否则 timestamp、视频和 action 全部错位。
- 对左右手要求按用途配置：`both_valid`、`main_hand_valid` 或 `either_valid`。
- 先求连续有效区间，再丢弃短于阈值的区间。
- 每个新 episode 保存 `source_episode_index`、`source_from_frame`、`source_to_frame`。
- 1–3 帧短缺失只在 derived clean view 插值，raw/canonical 保留原 mask。

### 4.2 抖动与平滑

顺序固定为：

```text
原始轨迹 QC → 异常片段标注 → 决定 allowed use → derived 平滑
```

禁止先做样条平滑再评估，否则会掩盖跳点、SLAM 抖动和时间错位。

- 位置可在短缺失段使用线性/样条，但必须限制最大 gap。
- wrist 和 joint rotation 必须在 SO(3)/quaternion 上处理，不能直接平滑 axis-angle 或 Euler 每个分量。
- 每个 clean artifact 保存滤波器、窗口、参数、输入指纹和输出版本。

### 4.3 原子动作分割

速度极小值可以生成候选边界，但不直接改 raw：

- 无效帧保持 NaN，分段内插值只用于边界检测；
- 左右手速度分别计算，再根据 main hand/task 组合；
- 使用 prominence、最小边界间隔和最小 segment 时长抑制过切；
- 输出 segment manifest，训练视图按 manifest 虚拟/物化切片；
- 保存 speed curve 与边界证据。

## 5. 视频脏段检测改进

现有“每 5 帧 YOLO person 面积加权重心、距离大于 350 px”只能作为候选器，不能单独决定删除：

- 350 px 与分辨率和 FOV 强耦合，应改为图像对角线归一化距离；
- egocentric 中 person box 会随手臂进入画面、遮挡和他人经过而变化；
- 每 5 帧可能漏掉短切镜；
- person 重心变化无法区分正常快速转头与镜头切换；
- 固定扩展 1 秒可能删除有效操作。

建议低成本融合三类信号：

1. HSV/embedding 帧差检测硬切镜；
2. 光流内点率、全局仿射/单应残差检测异常相机运动；
3. person box 归一化重心、数量和面积变化作为辅助。

只有多信号一致或人工确认后才进入 bad segment。输出 timeline、preview 和 manifest，不改原视频。

## 6. 相机内参与世界坐标

- GeoCalib 作为当前首选候选，但不能以十余张图的误差直接推广到所有来源。
- 对每个 shot 均匀抽帧估计 focal，保存中位数、MAD、p05/p95 和失败率。
- 焦距随时间显著漂移时，标记 `intrinsics_unstable`，不强行取单值。
- MegaSAM、MoGe-2、GeoCalib 的版本、权重 hash、输入分辨率和 resize/crop 必须进入 provenance。
- world pose QC 在平滑前执行，并区分 camera common-mode jitter 与单手 tracking jitter。

## 7. 语言标注成本控制

- 优先使用数据集原始 task/clip metadata。
- 无标签视频先按原子 segment 去重和聚类，再对 cluster representative 调用 Qwen/豆包。
- 大模型标签保存 model、prompt、temperature、时间和输入片段 hash。
- 低置信或任务过于笼统的标签进入人工队列；生产训练不依赖在线 API。

## 8. 质量门禁

### Ingest gate

- 视频可解码、FPS/帧数可确定；
- `.pth` 五个字段存在且 shape 一致；
- `T_pose` 与 `T_video` 差异在策略允许范围；
- axis-angle、translation、betas 有限；
- source identity/provenance 完整。

### Canonical gate

- rotmat 属于 SO(3)；
- left/right、world/camera、MANO J0 与单位约定一致；
- mask、timestamp、episode route 一致；
- Temporal QC 无严重 spike/freeze/flicker；
- MANO 抽样重投影通过。

### GR00T export gate

- v2 目录、`modality.json`、annotation 三处 key 一致；
- state/action 都是 float32 且维度与 modality slice 一致；
- 每行 action 严格等于下一源状态；
- data/video/episode length 一致；
- rot6d round-trip 通过；
- `stats.json`、`relative_stats.json` 从最终训练视图计算；
- train/val/test 按 source/person/scene 隔离后再统计。

### Robot20 gate

- URDF joint order 与数组顺序一致；
- 所有 q 在 limit 内；
- FK 指尖与 MANO target 对齐；
- 无明显自碰撞、速度/加速度跳点和控制饱和；
- retarget config、URDF、solver 都有 hash。

## 9. 执行顺序

1. 实现 `.pth` metadata-only inspector，不加载视频帧。
2. 实现连续 valid span 规划器和 source frame lineage。
3. 实现 `.pth` → canonical v3 ingest，复用 EgoQC v0.4。
4. 冻结 rot6d convention，增加 round-trip/shift-action 测试。
5. 实现 `mano108` v3 → GR00T v2 export 及 export gate。
6. 用 3 个来源各 100–300 episode 校准 GeoCalib、脏段、抖动阈值。
7. 在已完成的 URDF/FK/mesh 渲染基础上实现 `robot20` retargeting 与逐帧 QC。
8. 最后再物化大批量训练视图，不直接全量跑百 TB。

## 10. 当前阻塞项

- 目标手 retargeting 仓库或接口契约；
- rot6d 的具体 convention 与 GR00T data config；
- `.pth` 与视频真实命名、目录和多段关系样例；
- VITRA 授权边界及允许用途。
- 目标手 retargeting solver/interface（通用 STL mesh 与轻量 FK 渲染已就绪）。

## 11. 上游依据

- NVIDIA Isaac-GR00T 当前仓库与数据说明：<https://github.com/NVIDIA/Isaac-GR00T>
- GR00T Data Preparation（v2 + `modality.json`、v3 转 v2）：<https://github.com/NVIDIA/Isaac-GR00T/blob/main/getting_started/data_preparation.md>
- LeRobot Dataset v3（多 episode 大 shard）：<https://github.com/huggingface/lerobot/blob/main/docs/source/lerobot-dataset-v3.mdx>
- LeRobot 大规模数据迁移说明：<https://github.com/huggingface/lerobot/blob/main/docs/source/porting_datasets_v3.mdx>
- Dex Retargeting：<https://github.com/dexsuite/dex-retargeting>
