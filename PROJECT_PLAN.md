# EgoQC-Lite 项目计划

更新日期：2026-08-07

## 1. 项目目标

为持续增长、数百 TB 规模的 LeRobot v3 / egocentric / MANO 数据建立一套：

- 低成本：默认不依赖大模型 API，不全量逐帧解码视频；
- 可复现：规则、阈值、模型、报告均带版本；
- 可增量：只处理新增或变化的 shard；
- 可解释：每个失败结果包含指标、原因和证据；
- 不浪费数据：不删除原始数据，按用途生成 Gold、Silver、Bronze、Quarantine 视图；
- 可扩展：单机参考实现可以按 Parquet/MP4 shard 扩展到批处理集群。

## 2. 核心原则

1. 原始数据只读、不可变。
2. 质量 metadata 与视频、Parquet 分离存储。
3. 100% 数据运行结构、数值和几何检查。
4. 视频检查采用轨迹驱动抽帧，每个 episode 默认不超过 12 帧。
5. 聚合 MP4 按 shard 单次顺序解码，不按 episode 重复 seek。
6. 大模型只允许作为离线难例标注器，不是生产依赖。
7. 质量等级只是路由结果，原始指标和 issue code 才是事实来源。

## 3. 当前完成状态

### M0：低成本质量门禁 MVP — 已完成

- [x] Python 包和 `egoqc` CLI。
- [x] LeRobot v3 `info.json`、episode route、Parquet 扫描。
- [x] episode length、frame index、timestamp、NaN/Inf 检查。
- [x] 相机、手腕、15 个 MANO joint 的 SO(3) 合法性检查。
- [x] 世界系 wrist 经 `extrinsics_w2c` 到相机系的一致性。
- [x] wrist 世界系/相机系旋转一致性。
- [x] `*_hand_pose` rotmat 与 state Euler 表示一致性。
- [x] MANO betas 漂移、手有效率和最长缺失段。
- [x] 聚合视频 offset、区间重叠和 duration 检查。
- [x] Gold / Silver / Bronze / Quarantine 分级。
- [x] shard 指纹与真正的增量结果缓存。
- [x] 轨迹驱动视觉抽帧计划。
- [x] 聚合 MP4 单次顺序解码和 contact sheet。
- [x] JSONL、SQLite、HTML 报告。
- [x] 合成 LeRobot v3 fixture、正常样本和注入错误测试。
- [x] 17 个单元及端到端测试通过。
- [x] 挂载目录数据集 Registry、增量 manifest 和幂等执行器。
- [x] manifest 运行前签名复核和 stale 数据保护。
- [x] config/code 运行身份与兼容式 Registry schema 迁移。
- [x] 跨 run 内容寻址 Parquet/视频探测缓存。
- [x] Parquet 列投影和 episode 单次分组。
- [x] worker 原子 claim、lease 和 shard heartbeat。
- [x] 多 batch 本地并行 worker。
- [x] Episodes、issues、videos、shards Parquet 结果。
- [x] 吞吐、逻辑输入量、缓存命中率和 shard 耗时指标。
- [x] `doctor`、`status`、`self-test` 运维入口。
- [x] 可选 `_SUCCESS` 封板标记检查。
- [x] 单数据集交互式质量看板。
- [x] 全局 Registry 质量看板。
- [x] contact sheet 证据画廊。

## 4. 后续里程碑

### M0.6：EgoScale V3.0 采购合同落地

- [x] 将 V3.0 阈值、采样规模、交付物和三态指标状态转换为机器可读合同。
- [x] 区分强制指标、参考指标、需要 GT 的指标和验收范围外指标。
- [x] 明确 `first_valid_camera` 与 `gravity_aligned` 是需要声明的替代世界系方案。
- [x] 建立现有实现覆盖矩阵与 EgoDex 初步差距分析。
- [x] 实现 EgoDex HDF5/MP4 只读 adapter、CanonicalEpisode 与 capability manifest。
- [x] 实现 RekaDaily raw metadata-first adapter、视频-only 能力门禁与 PTS jitter 事件清单。
- [x] 增加数值标注 timestamp Jitter 与按逐帧 dt 计算的 Vt+3MAD 输出。
- [ ] 将视频逐帧 PTS 按 episode offset 与 label timestamp 对齐，输出真实 Et；当前无 PTS 时显式标记为不可测。
- [x] 增加首个手部有效帧、任务内离开视野时长和连续 5 秒后的有效视频时长计算（第一有效相机帧仍需背景特征置信度）。
- [x] 增加数值/几何/时序坏帧清单与去重坏帧比例统计。
- [ ] 增加中英文 Distinct-2、Pairwise Distance 和语义人工抽检 manifest。
- [ ] 接入 ODSR/MPJPE Gold Set 后计算 Micro-F1 与各指尖 MPJPE。
- [ ] 生成采购方三结论报告：合格、限期整改、不合格。

### M1：真实数据试运行与标准校准

目标：证明现有规则能处理真实数据，并把默认阈值升级为正式标准。

- [ ] 选择 3 个不同来源、不同质量的数据 batch。
- [ ] 每个 batch 先抽 100–300 个 episode。
- [ ] 运行 `egoqc scan` 和 `egoqc extract-samples`。
- [ ] 人工检查不少于 300 个 contact sheet。
- [ ] 建立人工 Gold Set，标记：
  - hand visibility；
  - pose alignment；
  - temporal alignment；
  - camera/world validity；
  - allowed uses。
- [ ] 统计每个规则的误杀率和漏检率。
- [ ] 校准位置、旋转、betas、时间戳和有效率阈值。
- [ ] 发布 `standard_version=egoqc-hand-v1`。

验收标准：

- 严重结构/坐标错误漏检率低于 1%；
- Gold 数据误杀率低于 5%；
- 同一输入、同一版本重复运行结果完全一致；
- 第二次运行不重新计算未变化的 Parquet shard。

### M2：MANO 自动重投影证据

目标：把现有可视化脚本改造成批量、可度量的视觉检查层。

- [x] 将 HaWoR/MANO 资源路径改为显式配置。
- [x] 将 J0 canonical 补偿封装为可测试模块。
- [x] 在抽样帧上生成：
  - 左右手 21 joints；
  - mesh contour；
  - wrist point；
  - projected bounding box。
- [ ] 接入轻量 2D hand detector 或 hand segmentation。
- [ ] 计算：
  - wrist/joint reprojection error；
  - projected box IoU；
  - mesh-mask IoU；
  - out-of-frame ratio。
- [ ] 在报告中显示原图、overlay、误差热图和问题时间点。

验收标准：

- 每个 episode 默认解码不超过 12 帧；
- 同一个 MP4 shard 只打开一次；
- overlay 失败不会中断结构/几何质量流程；
- 视觉指标保存 provenance 和模型版本。

### M3：合成异常与小型对齐模型

目标：不用持续大模型 API，训练一个专用 Overlay Alignment Model。

- [ ] 实现 corruption generator：
  - 视频/pose 偏移 ±1/2/4/8 帧；
  - intrinsics 主点和焦距扰动；
  - extrinsics 平移/旋转扰动；
  - w2c/c2w 反用；
  - 左右手交换；
  - 左手重复/遗漏镜像；
  - J0 canonical 补偿遗漏；
  - joint permutation；
  - [x] 基于局部插值残差/MAD 的 pose freeze、jitter、短缺失段首版检测；
  - betas 漂移。
- [x] 冻结 clip-level Gold/teacher/weak label 优先级和权重契约。
- [x] 实现按 person/operator/session 分组切分、评估集纯 Gold 约束和跨 split 泄漏审计。
- [x] 实现每类 Gold 覆盖 readiness 和 precision 95% Wilson 下界门禁。
- [ ] 根据 `docs/qc-training-data-contract.md` 收集真实 Gold clips，并离线扩展教师软标签。
- [ ] 训练轻量视觉时序模型。
- [ ] 输出：
  - aligned probability；
  - error type；
  - temporal offset；
  - left/right validity；
  - uncertainty。
- [ ] 低置信样本进入人工队列。

验收标准：

- 不使用云端大模型也能完成推理；
- 严重错位召回率高于 95%；
- 按 source/person/scene 隔离测试，禁止相邻帧泄漏；
- 小模型只运行在抽样帧或异常 clip 上。

### M4：全局数据 Registry 与训练视图

目标：从单数据集扫描器升级为持续数据管理系统。

- [x] 增加 `egoqc register`。
- [x] 增加公开数据字段级 completion plan 与零覆盖 Parquet overlay。
- [ ] 增加 `egoqc ingest`。
- [ ] 增加 `egoqc export-view`。
- [ ] 建立全局：
  - `datasets.parquet`；
  - `files.parquet`；
  - `episodes.parquet`；
  - `quality_runs.parquet`。
- [ ] 生成不复制视频的训练视图：
  - hand-pose-gold；
  - hand-pose-gold-silver；
  - ego-video-pretrain；
  - world-motion；
  - quarantine。
- [ ] 为每条训练记录保存 sampling weight 和 allowed uses。
- [ ] 支持标准升级后重新物化视图。

验收标准：

- 新 batch 可自动登记、扫描和加入视图；
- 删除/移动 derived metadata 不影响 raw；
- 任意训练运行可以追溯到数据版本、标准版本和源文件；
- 不通过复制 MP4 构造训练集。

### M5：数百 TB 分布式执行

目标：保持单机代码不变，将 shard 任务调度到现有计算资源。

- [ ] 建立 shard 级任务 manifest。
- [ ] 支持本地多进程。
- [ ] 根据实际基础设施选择 Daft、Ray 或批处理系统适配器。
- [x] 失败任务可重试、可幂等恢复。
- [ ] 将计算尽量调度到数据所在节点。
- [ ] 增加吞吐、读取放大、缓存命中率和失败率监控。

验收标准：

- 任务粒度是文件 shard，不是单帧；
- worker 故障不会损坏 raw 或已完成结果；
- 新增数据处理成本与新增量成正比；
- 同一版本不重复扫描历史数据。

### M6：VITRA ingest 与 GR00T 训练导出

详细接口契约见 `docs/vitra-groot-pipeline-alignment.md`。

- [x] 冻结双格式架构：canonical LeRobot v3 主存，GR00T LeRobot v2 派生视图。
- [x] 定义 `mano108` state/action slice 和 draft export profile。
- [x] 明确连续 valid span、next-state action 和 source lineage 规则。
- [ ] 实现 `.pth` metadata inspector 与 shape/finite gate。
- [ ] 实现连续 valid span planner。
- [ ] 实现 `.pth` → canonical v3 ingest。
- [ ] 冻结 rot6d convention 并增加 round-trip 测试。
- [ ] 实现 v3 → GR00T v2 `export-view` 与 modality/export gate。
- [ ] 接入原子动作 segment manifest。
- [x] 实现左右手 URDF 解析，支持冻结每手 20-DOF joint order、axis 和 limit 来源。
- [x] 增加 URDF/Robot20 自动一致性检查与资源缺失报告。
- [x] 按人手与 URDF 几何冻结 finger1–finger5 = thumb/index/middle/ring/pinky。
- [x] 接入与当前 URDF 精确匹配的官方 STL mesh，并实现无 GPU 的 20DoF FK 渲染。
- [ ] 取得 retargeting 接口后接入 MANO21→Robot20 IK 与逐帧 FK QC。

验收标准：

- raw/canonical 不因切段、平滑或导出被修改；
- 所有训练行可追溯到 source episode/frame；
- state/action/video/modality/stats 一致；
- 同一 export profile 和输入 revision 输出可复现；
- Robot20 标签必须通过 FK、限位与时序质量检查。

## 5. 推荐数据目录

```text
/storage/embodied/
├── raw/
│   └── {source}/{date-or-batch}/{dataset_id}/
├── quality/
│   └── {source}/{dataset_id}/{standard_version}/
├── views/
│   └── {view_name}/{view_version}/
├── registry/
│   ├── datasets.parquet
│   ├── files.parquet
│   ├── episodes.parquet
│   └── quality_runs.parquet
└── standards/
    ├── egoqc-hand-v1.json
    └── egoqc-video-v1.json
```

禁止把几百 TB 数据长期追加到一个不可拆分的 dataset root。推荐按来源、日期或采集 batch 划分，每个 batch 单独 finalize 和扫描。

## 6. 阿里云 OSS / CPFS 部署方案

数据平面采用“OSS 冷源 + CPFS 热计算层”，避免长期维护两份完整数据。

### 6.1 存储职责

- OSS 是 raw 数据和最终 quality metadata 的权威存储，原始对象只读并建议开启版本控制。
- CPFS 是短期高吞吐工作区，只预热当前需要扫描、抽帧或训练的 batch/shard。
- Registry、标准配置、质量结果、抽样证据和任务 manifest 写回 OSS。
- CPFS 中已完成且成功写回 OSS 的可再生缓存允许按策略淘汰。
- worker 只通过 CPFS 的 POSIX 路径读取大 Parquet/MP4，避免每个进程重复下载 OSS 对象。

推荐布局：

```text
OSS
├── raw/{source}/{date}/{dataset_id}/
├── inventory/{bucket}/{inventory_date}/
├── quality/{dataset_id}/{standard_version}/{run_id}/
├── registry/{registry_version}/
├── standards/{standard_version}/
└── evidence/{dataset_id}/{run_id}/

CPFS
├── egoqc/input/{dataset_id}/
├── egoqc/work/{run_id}/
├── egoqc/cache/{fingerprint}/
└── egoqc/staging/{run_id}/
```

### 6.2 增量发现

1. 优先读取 OSS Bucket Inventory，不在每次运行时全量 `ListObjects`。
2. 将 `bucket + key + version_id/ETag + size + last_modified` 作为对象身份。
3. Inventory 与全局 Registry 做差分，只生成新增、变化和删除对象的任务。
4. 对 multipart 对象不能把 ETag 当作内容 MD5；必要时记录 OSS CRC64 或自有 checksum。
5. 新 batch 应有 `_SUCCESS`/finalized 标记，未完成上传的数据不进入扫描。

### 6.3 数据预热与执行

1. Registry 根据 episode route 把任务聚合成 Parquet/MP4 shard manifest。
2. 通过 CPFS 数据流动按目录或文件列表将所需 shard 从 OSS 导入 CPFS。
3. 预热完成并校验对象数、大小和必要 checksum 后才启动 worker。
4. worker 在数据所在地域运行；任务粒度保持为 shard。
5. 结构和几何扫描优先，只有被抽中的 episode 才解码视频。
6. 结果先写 CPFS staging，完成后原子发布并导出到 OSS。
7. OSS 写回成功并登记 provenance 后，CPFS 工作缓存进入可回收状态。

### 6.4 一致性与失败恢复

- OSS raw 只读，EgoQC 不回写或覆盖源对象。
- 每个 run 使用唯一 `run_id`，产物不可原地覆盖。
- manifest 记录 OSS identity、CPFS 路径、数据流动 task ID 和处理状态。
- 导入、扫描、导出都必须幂等，可从 shard checkpoint 恢复。
- CPFS/OSS 数据流动完成状态不能代替业务校验，仍需核对 manifest。
- 多个 CPFS 数据流关联同一个 OSS Bucket 时，按阿里云要求启用 Bucket 版本控制。

### 6.5 成本控制

- OSS Inventory 用于海量对象盘点，避免高频递归列举。
- CPFS 只保留热点 batch 和复用概率高的 shard，不长期镜像全部 OSS。
- 同一视频 shard 在一个 run 内只预热和顺序解码一次。
- contact sheet 默认只保存失败、低置信和人工抽检样本。
- 数据流动带宽按扫描窗口升降，计算与存储位于同地域。
- 每个 run 记录 OSS 请求数、导入/导出字节、CPFS 峰值占用、视频解码帧数和单位 episode 成本。

### 6.6 需要新增的实现

- [x] `egoqc register`：登记 CPFS/OSS 挂载目录中的显式数据集。
- [x] `egoqc plan`：根据 Registry 差分生成 dataset/batch manifest。
- [x] `egoqc run-manifest`：在挂载目录上幂等执行任务。
- [ ] `egoqc inventory import-oss`：导入 OSS Inventory。
- [ ] 将 dataset/batch manifest 进一步拆为独立 shard manifest。
- [ ] OSS/CPFS URI 与本地 POSIX 路径映射配置。
- [ ] 数据流动任务提交、轮询和业务校验适配器。
- [ ] 结果写回 OSS与 CPFS 缓存回收状态机。
- [ ] 阿里云 RAM 最小权限、凭证注入和审计说明。

## 7. 版本与可复现要求

每次质量运行必须记录：

- dataset ID；
- source file fingerprint/ETag；
- LeRobot format version；
- standard version；
- EgoQC package version；
- rule configuration hash；
- visual model version；
- MANO model/version/hash；
-运行时间和 worker 环境。

正式标准只新增版本，不原地修改旧版本。

## 8. 成本预算原则

- 文件头和 metadata：100%。
- Parquet 数值/几何：100%。
- 视频抽帧：每 episode 最多 6–12 帧。
- 密集异常 clip：目标不超过 episode 的 5%。
- 本地教师/人工：目标不超过 episode 的 0.5%。
- 云端大模型 API：默认 0，可选且必须有月度硬上限。

## 9. 最近一个迭代

优先执行 M1，不直接训练小模型：

1. 确认第一个真实 LeRobot v3 数据的 OSS URI、对应 CPFS 挂载路径和地域。
2. 选择一个完整 batch，生成 100–300 个 episode 的 shard manifest。
3. 通过现有 OSS/CPFS 数据流动把相关 shard 预热到 CPFS。
4. 在 CPFS 路径运行现有扫描器，记录吞吐和读取量。
5. 生成 contact sheet，人工确认误报、漏报和坐标约定。
6. 修正规则并发布第一版正式标准。
7. 将质量结果、证据和 run manifest 写回 OSS。

在真实数据验证之前，不启动 SCIZOR、DemInf、全量 CLIP/RAFT/DOVER 或通用 VLM 蒸馏。
