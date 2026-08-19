# EgoQC-Lite

面向持续增长、百 TB 级 LeRobot v3 第一视角双手 MANO 数据的低成本质量门禁。

EgoScale 数据交付 V3.0 的机器可读采购合同位于
`config/egoscale_v3.contract.json`，实现覆盖、EgoDex 差距和待确认口径见
`docs/egoscale-v3-acceptance-mapping.md`。合同文件用于验收编排，不可直接替代
`egoqc scan` 的运行配置。

多来源 adapter 的职责、能力门禁和新增格式接入顺序见
`docs/multi-source-adapters.md`。

QC 小模型需要的 clip-level Gold Set、教师软标签、数据量、切分防泄漏和
99% precision 置信度口径见 `docs/qc-training-data-contract.md`。生产训练前先运行
`egoqc audit-qc-training`；未通过 readiness 不开启自动拒收。

论文级评测必须使用独立 validation/test：validation 只冻结阈值，test 不重新调参，并报告
person/session 聚类置信区间和供应商 worst-group。协议见
[`docs/ieee-experiment-protocol.md`](docs/ieee-experiment-protocol.md)，命令为
`egoqc evaluate-qc-research`。

自动候选 clip 选择与开放世界教师审查见 `docs/qc-auto-clip-selection.md`；各层训练数据的
起步量、用途和隔离要求见 `docs/qc-model-data-plan.md`。视觉教师只读取规则召回的 4–8 秒
片段和少量未标注随机对照，不扫描全量视频。

公开数据集缺少非关键字段时，先生成字段级补齐计划，再把确定性派生字段写入
独立 Parquet overlay。该过程不复制或覆盖 raw，不会把模型估计、默认值或名义
时间轴伪装成上游 Ground Truth：

```bash
egoqc plan-completion /path/to/public-lerobot \
  --config config/default.json \
  --output /path/to/derived/completion-plan.json

egoqc build-completion-overlay /path/to/public-lerobot \
  --plan /path/to/derived/completion-plan.json \
  --output /path/to/derived/completion-overlay
```

首版可安全补齐 `timestamp = frame_index/fps`、全局 `index`、`state_mask` 与
`*_kept` 互转、由 FOV/分辨率计算的针孔 `intrinsics`、未知 `main_type=-1` 和
未启用的 segment marker。输出明确区分 `derived_exact`、`derived_nominal`、
`defaulted`、`missing`；名义 timestamp 不具备独立传感器时钟能力。外参、GT、
重力、触觉等无法确定的字段始终保持 missing。源数据在 plan 后发生变化时，
overlay 构建会拒绝执行并要求重新规划。

VITRA `.pth`、视频清洗、原子动作切分、MANO108/Robot20 与 GR00T 训练视图的
统一接口约定见 `docs/vitra-groot-pipeline-alignment.md`。项目以 LeRobot v3
作为规模化 canonical 主存；GR00T 当前所需的 LeRobot v2 + `modality.json`
按训练 run 导出，不改变原始数据。

第一版不依赖大模型 API，采用四条原则：

1. 原始数据不可变，质量结果单独存储。
2. 结构与几何检查覆盖 100% 数据。
3. 视频视觉检查按轨迹异常自适应抽帧，不逐帧跑模型。
4. 结果按源文件指纹和标准版本缓存，只处理新增或变化的数据。

## 已实现

- LeRobot v3 `info.json`、episode route 和逐帧 Parquet 扫描。
- episode length、frame index、timestamp 和非有限值检查。
- 所有相机、手腕与 15 个 MANO joint 旋转矩阵的 SO(3) 检查。
- 世界系 wrist 经 `extrinsics_w2c` 转换后与 `observation.state` 的一致性。
- wrist rotation 世界系→相机系一致性。
- `*_hand_pose` rotmat 与 `observation.state` Euler 表示一致性。
- MANO betas 跨帧漂移、有效率和最长缺失段。
- 位置、腕部旋转、15 个手指关节及相机外参的时序抖动检测。
- 孤立跳点、短时 mask 闪烁和 pose freeze 候选检测。
- 按逐帧 timestamp `dt` 计算左右手速度及 `median(V)+3×MAD`。
- 数值 timestamp 帧间隔 jitter、非递增检测和名义时间轴对齐误差。
- 逐帧错误事件、去重坏帧比例和 3% 合同门禁。
- 基于轨迹加速度、mask 切换和固定分位点的视觉抽帧计划。
- PyAV 视频三级探测：header、准确计帧、分层抽样清晰度/曝光检查。
- 连续有效段内的 One-Euro 位置修复和 SO(3) 自适应 SLERP 修复预览。
- 修复前后抖动、动作路径保留率和 p99 修正量安全门。
- 旧版 `mano_hamer/front_camera` 到统一双手记录的只读 adapter。
- EgoDex 同名 HDF5/MP4 到 `CanonicalEpisode` 的只读 adapter，保留 25 个源手部节点与逐节点 confidence。
- 每种来源输出 capability manifest；没有 GT、MANO、独立时间轴时对应指标保持不可测，不用代理值冒充验收结果。
- 按 MP4 shard 单次顺序解码抽样帧并生成 contact sheet。
- JSONL、SQLite 增量缓存和单文件 HTML 报告。
- Gold / Silver / Bronze / Quarantine 分级。
- PostgreSQL 动态异常复检队列：多人领取租约、并发版本检查、服务端结论和审计历史。

## 安装

```bash
cd /path/to/EgoQC-Lite
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -U pip setuptools wheel
python -m pip install -e . --no-build-isolation
```

## 使用

多人动态复检和 PostgreSQL 部署方式见
[`docs/postgres-review.md`](docs/postgres-review.md)。SQLite Registry 仍用于低成本
跑批状态，PostgreSQL 仅承载需要共享和审计的人工复检数据。

先运行不依赖 raw 数据的完整自检：

```bash
egoqc self-test
```

检查真实挂载、输出目录和 Registry：

```bash
egoqc doctor \
  --source-root /mnt/cpfs/raw \
  --output-root /mnt/cpfs/egoqc/results \
  --registry /mnt/cpfs/egoqc/registry.sqlite \
  --config config/default.json
```

```bash
egoqc scan /path/to/lerobot_v3 \
  --output /path/to/quality-result \
  --config config/default.json
```

全量默认只读视频头。准确计帧会完整解码，抽样质量会做少量 seek，建议只用于
风险文件或抽检集：

```bash
egoqc scan /path/to/lerobot_v3 --output /path/to/qc --video-check count
egoqc scan /path/to/lerobot_v3 --output /path/to/qc --video-check sample-quality
```

生成不覆盖源数据的手部时序修复候选：

```bash
egoqc repair-preview /path/to/lerobot_v3 \
  --episode 0 \
  --output /path/to/derived/episode-000000
```

输出包括保持原始 fixed-size schema 的 `repair-preview.parquet`、轻量的
`repair-deltas.parquet` 和 `repair-metrics.json`。供应商验收只读取修复前的原始
指标：超过原始 motion 阈值直接输出 `vendor_rework_required`。动作路径保留率
只评价“修复预览能否作为返工参考”，永远不能把不合格源数据改判为 accept。
若同时传入 `--hawor-root`，会额外生成 repaired MANO mesh+skeleton 审核视频。

旧版 `mano_hamer/front_camera` 数据先做只读兼容性预览：

```bash
egoqc inspect-adapter /path/to/mano_hamer \
  --episode 0 \
  --output /path/to/results/adapter-episode-0.json
```

adapter 会反转源 `T_world_camera` 得到标准 `extrinsics_w2c`，并把 axis-angle
转换成 SO(3)/Euler 统一视图。左手 pose 是否已采用右手 canonical 镜像约定目前
被明确记录为待数据方确认，因此不会直接进入最终验收。

EgoDex 使用可选 HDF5 依赖，并按相对 episode 路径做只读预览：

```bash
python -m pip install -e '.[egodex]'
egoqc inspect-adapter /path/to/egodex \
  --episode part1/add_remove_lid/0 \
  --confidence-threshold 0.5 \
  --output /srv/egoqc/results/part1-add-remove-lid-0.json
```

输出包含视频规格、HDF5/MP4 帧数一致性、双手 25 节点列表、有效帧比例、
confidence 汇总、标签和 capability manifest。EgoDex 当前没有原生 MANO 参数、
手部 GT、轨迹 GT 或独立传感器 timestamp，因此 MANO/MPJPE/ATE/Et 不会被误判为
通过；相机 transform 方向也会保留为待数据方确认。原始 HDF5/MP4 不会被修改。

RekaDaily raw 快照可先只扫 `metadata/index.parquet`，不打开 WebDataset tar：

```bash
egoqc inspect-adapter /srv/egoqc/samples/rekadaily \
  --output /srv/egoqc/results/rekadaily-index.json
```

单条视频可以继续执行准确计帧、帧间隔 jitter 或分层画质抽样。adapter 会优先
读取 `sample/<project>/` 中的 loose sample；若不存在，则按 metadata 的
`project/video_id/src_ext` 直接定位 `data/<project>/*.tar` 内的成员并把可 seek
文件流交给 PyAV，不会解包第二份视频：

```bash
egoqc inspect-adapter /path/to/rekadaily \
  --episode <video_id> \
  --video-check count \
  --output /path/to/results/<video_id>-count.json
```

`count` 会输出 jitter mean/max、非递增 timestamp 数量，以及最多 32 个超阈值
间隔事件的帧号；报告中的 `source_access=webdataset_tar_member` 和 `video_uri`
记录只读 tar 来源。视频-only 数据的 MANO、MPJPE、ATE、Et 等保持不可测。

GPU 机器可在不解包 tar 的前提下运行手部预筛。当前 HaWoR/WiLoR YOLO backend
按 5 FPS 抽样，0.4 秒以内的内部检测闪断会被桥接，并输出连续离画超过 1 秒、
双手可见率、疑似超过两只手、有效时长和带框证据图：

```bash
python -m pip install -e '.[hand]'
egoqc screen-rekadaily-hands /path/to/rekadaily \
  --video-id <video-id> \
  --weights /path/to/detector.pt \
  --output /path/to/hand-screen \
  --sample-fps 5 --device 0 --workers 4
```

`edge_touch_ratio` 只作为 ego 手腕贴近底边/疑似截断的观察指标，不单独改变门禁。
持续多手候选采用置信度 ≥0.7、跨类别 NMS IoU 0.5、持续 ≥0.6 秒；原始
`raw_extra_hands_ratio` 保留用于审计。该指标只能说明检测到超过两只手，不能
自动判断手属于谁。
所有决定均标记为 model screening，采购拒收仍需人工复核。HaWoR/WiLoR 和
Ultralytics 权重/代码的生产许可必须由使用方另行审核。

### 外采数据验收原则

- `accept/reject/rework` 只由原始交付数据决定；
- EgoQC 不在验收路径中自动修复供应商数据；
- repaired Parquet/MP4 只能用于问题定位、阈值分析和返工示例；
- 供应商修好后必须作为新的 dataset revision 重新登记、重新跑全套门禁；
- 原始版本及其失败报告保留，不能用派生文件覆盖审计记录。

生产默认配置将任何 motion dimension 失败写入
`decisions/rework_manifest.jsonl`；`review_manifest.jsonl` 只保留尚需人工判断的
边界问题。若某个项目合同要求直接拒收，可把
`acceptance.motion_failure_decision` 改成 `reject`。

生成第二阶段视觉检查证据：

```bash
egoqc extract-samples /path/to/lerobot_v3 \
  --plan /path/to/quality-result/sample_plan.jsonl \
  --output /path/to/quality-result/evidence
```

输出目录同时包含：

- `index.html`：抽样帧和 MANO overlay 画廊；
- `review.html`：人工判定、备注、JSONL 导入导出工作台；
- `episodes-vlc.xspf`：按 episode 起止时间播放聚合 MP4 的 VLC 播放列表；
- `human-reviews.jsonl`：由审核页导出的 Gold Set 标签，不写回 raw 数据。

VLC 负责检查连续运动、抖动和视频/pose 时间错位；`review.html` 负责查看 MANO
几何证据并沉淀结构化标签。两者都只读取原始视频，不复制整个数据集。

测评运行期间使用 live 模式，页面每 2 秒读取 Registry heartbeat 和已经完成的
Parquet shard。新发现的 rework episode 会滚动加入队列，最终视频检查完成后
自动替换 provisional 结果：

```bash
egoqc serve-review \
  --evidence-root /srv/egoqc/results/evidence \
  --quality-root /srv/egoqc/results/current-quality \
  --registry /srv/egoqc/control/registry.sqlite \
  --host 127.0.0.1 --port 8765
```

开发机上运行服务时，在本机建立 SSH 隧道：

```bash
ssh -L 8765:127.0.0.1:8765 <user>@<dev-host> -p <ssh-port>
```

然后打开 `http://127.0.0.1:8765/review.html`。服务只使用 Python 标准库，
没有数据库服务、前端构建链或模型 API 成本。直接双击 `review.html` 时仍可使用
离线模式，但不会得到运行中更新。

`serve-review` 可以和 `run-manifest` 同时启动。即使 evidence 还没生成，它也会
先创建空工作台，随后把每个已完成 shard 的 provisional episode 滚动加入队列；
`extract-samples` 完成后刷新页面即可补上 contact sheet 和视频标签页。

复检页支持：系统返工队列、问题搜索、原始视频区间播放、MANO 参考视频、失败
指标、缺陷标签、备注、本地自动保存、JSONL 导入导出和键盘快捷键。系统已判
不合格时，人工确认合格只会输出 `override_requested` 并进入仲裁。

也可以只对人工抽检 episode 生成烧录 MANO mesh/joints 和状态文字的派生 MP4：

```bash
egoqc render-annotated-video /path/to/lerobot_v3 \
  --episode 0 \
  --output /path/to/evidence/episode-000000-annotated.mp4 \
  --hawor-root /path/to/HaWoR \
  --mano-data-root /path/to/HaWoR/_DATA
```

提供 `--review-labels human-reviews.jsonl` 后，还会把人工 decision/note 烧录到
视频 HUD。该命令始终新建派生 H.264 MP4，不修改 raw；试运行可用
`--max-frames 300` 限制输出长度。
用 `--start-frame 1200 --max-frames 120` 可以只渲染异常点附近约 4 秒，适合
大规模分层抽检，避免从每个长 episode 的开头重复转码。

生成单个 episode 的位置、腕部旋转、手指关节与相机抖动曲线：

```bash
egoqc temporal-plot /path/to/lerobot_v3 \
  --episode 0 \
  --output /path/to/quality-result/episode-000000-temporal.svg
```

对于包含数千 episode 的聚合根，先只检查一个 episode，避免启动全根扫描：

```bash
egoqc inspect-episode /path/to/large_lerobot_v3 \
  --episode 1234 \
  --output /path/to/results/episode-001234-qc.json
```

检查任意每手 20-DOF URDF 的关节顺序、轴、限位和缺失资源：

```bash
egoqc inspect-urdf \
  --left /path/to/robot-hand/left.urdf \
  --right /path/to/robot-hand/right.urdf \
  --expected-dof 20 \
  --output quality/robot20-urdf-report.json
```

给定 URDF 引用的 STL mesh，无需 GPU、Blender 或外部渲染服务即可生成
20-DoF FK mesh 证据图：

```bash
egoqc render-robot20 \
  --urdf /path/to/robot-hand/left.urdf \
  --pose neutral \
  --output quality/robot20-left-neutral.png

egoqc render-robot20 \
  --urdf /path/to/robot-hand/right.urdf \
  --pose midrange \
  --output quality/robot20-right-midrange.png
```

实际 retargeting 结果可用 `--q q0,q1,...,q19` 传入 20 个弧度制关节角。
`--max-triangles` 可在大批量证据渲染时限制三角面数，完整 mesh 默认保留。

Robot20 参考语义按人手顺序定义为 `finger1=thumb`、`finger2=index`、
`finger3=middle`、`finger4=ring`、`finger5=pinky`。MANO→Robot20 使用 21
关键点和骨段的 FK/IK 对齐，不直接复制 MANO 15 个局部旋转。

如果已经按 MANO 许可取得模型，并准备好 HaWoR 官方仓库，可同时生成 MANO
mesh、21 joints、wrist 和 projected bounding box overlay：

```bash
python -m pip install -e ".[mano]" --no-build-isolation
```

```bash
egoqc extract-samples /path/to/lerobot_v3 \
  --plan /path/to/quality-result/sample_plan.jsonl \
  --output /path/to/quality-result/evidence \
  --hawor-root /path/to/HaWoR \
  --mano-data-root /path/to/HaWoR/_DATA \
  --mano-alpha 0.48
```

已有 repaired annotated MP4 时，可通过 `--annotated-root` 把它们接入复检页的
“MANO 参考”标签页。

期望的受许可模型路径为：

```text
<mano-data-root>/data/mano/MANO_RIGHT.pkl
<mano-data-root>/data_left/mano_left/MANO_LEFT.pkl
```

MANO 模式复用 HaWoR 的 `lib/models/mano_wrapper.py`，因此该环境还需要 HaWoR
对应版本的 PyTorch 依赖。管线会按论文/官方推理约定执行 J0 canonical 补偿，
并将数据集中以右手 canonical 保存的左手局部旋转还原后再送入左手 MANO。
逐帧 overlay 失败只记录在 `evidence_manifest.jsonl`，不会中断基础质量扫描。
`mano_provenance.json` 保存 HaWoR 路径、透明度和左右 MANO 模型 SHA-256。

输出：

```text
quality-result/
├── cache.sqlite
├── summary.json
├── episodes.jsonl
├── issues.jsonl
├── bad_frames.jsonl
├── bad_frames.parquet
├── videos.jsonl
├── sample_plan.jsonl
└── report.html
```

打开 `report.html` 即可查看 episode 分级、问题原因和第二阶段视觉抽帧计划。
`bad_frames` 每行包含 `episode_index/frame_index/code/side/measured/threshold/unit`，
可直接用于供应商返工、异常点视频烧录和人工分层抽检。对于真实视频—标注
Et，如果未提供逐帧视频 PTS，则输出 `null` 和不可测原因，不使用
`frame_index/fps` 代替真实同步精度。

## OSS / CPFS 挂载目录管线

第一版不依赖阿里云 SDK。只要开发机能把数据挂载为普通目录，就可以运行：

```text
/mnt/cpfs/raw/source-a/batch-001/     # LeRobot v3 数据集
/mnt/cpfs/egoqc/registry.sqlite       # 全局状态
/mnt/cpfs/egoqc/manifests/            # 待执行任务
/mnt/cpfs/egoqc/results/              # 质量结果
```

`registry.sqlite`、manifest 和运行中的结果应放在本地盘或 CPFS。不要把
SQLite 数据库直接放到 OSS-FUSE/object-storage 挂载目录中。

### 1. 登记数据集

只登记明确的数据集根目录，不递归遍历整个百 TB 挂载点：

```bash
egoqc register \
  /mnt/cpfs/raw/source-a/batch-001 \
  /mnt/cpfs/raw/source-a/batch-002 \
  --source oss-prod \
  --source-root /mnt/cpfs/raw \
  --require-marker _SUCCESS \
  --registry /mnt/cpfs/egoqc/registry.sqlite
```

数据集很多时，使用每行一个根目录的列表：

```bash
egoqc register \
  --dataset-list /mnt/cpfs/egoqc/datasets.txt \
  --source oss-prod \
  --source-root /mnt/cpfs/raw \
  --registry /mnt/cpfs/egoqc/registry.sqlite
```

`--source` 和相对于 `--source-root` 的路径共同形成稳定数据身份。即使以后
更换挂载点，只要这两项不变，数据集 ID 仍然一致。

### 2. 生成增量计划

```bash
egoqc plan \
  --registry /mnt/cpfs/egoqc/registry.sqlite \
  --manifest /mnt/cpfs/egoqc/manifests/run-001.jsonl \
  --output-root /mnt/cpfs/egoqc/results \
  --config config/default.json
```

只有以下数据会进入 manifest：

- 第一次登记的数据集；
- Parquet、MP4 或 metadata 的 size/mtime 发生变化；
- 当前 `standard_version` 尚未成功运行的数据集。

### 3. 幂等执行

```bash
egoqc run-manifest \
  --registry /mnt/cpfs/egoqc/registry.sqlite \
  --manifest /mnt/cpfs/egoqc/manifests/run-001.jsonl \
  --config config/default.json \
  --hash-mode none \
  --cache-root /mnt/cpfs/egoqc/artifact-cache \
  --workers 4 \
  --continue-on-error
```

挂载层已经提供稳定对象身份时，推荐 `--hash-mode none`，避免为每个大文件
额外读取首尾数据。执行成功后再次生成计划，该版本的数据会被跳过。若数据在
`plan` 后发生改变，任务会标记为 `stale`，必须重新登记和生成计划。

任务身份由 `dataset_signature + standard_version + config_hash +
code_version` 共同决定。修改阈值即使不修改标准名称，也会生成新任务。不同
dataset revision 共享内容寻址的 Parquet 和视频探测缓存；一个 metadata 文件
变化不会迫使所有未变化 shard 重新计算。

多个 worker 可以读取同一个 manifest。Registry 使用原子 claim 和带过期时间
的 lease 阻止重复执行，并在每个 shard 后刷新 heartbeat。`--workers` 在一台
开发机上并行处理不同 dataset/batch，不会让多个线程争抢同一个 MP4。旧版
manifest 不含完整运行身份，升级到 0.3 后应重新运行 `egoqc plan`。

查看总体进度和失败/过期任务：

```bash
egoqc status --registry /mnt/cpfs/egoqc/registry.sqlite
```

生成跨 batch 的全局质量看板：

```bash
egoqc dashboard \
  --registry /mnt/cpfs/egoqc/registry.sqlite \
  --output /mnt/cpfs/egoqc/dashboard.html
```

单数据集的 `report.html` 包含质量分层、问题 Top 12、缓存命中率、最慢 shard
和可筛选的 episode 明细。运行 `extract-samples` 后，`evidence/index.html`
提供 contact sheet 证据画廊。

执行结果写入：

```text
results/{dataset_id}/{standard_version}/{config_hash}/{dataset_signature}/
├── summary.json
├── episodes.jsonl
├── episodes.parquet
├── issues.jsonl
├── issues.parquet
├── videos.jsonl
├── videos.parquet
├── shards.jsonl
├── shards.parquet
├── sample_plan.jsonl
├── cache.sqlite
├── shard_cache/
└── report.html
```

HTML 默认只列出最严重的 500 个 episode，完整结果始终保存在 JSONL 中，避免
大型 batch 生成超大网页。数量可通过 `report.max_episodes` 调整。

`summary.json` 记录运行起止时间、逻辑输入字节、吞吐率、Parquet/视频缓存
命中率；`shards.parquet` 保存每个 shard 的大小、耗时、fingerprint 和缓存状态。

### 运行前 ETA 与实时进度

先根据 manifest、Registry 历史吞吐和预期缓存命中估算冷/热运行时间：

```bash
egoqc estimate \
  --registry /mnt/cpfs/egoqc/registry.sqlite \
  --manifest /mnt/cpfs/egoqc/manifests/run-001.jsonl \
  --config config/default.json \
  --workers 4
```

`cold/warm` 同时给出 low、expected、high 秒数与预计完成时间。首次没有历史时使用
配置中的保守吞吐；有成功记录后自动改用历史中位数。

运行时使用 `--progress` 向 stderr 输出 JSONL：

```bash
egoqc run-manifest ... --workers 4 --progress
```

每条记录包含 dataset、已完成/总 shard、百分比、已运行时间、ETA、处理字节和当前
路径。相同信息写入 Registry，`egoqc status` 和重新生成的 dashboard 都能查看。

### 验收决定与不合格清单

0.5 起每个 episode 除 Gold/Silver/Bronze/Quarantine 外，还输出：

- `format_pass`：schema、索引、任务、视频和文件约定；
- `numeric_pass`：坐标、SO(3)、表示与有限值；
- `motion_pass`：抖动、跳点、冻结和 mask；
- `fit_pass`：独立 2D 拟合检查，未运行时为 null；
- `final_pass` 和 `decision`。

新增目录：

```text
decisions/
├── episode_decisions.jsonl
├── episode_decisions.parquet
├── quarantine_manifest.jsonl
├── reject_manifest.jsonl
├── review_manifest.jsonl
├── retry_manifest.jsonl
├── retry_files.jsonl
├── rejected_episodes.parquet
├── rejected_files.jsonl
└── rejected_files.parquet
```

`retry` 是读取/探测等运行故障；再次运行同一 dataset 时，内容寻址缓存会跳过成功
shard，只重新计算失败部分。`quarantine/reject` 只产生派生清单，不移动或删除原始数据。

多个结果目录的失败文件可以合并成去重计划：

```bash
egoqc plan-retry \
  --quality-root results/run-a results/run-b \
  --output manifests/retry-shards.jsonl
```

当前执行方式仍是使用原 dataset manifest 重跑，artifact cache 会命中健康 shard；
`retry-shards.jsonl` 同时可作为后续 Ray/Kubernetes shard worker 的稳定输入。

### 离线交互式调参

```bash
egoqc tune \
  --quality-root /mnt/cpfs/egoqc/results/<run> \
  --config config/default.json \
  --output /mnt/cpfs/egoqc/results/<run>/tuner.html
```

浏览器中调整时序阈值会立即重算当前 episode 样本的触发数量和明细，并可下载新
JSON 配置。页面完全离线，不调用大模型或外部 API，也不会修改运行中的标准。
下载配置后重新执行 `plan`；config hash 会保证不同阈值的结果不会混用。

### 4. 日常增量循环

```text
新增 batch 完成并封板
  → 更新 datasets.txt
  → egoqc register
  → egoqc plan
  → egoqc run-manifest
  → 质量结果归档/同步回 OSS
```

## 百 TB 运行方式

- 按 data/video shard 分配 worker，不按 episode 随机打开视频。
- 每个 MP4 汇总全部目标 frame index 后只顺序打开一次。
- `--hash-mode headtail` 只读取文件首尾各 1 MiB；可信对象存储可用
  `--hash-mode none`，直接利用 size/mtime/ETag。
- `quality/` 与原始数据分离，可以反复升级标准而不复制 MP4。
- 当前 manifest 调度粒度是 dataset/batch，数据集内部按 shard 读取和缓存。
  后续 M5 会把 manifest 进一步细化为独立 shard 任务并接入多进程、Ray
  或现有批处理系统。

## 约定

- Euler 使用与 `scipy.spatial.transform.Rotation.from_euler("xyz")` 对应的
  extrinsic xyz 约定。
- 左手 raw rotmat 与 Euler 表示应处于同一个“右手镜像 canonical”空间；
  渲染真实左手时再执行镜像还原。
- 质量检查不会像可视化脚本一样把非法 rotation 或 NaN 静默替换为单位阵。
- 默认阈值只是启动值，应使用人工 Gold Set 校准后固化为新的标准版本。

## 下一阶段

`sample_plan.jsonl` 是视觉层的稳定接口。下一步按计划实现：

1. MANO joints/contour overlay evidence。
2. 程序化注入时间偏移、外参、左右手和镜像错误。
3. 小型 Overlay Alignment Model。
4. 低置信片段人工复核；大模型教师保持可选。
## RekaDaily 训练视图

公开 raw 视频可先生成低成本视频预训练候选，并用严格状态机逐步晋级到 MANO Silver：

```bash
egoqc build-rekadaily-views /path/to/RekaDaily-10k-raw \
  --output /path/to/quality/rekadaily-views \
  --materialized-only
```

输出包含不合格原因、手部预筛/MANO/人工对齐队列、许可证治理状态和完整 provenance。
详细契约见 `docs/rekadaily-training-views.md`。
