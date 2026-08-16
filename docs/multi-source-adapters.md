# 多来源数据适配

EgoQC 不要求所有原始数据先复制成 LeRobot v3。每种来源先由只读 adapter 映射到
`CanonicalEpisode`，再由 capability manifest 决定哪些门禁可以执行。原始数据始终
留在 OSS/CPFS 挂载目录，质量报告、缓存与派生证据写入独立结果目录。

## 当前来源

| 来源 | 自动识别 | 统一视图 | 当前可测 | 明确不可直接验收 |
|---|---:|---|---|---|
| LeRobot v3 标准格式 | 是 | 原生 122/135 维结构 | schema、视频、SO(3)、时序、MANO 一致性 | 无 GT 时 MPJPE/ATE |
| `mano_hamer/front_camera` | 是 | legacy 双手记录 | schema 与几何兼容预览 | 左手约定未确认前不可最终验收 |
| EgoDex HDF5 + MP4 | 是 | `CanonicalEpisode`、双手各 25 源节点 | 配对、帧数、视频规格、SE(3)、confidence、标签 | MANO、MPJPE、ATE、独立时钟 Et |
| RekaDaily raw WebDataset | 是 | metadata-first 视频记录 | 索引、FPS、分辨率、容器、准确计帧、PTS jitter、抽样画质 | 手可见性、MANO、MPJPE、ODSR、ATE、Et |

## 接入新格式的边界

新 adapter 只负责读取、单位/形状归一化、来源语义和 provenance，不负责修复数据，
也不凭空补齐缺失能力。评估器只消费 canonical 字段；某项所需 capability 缺失时，
结果必须是 `null/not_applicable`，不能输出 pass。

推荐接入顺序：

1. 实现浅层格式识别，禁止为了识别而递归扫描百 TB 数据。
2. 将一个 episode 映射为严格递增 timestamp、视频引用、相机数据、手部轨迹和标签。
3. 校验所有时间维长度一致，并声明坐标方向、单位、模型和时间戳来源。
4. 用一个双手样本和一个单手样本验证 mask/confidence 语义。
5. 通过后再接批量清单、缓存、评估器和人工复检界面。

EgoDex 实测中，拇指字段使用 `leftThumb...`，其余手指使用
`leftIndexFinger...` 形式。adapter 使用显式 25 节点表并在缺字段时失败，避免将
漏掉拇指的 21 节点结果静默送入后续评估。

EgoDex 的 `valid_ratio` 只表达腕部 root confidence 达阈值，即“手存在”；
`all_joints_confident_ratio` 才表达该帧所有 25 个节点都达阈值。两者必须分开，
`joint_values_confident_ratio` 则给出全部“帧×节点”中达阈值的比例。不能以 root
可见率代替整手姿态质量；全节点指标的正式阈值仍需数据方说明 confidence 语义后标定。

RekaDaily 适配器默认只读取约数 MB 的 `metadata/index.parquet`，不会为格式识别
遍历或打开 5–8 GB 的 tar。只有进入抽检清单的 loose sample 才解码；批量 tar
流式 worker 应在 metadata、header 和画质门禁之后运行。
