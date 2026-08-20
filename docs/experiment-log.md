# EgoQC-Lite 实验日志

最后更新：2026-08-20（Asia/Singapore）
日志状态：持续追加；历史失败与中断不得删除，只能新增更正记录。

## 记录原则

- 原始数据只读；公开仓库不记录供应商原始路径、主机地址、凭证或人员信息。
- 每个实验记录目标、固定输入、参数、代码提交、环境、结果、问题、根因、处置和结论边界。
- `weak teacher agreement`、合成干预命中率和 JSON 格式覆盖率都不是人工 Gold 准确率。
- validation/test 只能使用隔离的人工 Gold；当前人工 Gold 数量仍为 0，因此禁止宣称 99% 准确率。
- 机器可读记录位于
  [`artifacts/experiments/experiment-run-log.jsonl`](../artifacts/experiments/experiment-run-log.jsonl)。

## 当前实验状态

| 项目 | 当前值 | 解释 |
|---|---:|---|
| 可访问数据毛量 | 约 2,532.25 小时 | 1,008.32 小时为 metadata 精确值，另有 1,523.93 小时为抽样估计；尚未跨源去重 |
| 已冻结 few-B 对照样本 | 200 clips | 100 个弱正例、100 个弱负例，200 个独立 split group |
| 供应商本地 8B | 434 / 434 clips | 434 条结构化 JSON 全部有效；最终结果 SHA-256 已记录 |
| 新增本地弱标签池 | 434 | 206 条高置信 train-only；228 条进入人工复检 |
| 已准备弱/合成训练记录 | 10,018 + 434 | 只能进入 train，不能进入 validation/test；需去重后统计最终量 |
| 人工 Gold | 0 | 正式准确率、99% precision 和自动拒收均未解锁 |
| 外部供应商 API 调用 | 0 | 434 条供应商候选未发送给外部 API |

## EXP-001：多来源数据资产盘点

- 日期：2026-08-20 前完成回填。
- 目标：确认真实可用数据规模，而不是只统计当前小样本实验集。
- 输入：供应商 LeRobot v3、内部/已整理 LeRobot v3、公开 ego 视频与手部 SE(3) 标注。
- 结果：metadata 精确统计 1,008.3195 小时；公开数据抽样估计 1,523.93 小时；毛量合计
  2,532.2495 小时。
- 遇到的问题：不同来源统计口径不同，并且跨批次、跨来源可能重复。
- 根因：公开数据只有任务均衡抽样画像，供应商与内部数据有完整 metadata；不能把估计值和精确值混为一谈。
- 解决：账本同时保存 `hours_kind`；总量明确标注 gross volume；后续以感知去重和
  `split_group` 冻结唯一训练量。
- 结论边界：当前数字不是唯一训练小时数。
- 证据：[`source-ledger-v1.json`](../artifacts/experiments/source-ledger-v1.json)。

## EXP-002：内部 LeRobot 校准案例集

- 日期：2026-08-19。
- 目标：验证规则证据、原视频、MANO 叠加和人工复检入口能否闭环。
- 协议：12 条规则高风险、6 条严格干净对照、6 条低事件对照；8 workers，seed 43。
- 结果：24/24 案例生成成功，0 失败，耗时 5.595 秒。
- 遇到的问题：规则报警以瞬时速度离群为主，缺少人工真值确认是否真的影响训练。
- 解决：保存 MP4、JPG、JSON 和可播放画廊，并把 `human_label_status` 固定为 `pending`。
- 结论边界：这是可视化与证据链冒烟，不是规则准确率。
- 证据：[`oss-pilot-v1/experiment.json`](../artifacts/experiments/oss-pilot-v1/experiment.json)。

## EXP-003：供应商跨来源校准案例集

- 日期：2026-08-20。
- 目标：检查内部样本上成立的流程能否迁移到独立供应商来源。
- 协议：12 条规则高风险、1 条严格干净对照、6 条低事件对照；8 workers，seed 47。
- 结果：19/19 案例生成成功，0 失败，耗时 3.654 秒。
- 遇到的问题：严格干净对照只有 1 条，无法估计假阳性。
- 解决：后续扩展队列中单独保留 16 条 clean-gap control 和 100 条 low-event control；
  不使用只含异常的便利样本训练模型。
- 结论边界：19 条案例只证明跨来源工程可运行。
- 证据：[`supplier-pilot-v1/experiment.json`](../artifacts/experiments/supplier-pilot-v1/experiment.json)。

## EXP-004：Phase A 可控干预

- 日期：2026-08-20。
- 目标：验证规则/几何专家是否会对已知通道破坏产生可定位、随强度不减弱的响应。
- 协议：4 个 episode，7 类干预，每类 low/high 两档，共 56 次干预和 4,090 条 evidence delta。
- 结果：目标响应率 1.0，单调不减率 1.0；时间戳区间定位均值 0.9983，其余已定义目标为 1.0。
- 遇到的问题：`beta_drift` 没有可直接复用的区间精度定义；合成错误过于可控，结果容易被误写成真实准确率。
- 解决：`beta_drift` 报告为 null；摘要强制写入“synthetic sanity check”；下一阶段必须在双人复核的真实错误上验证专家可靠性。
- 结论边界：100% 只代表被注入的错误能触发预期专家，不代表真实数据 recall/precision。
- 证据：[`aggregate-summary.json`](../artifacts/experiments/interventions/oss-phase-a-v1/aggregate-summary.json)。

## EXP-005：百炼公共数据教师连通性试验

- 日期：2026-08-20。
- 目标：验证低成本视频抽帧、结构化提示和百炼兼容接口。
- 协议：公共 ego 数据 2 个请求；`qwen3-vl-plus`；1.5 FPS，最多 12 帧，最长边 448，JPEG 72。
- 第一次问题：默认 endpoint 配置未生效，2/2 请求失败且 token 使用为 0。
- 根因：运行环境残留/缺失的 base URL 与 CLI 版本不同步；另有一次命令把 `1~` 传给并发参数。
- 解决：部署包含 `run-teacher-api` 的最新代码；显式指定北京兼容接口；修正并发参数。
- 最终结果：2/2 成功，input tokens 3,138，output tokens 1,414，凭证未写入产物。
- 结论边界：仅是连接和输出结构试验，样本量不足以评价教师质量。
- 证据：[`teacher-api-public-pilot-v1/summary.json`](../artifacts/experiments/teacher-api-public-pilot-v1/summary.json)。

## EXP-006：few-B 20 条容量冒烟

- 日期：2026-08-20。
- 目标：比较 2B、4B、8B 基座的显存、延迟、JSON 覆盖和弱教师一致性。
- 结果：2B/8B JSON 覆盖均为 1.0，4B 为 0.35；8B 弱 F1 为 0.222；2B/4B 弱 recall 为 0。
- 遇到的问题：20 条样本方差过大，且不同模型生成 token 数不同会混淆延迟。
- 解决：压缩为固定 sparse JSON schema，`max_new_tokens=128`；扩大到冻结的 200 条同协议样本。
- 结论边界：弱标签一致性不是人工准确率。
- 证据：[`comparison-balanced20-v1`](../artifacts/experiments/few-b-v1/comparison-balanced20-v1/comparison.md)。

## EXP-007：few-B 200 条冻结对照

- 日期：2026-08-20。
- 固定协议：200 clips，8 帧，最长边 448，BF16，H20，`max_new_tokens=128`；
  样本 SHA-256 为 `0d7251e2487b936322f2bf428877a7458939de2ed1e2ce168f51a5edd6088614`。
- 结果：

| 模型 | 参数 | 显存 | JSON 覆盖 | P50 | P95 | video-h/wall-h | 弱 recall | 弱 F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-VL-2B | 2.128B | 3.96 GiB | 100% | 1.046 s | 1.224 s | 5.60 | 0.000 | NA |
| Qwen3-VL-4B | 4.438B | 8.27 GiB | 50.5% | 1.826 s | 3.314 s | 2.79 | 0.000 | NA |
| Qwen3-VL-8B | 8.767B | 16.33 GiB | 100% | 1.314 s | 1.483 s | 5.13 | 0.132 | 0.189 |

- 遇到的问题：未训练基座明显漏检；4B 还存在严重格式遵循问题；8B 也只有 13.2% 弱 recall。
- 解决：不把 base model 直接部署；4B 作为主 SFT 候选、8B 作为 challenger、2B 作为轻量对照；
  validation/test 必须换成人工 Gold。
- 结论：SFT/蒸馏是必要条件，但尚未证明 SFT 后能达到目标精度。
- 证据：[`comparison-balanced200-v1`](../artifacts/experiments/few-b-v2/comparison-balanced200-v1/comparison.md)。

## EXP-008：2B CPU 单条基线

- 日期：2026-08-20。
- 协议：Qwen3-VL-2B，FP32，32 CPU cores，1 个约 6 秒 clip。
- 结果：端到端 5.73 秒，约 1.05 video-h/wall-h。
- 遇到的问题：单条样本且仅 FP32，不能代表 INT8/INT4 生产性能。
- 解决：保留为下界；后续必须在同一冻结 cohort 上补 INT8/INT4、P50/P95、RSS 和精度差。
- 证据：[`qwen3-vl-2b-fp32-cpu-1clip-benchmark.json`](../artifacts/experiments/few-b-v1/qwen3-vl-2b-fp32-cpu-1clip-benchmark.json)。

## EXP-009：434 个独立供应商候选组

- 日期：2026-08-20。
- 目标：扩大来源和控制样本，同时阻止同一原视频相邻 clip 泄漏。
- 协议：每个 `split_group` 只取一条；4 个来源族，共 434 个独立组；318 条 deterministic bad、
  16 条 clean-gap control、100 条 low-event control。
- 遇到的问题：原队列同一原视频可能贡献多个相邻片段；供应商帧发送外部教师的授权范围不够明确。
- 解决：加入 `--one-per-split-group`；外部供应商 API 批次不执行，改用本地 8B；
  任何 API 教师标签都只允许进入 train，不得成为 Gold。
- 结论边界：434 是候选组数量，不是高质量已标注样本数量。
- 对应提交：`3872628`、`79cc8ff`。

## EXP-010：本地 8B 供应商候选推理与 I/O 优化

- 日期：2026-08-20。
- 输入：EXP-009 的 434 个独立组；8 帧、448 edge、BF16、H20；所有 raw 只读。
- 阶段 A：每帧单独 seek，22/434 时平均 6.565 秒/clip，ETA 2,704.8 秒。
- 阶段 B：单次 seek 后顺序解码，新增片段平均 3.261 秒/clip；最近 8 条中解码 2.383 秒、
  预处理 0.041 秒、生成 0.709 秒，证明瓶颈在挂载视频解码。
- 阶段 C：4 路有界预取，CPU 解码与 GPU 推理重叠；从 134 条断点继续后，新增 251 条平均
  1.227 秒/clip，较阶段 A 提升约 5.35 倍。
- 当前结果：partial 文件 386 条，386 条结构化 JSON 有效；SHA-256 为
  `5e6d22978867c281e82fbd537fe835153702fca077346647978c950c478bcf03`。
- 遇到的问题：任务中断发生在写 partial 与写 progress 之间，因此 partial 为 386 行，而 progress
  记录为 385；早期实现中断后会从头运行；tmux 窗口退出后曾留下仍在运行的子进程。
- 解决：加入协议一致的 `--resume` 去重续跑；每条完成即更新 partial；加入有界预取；
  通过精确 PID 中断残留进程并确认 GPU 释放。
- 结论边界：100% JSON 有效只表示格式正确；本批仍是未校准 base 8B 弱标签，不能作为验收真值。
- 证据：[`supplier434-local8b-partial-v1/progress-snapshot.json`](../artifacts/experiments/few-b-v2/supplier434-local8b-partial-v1/progress-snapshot.json)。
- 对应提交：`5c3d56a`、`0745c32`、`fdae8a7`。

## EXP-011：本地弱标签池治理

- 日期：2026-08-20。
- 目标：把本地 sparse prediction 转为可训练但不可用于验收的 manifest。
- 规则：低置信、拒答、格式错误、规则/模型分歧进入人工复检；高置信结果最多权重 0.25；
  所有记录固定为 train；没有显式 MANO overlay 时 `mano_overlay_drift` mask 为 0。
- 遇到的问题：模型看到 raw RGB 时无法判断 MANO mesh 漂移；若把缺失任务直接填 0 会制造伪负例。
- 解决：基于模态可用性设置 label mask，而不是默认补零；人工 Gold 仍是唯一评测真值。
- 当前状态：转换代码和测试已完成，386 条 partial 尚未正式生成最终训练池。
- 对应提交：`69255e2`。

## EXP-012：raw 不可变保护与运行事件

- 日期：2026-08-20。
- 事件：用户要求再次确认 OSS raw 不会被改动。审计确认所有模型输出都在 workspace，供应商外部
  API 未运行；但发现一个 tmux 子进程仍在只读解码 raw。
- 处置：先尝试向 tmux pane 发送中断；pane 已消失后，解析出精确 PID 并发送 `SIGINT`；最终
  模型与 GPU 进程清零，partial 保留。
- 硬保护：任何 `--output` 位于 `/mnt/data` 时直接拒绝；每个 raw 视频读取前后比较
  device、inode、size、mtime，变化即终止；所有派生物继续写入 workspace。
- 验证：166 项测试通过；提交 `5ca1cc1`。
- 结论：没有发现 raw 文件被本项目修改。后续恢复实验必须使用含硬保护的新进程。

## EXP-013：候选 clip 预解码热缓存

- 日期：2026-08-20。
- 目标：避免 few-B 容量实验和后续 SFT 反复从挂载视频随机 seek，同时不复制整段原视频。
- 协议：434 个独立 clip，每条均匀 8 帧，最长边 448，JPEG quality 82，8 个解码 workers；
  缓存只写入 workspace，index 不保存 raw 路径。
- 结果：434/434 缓存成功，共 3,472 帧；JPEG payload 71,878,578 bytes，目录占用约 82 MB；
  index SHA-256 为 `37d06820fec9d8e21641aa09f2d54d012ed4ccd3a28d3c7ba21c5adcb3d9dc9d`。
- 性能：414 条检查点时平均 0.463 秒/clip；首版最终 summary 漏记最终 elapsed，因此不伪造最终均值。
- 遇到的问题：启动时 `tee` 在 CLI 创建输出目录之前打开日志，导致 shell 日志未建立；内部
  `progress.json/index.jsonl` 不受影响。首版 summary 只保存容量，没有保存最终 elapsed。
- 解决：依赖原子滚动 progress/index 而不是 shell log 恢复；代码补充最终
  `elapsed_seconds/average_seconds_per_new_clip`；未来 runner 在启动前创建日志父目录。
- 缓存一致性：key 绑定 clip ID、起止时间、frame count、分辨率、JPEG 参数、raw 路径哈希以及
  device/inode/size/mtime；任一变化都使缓存失效。`--require-frame-cache` 禁止静默回退 raw 解码。
- 结论边界：这是派生热缓存，不是新标注，也不改变 raw 或验收结论。
- 证据：[`supplier434-frame-cache-v1/summary.json`](../artifacts/experiments/few-b-v2/supplier434-frame-cache-v1/summary.json)。
- 对应提交：`c1f58db`。

## EXP-014：供应商 434 条本地 8B 完成与弱标签分流

- 日期：2026-08-20。
- 恢复协议：从 386 条断点继续；剩余 48 条强制从 EXP-013 帧缓存读取；BF16 H20，8×448。
- 结果：434/434 成功，434/434 结构化 JSON 有效；最终 predictions SHA-256 为
  `ad72d8943cc2b1c941c3d043cbab6fbd3a093acf7d01ecde7151a925f2191673`。
- 缓存阶段性能：48 条在 38.57 秒 wall time 内完成，平均进度约 0.825 秒/clip，处理吞吐
  6.33 video-h/wall-h；较最初 6.565 秒/clip 约快 8 倍。模型权重指纹仍额外耗时 15.08 秒，
  模型加载耗时 4.53 秒，尚未缓存。
- 训练治理结果：434 条进入 train-only all pool；其中 206 条进入 high-confidence view；228 条
  因规则/模型分歧进入人工复检；0 跳过、0 重复。强正例以 `action_not_observable` 为主，说明
  base 8B 的标签分布仍偏置，不能直接当 Gold。
- 文件 SHA-256：all pool
  `2be3f3f41bb002200f2b6f30afb623bf80f80800b6371a8114a12365b23ef8f5`；high-confidence
  `03df268219cfd9c886d5400c33aba115ba80be7d1c02ce0fafad99c0a1c43a1d`；human-review
  `45a36f2e0a1deb128881bcba9df80f0d70ec369f927bb22f783860aea1d378d7`。
- 结论边界：206 条只是高置信弱标签；228 条必须人工判断；任何一条都不能进入正式 test Gold。
- 证据：[`supplier434-complete-v1/summary.json`](../artifacts/experiments/few-b-v2/supplier434-complete-v1/summary.json)。

## 已知未解决问题与下一步

1. 完成人工 Gold：每类至少覆盖正/负例、双人独立标注和第三人仲裁；按原视频/person/session/
   supplier 分组切分。
2. 在 Web 面板人工复检 228 条规则/模型分歧，并从中建立第一批跨来源 Gold；不得把弱标签直接当 Gold。
3. 对 2B/4B/8B 做同一 Gold validation/test 的 SFT 对照；报告 99% precision 下 recall、置信区间和 worst-group。
4. 补 2B/4B 的 CPU INT8/INT4 与 GPU BF16/INT8 对照；模型指纹应缓存，避免每次 resume 重算约 15 秒。
5. 为百万小时部署增加 shard 级 NVMe 热缓存、解码队列背压、worker 心跳和孤儿进程回收。
6. 将每次 CLI 运行的摘要自动追加到本日志的 JSONL，而不是依赖聊天记录或人工回填。

## 新实验最小记录模板

```text
实验 ID / 日期 / 负责人：
研究问题与预注册指标：
输入 manifest + SHA-256 + split-group 规则：
代码 commit / 容器 / GPU / 依赖：
模型、prompt、分辨率、帧数、seed、并发：
开始/结束/中断/恢复时间：
成功、失败、拒答、缓存数量：
P50/P95、吞吐、CPU/GPU/内存：
遇到的问题 / 根因 / 解决方案：
结果与结论边界：
产物相对路径与 SHA-256：
```

## EXP-015：普遍性证据审计

- 结论：434-clip few-B 实验降级为 systems pilot，只用于证明缓存、解码与推理链路可运行，不作为准确率或跨域普遍性证据。
- 开发机只读盘点：公共数据 2,856 rows / 2,856 `split_group`；三批供应商数据 1,800 / 308；内部 OSS 数据 378 / 126；总计 5,034 clip / 3,290 独立原视频组。
- 发现问题：供应商与内部队列含同一原视频多 clip；按 clip 随机划分会产生泄漏和虚高指标。
- 解决方案：预注册 100,000 独立 clip 系统池、30,000 独立 clip 训练池、10,000 独立 clip 人工 Gold 池；Gold 按 validation 2,500 / 同域 test 3,500 / 留源 external test 4,000 划分。
- 统计约束：每任务自动正判若零假阳性，至少 381 次才能使 95% Wilson precision 下界超过 0.99；不足时明确报告 evidence insufficient。
- 数据安全：本次仅读取 `/mnt/workspace` 队列元数据，未修改 `/mnt/data` OSS/CPFS 原始数据。
- 协议：`docs/generality-evaluation-protocol.md`；机器配置：`config/qc_generality_protocol_v1.json`。

## EXP-016：共享挂载别名安全更正

- 发现：`/mnt/data` 与 `/mnt/workspace` 的 realpath 名称不同，但设备号、inode、挂载源与一级目录完全相同，实际为同一 CPFS 根的两个别名。
- 影响：仅检查输出字符串是否位于 `/mnt/data` 下会漏掉 `/mnt/workspace/oss/...` 这类对同一原始目录的别名写入。
- 处置：暂停远端 cohort 生成；保护逻辑改为原始子命名空间 + 设备号/inode 别名识别；后续派生产物统一进入 `/mnt/workspace/egoqc-derived`。
- 本轮远端动作：仅只读执行 `realpath`、`stat`、`findmnt`、一级目录枚举；此前只更新了 `/mnt/workspace/ie-qc-code` 代码，尚未运行 cohort 规划器。
- 首次 inode 自检发现 `/mnt/data/oss` 是覆盖在共享根上的独立 OSS 子挂载，而 `/mnt/workspace/oss` 指向底层 CPFS 目录；两者子目录 inode 不同。保护逻辑因此进一步加入“声明式挂载根别名 + 相对路径映射”，即使子挂载遮蔽了 inode 关系，也拒绝两个拼法下的 raw 命名空间。
