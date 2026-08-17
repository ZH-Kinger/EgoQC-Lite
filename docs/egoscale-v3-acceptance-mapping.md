# EgoScale V3.0 验收标准落地映射

本文把《EgoScale 数据交付验收 V3.0》转换为 EgoQC-Lite 可执行合同。机器可读版本位于
`config/egoscale_v3.contract.json`。该合同独立于当前 LeRobot 扫描配置，不能直接作为
`egoqc scan --config` 使用。

## 1. 判定边界

- 强制指标缺失或实测失败，可以触发整改或不合格。
- 文档只有测试值、没有要求的项目只作为参考，不单独否决。
- 依赖 GT、人工参考或同步硬件的指标，在参考数据缺失时必须输出 `null`。
- `null` 不等于通过，也不等于失败；报告必须同时给出缺失的输入与补测方法。
- 原始交付不可修改。整改后必须登记新的 dataset revision。

## 2. 世界坐标系约定

V3.0 允许供应商采用以下两种世界系，交付时必须声明：

1. `first_valid_camera`：第一有效帧相机系直接作为世界系，首帧 pose 为 identity。
2. `gravity_aligned`：原点仍为第一有效帧相机光心，Y 轴按重力向下，X 轴由首帧相机
   右方向投影到重力平面得到。

这两种模式不能在相机存在 pitch/roll 时同时满足“旋转为 identity”和“严格重力对齐”。
报告不得把二者混为同一个检查项。第一有效帧选择必须记录背景跟踪点数、背景光流、
手部置信度阈值、连续稳定帧数和最终帧号。

## 3. 当前实现覆盖矩阵

| V3.0 项目 | 当前状态 | EgoQC 输出/后续动作 |
|---|---|---|
| LeRobot v3 结构与 Schema | 已实现 | 结构、dtype、shape、路由硬门禁 |
| FPS、分辨率、视频可解码 | 已实现 | header/count/sample-quality |
| 帧间隔 Jitter | 已实现 | 数值 label timestamp 与视频 PTS 分别输出 mean/max ms、非递增计数与异常帧 |
| Et 图像/标注同步 | 部分实现 | label 对名义轴已输出；没有逐帧视频 PTS 时 `time_alignment_error_max_ms=null`，不伪判通过 |
| Vt + 3MAD | 已实现 | 使用逐帧 timestamp dt 计算速度，输出 median、MAD、limit、异常帧与左右手比例 |
| 时序抖动和瞬移 | 已实现 | wrist/joint/camera 局部残差 p99 与孤立跳点 |
| MANO 重投影目视证据 | 已实现 | 原图、mesh、骨骼、标注视频和人工复检 |
| 单轴/三轴旋转参考误差 | 待参考 | 无人工/GT 参考时为 null |
| MPJPE 与各指尖 MPJPE | 待 GT | 无 3D GT 时为 null |
| ODSR Micro-F1 | 待 Gold Set | 需要人工遮挡真值，不能用 valid rate 替代 |
| 坏帧比例与坏帧清单 | 核心实现 | `bad_frames.jsonl/parquet` 覆盖数值、几何、时序和长时离开视野；模糊/曝光仍是视频抽样候选 |
| 中英文 Distinct-2 | 待实现 | 按 Qwen2-0.5B tokenizer 固定版本计算 |
| Pairwise Distance | 待实现 | TF-IDF 1-2gram、词表5000、固定随机种子 |
| 语义标注准确率 | 待人工抽检 | 至少记录样本 manifest、审核人和混淆原因 |
| 任务/场景 episode <30 | 待元数据 | 需要稳定 scene_id/task_id，不能只用自然语言分组 |
| 视觉多样性 | 参考项 | 未给硬阈值，不触发否决 |
| 深度指标 | 验收范围外 | 仅供应商内部参考 |
| ATE/RPE | 参考/待 GT | V3.0 总表未给 ATE 硬阈值，不触发否决 |

## 4. EgoDex 初步映射

EgoDex 原始数据是任务目录下同名 `.hdf5 + .mp4`，不是 LeRobot v3。抽查结果显示：

- HDF5 与视频帧数一致，30 FPS、1920x1080、无音轨；
- HDF5 提供相机内参、逐帧相机/手/手指 4x4 transform 和逐关节 confidence；
- 旋转矩阵合法且样本中没有 NaN/Inf；
- 视频编码为 MPEG-4 Part 2，需要为 Web/标准化交付生成 H.264 派生代理；
- 有英文 task/LLM description、环境与物品属性，未发现中文和原子动作时间段；
- 没有原生 MANO pose/betas，MANO mesh 只能作为拟合后的派生结果；
- 没有发现 3D GT、遮挡 Gold Set、IMU 重力来源，因此 MPJPE、ODSR、ATE 暂为 null。

接入顺序应为：EgoDex 只读 adapter、HDF5/MP4 配对门禁、confidence 可见性、
timestamp/Jitter/Vt、骨骼证据、20DoF 映射，最后才是可选 MANO 拟合。

## 5. V3.0 报告最小结构

每个指标至少保存：

```json
{
  "metric": "hand_mpjpe",
  "status": "null",
  "value": null,
  "unit": "mm",
  "requirement": "<=10",
  "reason": "ground_truth_missing",
  "sample_manifest": null,
  "implementation_version": null
}
```

汇总结论只从强制指标计算，并同时保留原始问题：

- `qualified`：全部强制项通过，抽检无系统性错误；
- `remediation`：少量可在不重采条件下修复的问题；
- `unqualified`：关键算法指标明显失败、组织不完整或存在系统性错标。

内部路由保持 `accept/review/rework/reject/quarantine/retry/escalate`，对外交付时映射为
上述三种合同结论，避免丢失可操作的失败原因。

## 6. 尚需书面确认

1. 世界系采用 `first_valid_camera` 还是 `gravity_aligned`，不得同时强制首帧旋转 identity。
2. MP4 容器是否足够，还是标准化交付必须 H.264。
3. “坏帧”是否包含模糊、曝光、遮挡、解码失败、pose 失败及时间不同步。
4. ODSR 的正类定义、逐帧/逐手统计方式与人工 Gold Set 标注规范。
5. MPJPE 是否腕部 root-relative，指尖和全手是否都以未对齐 MPJPE 作为硬门禁。
6. 语言抽样不足 3000 条时，是全量计算还是判交付量不足。
7. 总时长 500 小时允许的统计误差、去重规则与有效时长口径。
