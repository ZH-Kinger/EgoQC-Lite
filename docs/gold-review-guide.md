# Episode Gold 人工标注指南

这套任务直接运行在现有 PostgreSQL 复检台中。一个任务对应一条真实 episode；原视频、
MANO 叠加和规则数值是同一个任务的不同证据视图。面板只写审核数据库，原始 OSS/CPFS
保持只读，短视频写入 `/mnt/workspace` 派生区。

## 默认生产审核顺序

1. 完整播放原视频，并按需切换 MANO 叠加。面板会直接显示机器建议和命中的底层规则。
2. 机器判断正确时点“确认机器结论”。面板自动保存所有逐规则答案，不再要求人工重复填写
   可见问题、原因、时间点和置信度。
3. 发现误报时点“有误报，展开修改”。所有规则默认是真错误，只修改误报或无法判断的项，
   然后点“保存修正”。
4. 证据不足时点“无法判断”，任务进入仲裁。提交后卡片从“我的任务”消失，结构化答案与
   可选备注进入 append-only 审计历史。

这个快速流程产出的是人工确认的 assisted label。为了避免 confirmation bias，仍需从每批
任务随机抽取 5%-10% 使用双人独立盲审和第三人仲裁；只有这部分锁定样本可以作为论文
Gold validation/test 和准确率声明依据。

## 最容易混淆的边界

- **真实快速动作不是追踪跳变。** 真实动作中手、物体和局部纹理连续移动；追踪跳变通常是
  mesh/骨骼突然离开手后又返回，或者手没动但标签瞬移。速度规则在穿鞋带、快速抓放等任务
  上容易产生 hard negative，必须单独标 `true_fast_motion`。
- **相机运动不是手抖。** 画面中背景和双手一起运动，优先判断相机运动；背景稳定而只有
  mesh 跳动，优先判断手部追踪问题。SLAM 发散时背景视频可能正常，但世界系双手和相机
  轨迹会同时异常，应在备注中写明“疑似坐标系/SLAM”。
- **遮挡不等于坏标注。** 手被物体短暂遮挡但 mesh 仍合理延续，可以保留；mesh 穿过物体、
  贴到另一只手或遮挡后长期漂移，才标 MANO overlay drift。
- **离画按真实手判断。** 不要只看 `state_mask` 或 mesh。真实手离画超过验收标准才标
  `hand_absent`；真实手仍在画面而 mesh 消失，是标注缺失/追踪问题。
- **左右手交换看时间连续性。** 交叉操作本身不是左右手交换；只有同一只真实手对应的颜色/
  identity 突然切换，才标交换。
- **模糊按可用性判断。** 有运动模糊但仍能判断接触、操作对象和手指状态，可作为难例保留；
  无法判断关键动作或 overlay 是否正确时选“不确定”，不要硬猜负例。

## 各规则怎么判断

| 规则 | 标“真错误”的可观察证据 | 常见误报 |
|---|---|---|
| 瞬时速度异常 | mesh/腕点单帧瞬移、跳回、穿越物体 | 合理的快速穿带、抓放、甩动 |
| 单帧轨迹跳点 | 前后帧连续，但中间一帧明显偏离 | 视频本身掉帧或极快动作 |
| 手腕位置抖动 | 背景稳定时腕点高频来回跳 | 细小但真实的手腕修正动作 |
| 手腕旋转抖动 | 手形连续而腕坐标轴/mesh 突然翻转 | 手腕快速翻面 |
| 手指旋转抖动 | 指节 mesh 抽动、穿模、局部翻转 | 快速捏取、穿孔时的真实屈伸 |

`bad_frame_ratio_exceeded` 不要求人工单独判断。它是聚合量：管线会在底层坏帧完成
真/误报复核后重新计算是否超过 3%，避免审核员凭肉眼估比例。

## Gold 质量要求

- 生产快速审核不要求手填时间点。需要补充证据时，备注中的时间按 episode 内播放器时间，
  不要填写源聚合 MP4 的绝对时间。
- “不确定”是有效标签：遮挡、严重模糊或 overlay 不可用时必须使用，后续进入仲裁，不进入
  自动拒收阈值训练。
- 备注只写可复核事实，例如“12.3s 左手 mesh 跳到鞋外，12.4s 返回”；不要只写“感觉不准”。
- 研究/论文 Gold 需要两名审核员独立标注和第三人仲裁。当前单人 pilot 可以用于修正规则和
  准备训练候选，但不能直接宣称 99% 准确率。
- 相同人物、采集 session、相邻片段和同一源视频不能跨 train/validation/test；人工 Gold
  优先级高于教师 API 和程序化弱标签。

## 生成并导入任务

```bash
egoqc build-phase-a-review-events /mnt/data/readonly/lerobot-v3 \
  --baseline-evidence /mnt/workspace/ie-qc/phase-a/evidence/baseline-evidence.jsonl \
  --annotated-root /mnt/workspace/ie-qc/phase-a/annotated \
  --output /mnt/workspace/ie-qc/phase-a/postgres-review

egoqc import-review-events \
  /mnt/workspace/ie-qc/phase-a/postgres-review/review-events.json \
  --dataset-name phase-a-real-baseline \
  --run-name gold-pilot-v1 \
  --source-root /mnt/data/readonly/lerobot-v3

egoqc serve-postgres-review \
  --evidence-root /mnt/workspace/ie-qc/phase-a \
  --host 127.0.0.1 --port 8767
```
