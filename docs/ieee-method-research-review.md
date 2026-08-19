# EgoQC-Lite 投稿前方法调研与研究边界

更新时间：2026-08-19

本文只定义研究问题、先验工作、可辩护的新方法和证伪条件。它不是论文结果，也不表示当前
系统已经达到 IEEE 投稿或生产 99% 准确率。正式结论只能来自锁定的人工 Gold test、跨来源
外部测试、重复实验和下游机器人学习验证。

机器可读的预注册主张与否决条件位于
[`config/qc_research_claims_v2.json`](../config/qc_research_claims_v2.json)。

## 1. 结论先行

不建议把论文写成“一个包含规则、VLM、MANO、蒸馏、主动学习和 Web 复检的完整平台”。
这些组件分别都有成熟先例，简单组合容易被审稿人判断为系统集成。

建议聚焦如下研究问题：

> 在缺少逐帧真值、证据源相关、模态不完整且错误类型持续演化的情况下，能否利用可控数据
> 破坏作为受控干预，学习条件化证据可靠性，并在给定误收风险和计算预算下，对第一人称具身
> 数据进行时序定位、开放集质检和选择性自动验收？

建议方法暂称 **Interventional Evidence QC (IE-QC)**，由三个不可拆散的部分组成。这里的
“interventional”只表示对数据生成或标注通道施加已知可控变换；在没有额外结构假设、
识别性证明和真实干预验证前，不宣称恢复了真实世界因果图。

1. **Interventional Reliability Learning**：利用时间偏移、姿态冻结、坐标变换、相机跳变等
   已知干预学习各证据源的条件可靠性，降低对弱监督源条件独立假设的依赖。
2. **Availability-aware Temporal Evidence Graph**：显式处理缺失模态和持续时间，联合推断
   逐帧已知错误、未知异常和可靠性，而不是逐帧独立分类或固定投票。
3. **Risk-bounded Sequential Acquisition**：根据预期风险下降/成本按需调用昂贵专家；只有
   校准后风险满足要求才自动决策，否则升级到更多专家或人工复检。

机器人数据价值、语义去重、自动修复和技能覆盖仍有产品价值，但不应同时成为这篇论文的
主要方法贡献。

## 2. 相关工作矩阵

### 2.1 数据集构建与几何质量证据

| 工作 | 已经解决的问题 | 对 EgoQC 的约束 | 不能声称为本项目原创的内容 |
|---|---|---|---|
| [DROID](https://arxiv.org/abs/2403.12945) | 大规模分布式机器人采集、跨场景和跨采集者数据 | 数据质量必须最终用策略性能和泛化验证 | 大规模采集、数据多样性本身 |
| [HOT3D](https://arxiv.org/abs/2411.19167) | 多视角 ego 手物跟踪、相机/手/物体真值与 MANO | 应使用多视角或 mocap 子集作为几何外部测试 | MANO、可见性和 3D 跟踪有效性元数据 |
| [Ego-Exo4D](https://arxiv.org/abs/2311.18259) | ego/exo 多视角、人工与自动标注分层、置信证据 | 自动标签、插值标签和人工 Gold 必须保持来源差异 | provenance-first 标注分层 |
| [HaWoR](https://arxiv.org/abs/2501.02973) | 相机系手重建、世界系相机轨迹和缺失手运动补全 | 手、相机、尺度和补全质量必须分开评估 | 世界系手运动解耦和 motion infilling |
| [UniHOI](https://arxiv.org/abs/2411.09145) | 联合估计内参、位姿、深度和 4D HOI 一致性 | 几何一致性不能只依赖单个 MANO 投影阈值 | 统一 4D HOI 重建 |
| [StableHand](https://arxiv.org/abs/2605.18553) | 预测双手腕部/手指四通道观测质量并据此重建不可靠片段 | 2026 预印本，是质量感知手运动建模的直接 challenger | 学习逐帧手部观测质量本身 |
| [HandsOnWorld](https://arxiv.org/abs/2607.02075) | 从动作语义、画质和 3D 几何三个层面过滤 monocular 重建 | 2026 预印本，是多层 ego 清洗的直接 challenger | 语义+图像+几何三级过滤概念 |
| [EgoInfinity](https://arxiv.org/abs/2606.17385) | Web 级视频到 4D HOI、交互优化和机器人重定向的数据引擎 | 2026 预印本，限制“首个通用数据引擎”类主张 | 模块化 video-to-action 数据引擎和跨 embodiment 重定向 |
| [Benchmarking 2D Egocentric Hand Pose Datasets](https://arxiv.org/abs/2409.07337) | 从数据特征和模型表现两侧评估 ego 手姿态数据集 | 需要跨数据集、跨模型的质量评估协议 | “用下游手姿态模型反查数据问题”的一般框架 |

启示：重投影、可见性、轨迹平滑、相机/手解耦都已有明确先例。EgoQC 的潜在新意不能是
“首次使用 MANO、光流或 SLAM 检查数据”，而应是如何在无真值、证据相关和模态缺失时
联合推断这些证据的可信度。

### 2.2 弱监督、多源标签和时序异常

| 工作 | 核心机制 | 可复用部分 | 剩余研究空白 |
|---|---|---|---|
| [Snorkel](https://arxiv.org/abs/1711.10160) | 估计未知准确率和相关性的 labeling functions | 把规则、教师、人工视为不同弱监督源 | ego 证据可靠性会随遮挡、设备和模态可用性变化 |
| [Learning Dependency Structures for Weak Supervision](https://arxiv.org/abs/1903.05844) | 学习弱监督源依赖结构 | 不应假定相似规则相互独立 | 仅靠观测相关性仍难识别条件化可靠性 |
| [Mitigating Source Bias for Fairer Weak Supervision](https://arxiv.org/abs/2303.17713) | 用 counterfactual fairness 建模和修正弱监督源偏差 | 反事实/干预用于弱监督并非空白 | 尚未处理 ego 时序、几何证据和可控标注通道破坏 |
| [UMIL](https://arxiv.org/abs/2303.12369) | 从 episode 弱标签定位 snippet 异常并缓解上下文偏差 | episode 标签到局部错误定位 | 不包含 MANO、相机和物理证据 |
| [OpenVAD](https://arxiv.org/abs/2208.11113) | MIL + evidential learning + normalizing flow 发现已知/未知异常 | 开放集不确定性和伪异常 | 面向监控异常，不解决具身标注完整性 |
| [Temporal Corruption Robustness](https://arxiv.org/abs/2403.20254) | 丢帧、模糊等时序污染基准与一致性训练 | 建立可控 corruption benchmark | 不推断数值标注与视频的跨模态矛盾 |

启示：弱监督标签融合、MIL 和开放集异常都不是新概念。可辩护的差异是将**已知数据破坏视为
干预变量**，用干预前后的专家响应识别“哪个证据源在什么条件下可靠”，并把结果放入具有
持续时间和模态可用性的时序图模型。

### 2.3 视频表征与 ego 预训练

| 工作 | 适合承担的角色 | 不适合作为的角色 |
|---|---|---|
| [VideoMAE V2](https://arxiv.org/abs/2303.16727) | 稳定、可复现的 RGB 视频基线 | 不能单独验证几何标注正确性 |
| [InternVideo2](https://arxiv.org/abs/2403.15377) | 强视频语义教师或 challenger | 不宜成为唯一基线或唯一真值源 |
| [EgoVLP](https://arxiv.org/abs/2206.01670) | ego 视频-文本预训练和 EgoNCE 对照 | 与 VITRA/去畸变无关；不能当成手部重建论文 |
| [V-JEPA 2](https://arxiv.org/abs/2506.09985) | 动作无关视频表征、预测残差和世界模型教师 | 预测困难不能直接等价为坏数据 |

重要更正：arXiv:2206.01670 是 **Egocentric Video-Language Pretraining (EgoVLP)**，不是
VITRA 去畸变或手部重建方法。后续文档和论文引用必须按真实题目与贡献使用。

### 2.4 机器人数据质量与训练价值

| 工作 | 质量定义 | 与本研究的关系 |
|---|---|---|
| [Data Quality in Imitation Learning](https://arxiv.org/abs/2306.02437) | action divergence 与 transition diversity | 说明“视觉干净”不等于“策略训练价值高” |
| [DemInf](https://arxiv.org/abs/2502.08623) | 轨迹对状态-动作互信息的贡献 | 作为下游数据价值基线，不作为标注正确性检测器 |
| [Demo-SCORE](https://arxiv.org/abs/2503.03707) | 根据在线机器人成功/失败经验反查示范质量 | 说明真实策略结果是强验证信号 |
| [QoQ](https://arxiv.org/abs/2603.09056) | 训练样本对验证损失的 influence | 2026 预印本，可作 challenger，不能声称 influence 估值是原创 |
| [JEST](https://arxiv.org/abs/2406.17711) | learner-reference loss 差与联合 batch 选择 | 训练价值选择应与当前 learner 绑定 |
| [For-Value](https://arxiv.org/abs/2508.10180) | 前向式数据价值估计 | 可作大规模估值参考，不替代真实下游消融 |
| [SemDeDup](https://arxiv.org/abs/2303.09540) | 语义 embedding 去重 | 只解决冗余，不解决标注正确性 |
| [Rethinking Data Shapley](https://arxiv.org/abs/2405.03875) | 分析数据估值用于选择时的失效与不稳定 | 任何 value score 都必须用真实重训练/策略结果验证 |

启示：本论文应把“数据质量检测”和“下游训练价值”严格分开。前者是主要方法，后者是验证：
在相同训练算力下，IE-QC 选择或 mask 后的数据是否改善策略成功率。

### 2.5 选择性预测与成本级联

| 工作 | 已有结论 | 对 IE-QC 的要求 |
|---|---|---|
| [Conformal Risk Control](https://arxiv.org/abs/2208.02814) | 可对单调损失的期望风险做分布无关校准 | 不能把经验 99% 当统计保证；必须保留校准集 |
| [Agreement-Based Cascading](https://arxiv.org/abs/2407.02348) | 根据模型间一致性将难例升级到昂贵模型 | “小模型不确定就调大模型”不是原创 |
| [Video Test-Time Adaptation](https://arxiv.org/abs/2211.15393) | 时序增强一致性可改善视频分布偏移 | 在线适配必须避免错误伪标签累积 |

启示：论文的新意不能只是级联。需要把**风险校准、可选证据获取和具身证据图**统一为明确的
序贯决策目标，并与简单 confidence threshold、固定 cascade 和全专家上限比较。

### 2.6 人类 ego 视频到机器人学习

| 工作 | 证据 | 论文中应承担的作用 |
|---|---|---|
| [HRP](https://arxiv.org/abs/2407.18911) | 手、物体和接触 affordance 预训练在多机器人任务上带来提升 | 证明手物局部证据有机器人相关性 |
| [HumanEgo](https://arxiv.org/abs/2605.24934) | 实体级 HOI 表示用于跨 embodiment 学习 | 2026 预印本，只作最新 challenger/趋势依据 |
| [Human2Sim2Robot](https://arxiv.org/abs/2504.12609) | 物体轨迹和人手初始姿态用于 sim-to-real | 支持物体状态变化比单纯 MANO 误差更接近任务价值 |

启示：如果没有下游机器人或可信模拟器实验，RA-L 审稿人可能把工作视为一般视频质量系统，
而不是机器人学习贡献。

## 3. 新颖性审计

### 3.1 不能单独作为论文贡献的内容

- 使用大模型给视频打标签；
- 蒸馏一个小视频模型；
- 多专家投票或 stacking；
- 用 MIL 从 episode 标签定位 clip；
- 用 normalizing flow/energy 做未知异常；
- 用 conformal prediction 做拒答；
- 用 confidence threshold 触发昂贵模型；
- 用 MANO 重投影、光流、SLAM 或速度阈值检查数据；
- 用 influence、互信息或 embedding 评估数据价值；
- 提供 Web 复检、PostgreSQL、S3/NFS 和 LeRobot 转换。

它们可以作为系统组件、基线或工程贡献，但相关工作中都有直接先例。

### 3.2 候选核心创新：干预式证据可靠性学习

传统弱监督标签模型从多个 labeling function 的一致/冲突关系推断源准确率，常依赖条件独立、
少量 Gold 或正确的依赖结构。Ego 数据中这些假设尤其脆弱：

- `OUT`、手部检测和 MANO overlay 都共同依赖手是否可见；
- 光流、抖动和 SLAM 都共同依赖相机运动与纹理；
- VLM、手检测器和 segmentation 可能共享相似预训练数据；
- 模态缺失不是随机的，通常与特定供应商、设备或处理失败有关。

IE-QC 的核心做法是对同一个样本构造已知干预：

\[
x^{(a,k,I)} = \operatorname{Corrupt}(x; a, k, I),
\]

其中 \(a\) 是干预强度，\(k\) 是错误机制，\(I\) 是被影响的时间区间。例子包括：

- 视频相对数值标注偏移若干帧；
- 冻结手指姿态但保留腕部运动；
- 冻结腕部或相机轨迹；
- `w2c/c2w` 反转、单位缩放、左右手交换；
- 旋转布局或镜像约定错误；
- 丢帧、重复帧、局部模糊、遮挡；
- 内参扰动和可逆镜头畸变。

收集每个专家在干预前后的响应差：

\[
\Delta e_{i,t}^{k,a}=e_i(x_t^{(a,k,I)})-e_i(x_t).
\]

可靠性网络不只学习专家是否与伪标签一致，而学习：给定可见性、设备、模态、运动状态和干预
类型，哪个专家应该对目标错误产生响应、哪个专家应该保持不变。

\[
r_{i,t}=\sigma(g_i(c_t,m_t,e_{i,t})),
\]

其中 \(c_t\) 是观测上下文，\(m_t\) 是模态可用 mask。干预损失包含：

1. **Sensitivity**：目标专家对相关干预的响应达到 margin；
2. **Specificity**：无关错误头在干预前后保持一致；
3. **Temporal localization**：响应集中在已知区间 \(I\)；
4. **Monotonicity**：在合理范围内，干预强度增加不应降低目标异常分数；
5. **Source calibration**：少量 Gold 上校准真实错误概率，修正 synthetic-to-real gap。

这一机制的研究价值在于：可控干预为源可靠性提供额外识别信号，而不是仅从弱标签的静态
相关性猜测可靠性。

### 3.3 候选核心创新：可用性时序证据图

定义隐状态 \(z_t\) 包含正常、若干已知错误和未知异常；显式建模 segment 持续时间：

\[
p(z_{1:T}\mid E,M,X) \propto
\exp\{-E_{unary}-E_{transition}-E_{duration}-E_{consistency}\}.
\]

- `unary`：条件可靠性加权后的专家证据；
- `transition`：遮挡→不确定→丢失→惯性外推→重新捕获等合理转移；
- `duration`：时间偏移、冻结、SLAM 发散不应表现为孤立随机帧；
- `consistency`：图像、MANO、相机、手物接触和文本的跨模态残差；
- `availability`：缺失模态关闭相关因子，不允许以零向量冒充“正常”。

实现上首先采用 segmental Transformer 或 semi-Markov CRF；不应一开始同时实现多个复杂图
模型。与 frame-wise MLP、BiGRU、Transformer、普通 CRF 做固定对照。

### 3.4 候选核心创新：有风险约束的序贯证据获取

每个样本不再固定运行全部专家。状态包含当前后验、已用专家、模态可用性和累计成本；动作是
停止/拒答或运行下一个专家：

\[
j^*=\arg\max_j
\frac{\mathbb E[R_{before}-R_{after}\mid e_j]}{C_j}.
\]

最终不是承诺所有样本 99% 正确，而是报告风险-覆盖率-成本曲线：

- 自动处理覆盖率；
- 自动通过/拒收的 95% 风险上界；
- 每视频小时的 CPU/GPU/API/人工成本；
- 相对全专家上限的性能差；
- 相对简单 confidence cascade 的成本差。

第一版可用监督式 router 学习全专家离线 trace；只有它显著优于固定顺序和单阈值 cascade 后，
才考虑 contextual bandit 或强化学习。

## 4. 训练目标

建议总损失为：

\[
\mathcal L =
\mathcal L_{gold}
+\lambda_w\mathcal L_{weak}
+\lambda_i\mathcal L_{intervention}
+\lambda_t\mathcal L_{segment}
+\lambda_o\mathcal L_{open}
+\lambda_c\mathcal L_{calibration}.
\]

- `gold`：人工 Gold 的逐帧/区间/episode 多任务损失；
- `weak`：规则、几何专家和教师模型的概率标签，不将任一教师当真值；
- `intervention`：敏感性、特异性、定位和强度单调性；
- `segment`：边界、持续时间和状态转移；
- `open`：已知错误与未知/分布外异常的 energy 或 evidential loss；
- `calibration`：Brier/ECE 辅助优化，最终阈值仍只由独立 validation 校准。

不建议首版加入所有可能损失。主实验按下列累积顺序消融：

1. Gold-only RGB baseline；
2. + weak teachers；
3. + conditional reliability；
4. + intervention supervision；
5. + segment model；
6. + open-set head；
7. + selective risk；
8. + cost router。

## 5. 论文主张与证伪条件

| 候选主张 | 必需证据 | 否决条件 |
|---|---|---|
| 干预监督改善真实错误识别 | 真实、未见供应商 Gold test；与 synthetic-only augmentation 和普通 label model 比较 | 只在合成测试提升，真实测试置信区间无改善 |
| 条件可靠性优于固定投票/stacking | 至少三个来源、模态缺失和遮挡切片 | 提升只来自增加参数或额外 Gold 数据 |
| segment graph 改善局部定位 | frame/interval AP、boundary F1、segment IoU | episode 分类提高但时间定位无提升 |
| open-set 能发现新错误 | 按错误家族留一法和真实未知错误 test | 不能超过 max-softmax、energy 或 OpenVAD 风格基线 |
| 序贯获取降低成本 | 相同风险或相同 recall 下比较 CPU/GPU/API 成本 | 不优于单 confidence threshold 或固定 cascade |
| 质检对机器人训练有价值 | 相同训练预算下，原始/规则/IE-QC 三组策略结果 | 仅 QC 指标提高，下游策略无统计可信提升 |

如果前三项中至少两项不成立，不应把工作包装成新算法论文；应转为工程系统、数据报告或负面
结果研究。

## 6. 实验协议建议

### 6.1 数据划分

- 训练：人工 Gold + teacher Silver + programmatic Weak + synthetic interventions；
- validation：只使用人工 Gold，用于选择模型、校准和冻结阈值；
- test：只使用锁定人工 Gold，只运行一次固定模型与固定阈值；
- external test：至少一个未见供应商或未见相机；
- open-set test：按错误**家族**留一，而不是随机留下相邻 corruption 强度；
- public test：使用有公开许可、可复现的数据子集；内部供应商数据只报告不可逆统计；
- group split：supplier/person/session/source video 及其派生 clip 不得跨集合。

### 6.2 Gold 标注

- 两名独立复检者 + 分歧仲裁；
- 同时标注错误类别、开始/结束时间、严重度、可修复性和证据类型；
- 报告 Cohen's kappa 或 Krippendorff's alpha、分歧率和仲裁率；
- 不强迫复检者对不可判断样本给二元标签，保留 `uncertain/unmeasurable`；
- teacher、规则和合成标签不得进入 validation/test 真值。

### 6.3 基线

1. Deterministic rules；
2. VideoMAE V2 global RGB；
3. global RGB + hand/object ROI；
4. BiGRU/temporal Transformer；
5. 普通 MIL/UMIL 风格局部化；
6. Snorkel-style label model；
7. fixed weighted vote；
8. out-of-fold stacking；
9. OpenVAD-style open-set baseline；
10. strongest VLM teacher；
11. full expert oracle upper bound；
12. IE-QC student/cascade。

### 6.4 指标

主要指标：

- 每类 recall at validation-selected precision；
- precision 是否达到目标以 95% Wilson 下界判断；
- supplier/person-session clustered bootstrap；
- interval AP、segment IoU、boundary F1；
- open-set AUROC/AUPRC/FPR95；
- selective risk-coverage curve 与 AURC；
- worst supplier/camera/source group；
- video-hours/GPU-hour、峰值显存、API 请求率、人工分钟/小时视频。

下游指标：

- 固定训练步数和固定数据预算下的策略成功率；
- 未见物体、背景、相机和操作者的成功率；
- 失败恢复与长时任务完成率；
- 三个或更多随机种子的均值、标准差和配对置信区间。

## 7. 投稿范围

### RA-L/ICRA 候选

一篇 6--8 页论文只保留：干预式可靠性学习、时序证据图、风险约束级联，以及一个明确的
机器人下游实验。Web、存储、格式转换、自动修复和数据市场不进入主要方法图。

### 更长的 T-RO 候选

只有在以下条件成立后考虑：

- 多来源大规模 Gold benchmark；
- 理论或可证明的风险性质；
- 多机器人/多策略/多任务下游验证；
- 对人工成本、系统吞吐和分布漂移进行完整长期分析。

### 不建议当前直接投 T-PAMI

除非方法被证明不仅适用于 ego/robot QC，还在多个通用多模态时序弱监督与开放集 benchmark
上成立，否则通用性和方法深度不足。

## 8. 实现顺序与停止点

### Phase A：证明干预信号有效

1. 冻结 5--7 个错误家族；
2. 在不修改源文件的前提下生成可逆 corruption manifest；
3. 运行现有规则、手部、几何和视觉教师，保存干预前后 evidence delta；
4. 人工复核一小批真实对应错误；
5. 验证 synthetic intervention response 是否能预测真实专家可靠性。

停止条件：如果 intervention response 与真实 Gold 的专家正确性没有稳定相关性，不继续复杂
图模型，改做普通监督基线。

### Phase B：条件可靠性与时序模型

1. 先做 fixed vote、stacking、Snorkel-style label model；
2. 再加 context-conditioned reliability；
3. 最后加入 segment duration 和 open-set head；
4. 全部使用相同 backbone、Gold 数量和训练算力。

停止条件：如果条件可靠性无法在 external test 上超过 stacking，则不把它作为论文贡献。

### Phase C：风险和成本

1. 保存每个专家的离线执行时间、资源和输出；
2. 训练监督式 next-expert router；
3. 在固定风险下比较 fixed cascade、confidence cascade 和 IE-QC；
4. validation 校准，test 只运行一次。

停止条件：如果 router 不能在相同风险下显著降低成本，生产可继续使用固定 cascade，论文删除
该主张。

### Phase D：机器人相关性

固定同一策略结构、训练步数和随机种子，比较：

- 原始数据；
- 规则 gate 后数据；
- IE-QC hard filter；
- IE-QC interval mask/weight；
- 随机删除等量数据控制组。

只有 IE-QC 在相同数据或计算预算下带来稳定下游提升，才能声称“质量评估改善机器人学习”。

## 9. 复现与伦理

- 原始 OSS/CPFS 输入只读，所有 corruption、mask、repair 和训练视图写入独立输出；
- 固定代码 commit、容器 digest、模型权重版本、prompt、API 日期和数据 fingerprint；
- 保存全部 seed 和失败运行，不只保留最优模型；
- 对公开数据记录许可；对供应商和内部数据记录授权、保留策略和不可公开字段；
- ego 视频涉及真人、住宅、屏幕或语音时，论文需说明知情同意、隐私处理和适用的伦理审批；
- 公开可复现子集不得包含可逆身份信息或供应商敏感路径。

## 10. 当前决策

在完成 Phase A 小规模证据实验前：

- 不训练大规模最终学生；
- 不宣称 conditional evidence fusion 是有效方法；
- 不扩展更多 UI 或“使用场景分类”；
- 不用教师 API 标签评估教师自身；
- 不根据已经清洗过的数据结果宣称通用 ego 适配；
- 不为追求功能数量扩大论文主张。

下一步应先实现最小 corruption intervention manifest 和 evidence-delta 数据契约，再用真实 Gold
小样本验证核心假设。这是成本最低、最能避免错误路线的实验。
