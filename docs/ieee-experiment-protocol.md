# EgoQC-Lite 论文级实验协议

本文定义 EgoQC-Lite 的可复现实验口径。目标是形成可审稿的实验链路，并不意味着仅凭运行
脚本即可宣称达到某个 IEEE 标准或 99% 生产准确率。任何结论必须由锁定的人工 Gold test
和统计置信区间支持。

## 1. 研究问题

1. 第一人称领域自监督继续预训练是否提高固定 99% precision 下的 recall？
2. 全局 RGB、手部 ROI 和 MANO/SLAM 数值序列融合是否改善跨供应商 worst-group 表现？
3. 多专家教师的知识能否蒸馏到高吞吐学生，同时维持可靠自动化覆盖率？
4. 三路 selective decision 是否优于强制二分类？

预注册的机器可读实验矩阵位于 `config/qc_ieee_experiments.json`。更换主要指标、测试集或
阈值规则必须产生新的协议版本，不能覆盖原结果。

## 2. 数据隔离

- 训练集允许人工 Gold、教师 Silver、程序化 Weak 和受控合成数据。
- validation/test 只允许人工 Gold；合成样本和教师标签不得进入。
- validation 仅用于模型选择、温度校准和阈值冻结。
- test 只运行一次固定模型与固定阈值，不得根据 test 结果重新调参。
- 相同 supplier/person/operator/session、原视频及相邻派生 clip 不得跨 validation/test。
- 至少保留一个未见供应商或未见相机作为 external test。
- 原始 OSS 数据保持只读；所有派生物必须包含 source revision/ETag 和代码版本。

## 3. 主要与次要终点

主要终点是每类 `recall at precision >= 99%`。precision 是否达标以 95% Wilson 下置信界判断，
不是以经验值判断。validation 上选择满足约束且 recall 最大的阈值，然后把该阈值原样应用于
test。

同时报告：

- AP、AUROC、F1；
- Brier score、15-bin ECE；
- 自动拒收覆盖率；
- precision/recall 95% 区间；
- person/session 聚类 bootstrap 区间；
- supplier、camera、source dataset 的 worst-group precision/recall；
- 每 GPU 小时处理的视频小时数、峰值显存和端到端 I/O 吞吐。

不能用类别极不均衡条件下的 overall accuracy 作为主要结论。

## 4. 基线与消融

固定运行 E0–E8：规则基线、RGB 视频基线、多尺度时序、手部 ROI、运动学融合、领域继续预训练、
教师集成、学生蒸馏、学生加 selective decision。每项至少运行三个随机种子，并报告均值、标准差
和所有独立运行，不只报告最佳 seed。

训练阶段可以使用大模型和多卡全参数训练；部署模型仍应通过蒸馏控制百万小时推理吞吐。
最新但尚未稳定发布的骨干只能作为 challenger，必须保留 VideoMAE V2 等可完全复现基线。

## 5. 正式评测命令

```bash
egoqc evaluate-qc-research \
  --validation-predictions /path/to/validation-predictions.jsonl \
  --validation-gold /path/to/validation-gold.jsonl \
  --test-predictions /path/to/test-predictions.jsonl \
  --test-gold /path/to/test-gold.jsonl \
  --task-config config/visual_model_tasks.json \
  --bootstrap-replicates 1000 \
  --minimum-group-samples 30 \
  --output /path/to/locked-evaluation
```

输出包含：

- `qc-research-evaluation.json`：固定阈值、test 指标、区间、协议有效性和输入 SHA-256；
- `qc-research-per-group.jsonl`：按供应商、相机和来源切片的结果。

若 validation/test 存在 video 或 person/session 身份交叉，协议会标记为无效，所有自动拒收任务
都保持关闭。

## 6. 投稿前复现清单

- 固定代码 commit、容器 digest、CUDA/cuDNN、驱动、GPU 型号和依赖 lockfile；
- 保存训练 manifest、Gold label、模型 checkpoint 和输入 SHA-256；
- 保存所有 seed 的日志与失败运行，禁止只保留最优结果；
- 报告数据排除流程、各标签数量、标注者一致性和仲裁比例；
- 报告每个供应商与错误类别的 confusion matrix；
- 报告 API 教师名称、版本、prompt、日期和缓存；
- 公开数据实验给出可执行复现命令；供应商数据只公开不可逆统计与协议；
- 提供典型 true positive、false positive、false negative 和 OOD 案例可视化；
- 在论文中明确新预印本模型与已稳定公开基线的区别。
