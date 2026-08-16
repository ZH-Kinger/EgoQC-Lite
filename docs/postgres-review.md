# PostgreSQL 动态人工复检

PostgreSQL 只保存异常事件、领取租约、人工结论和审计历史。原始 OSS/CPFS 数据保持
只读，现有 SQLite Registry 继续负责扫描任务、worker lease 和处理进度。两者不要
强行合库：跑批状态是单机/文件系统强相关状态，人工复检是跨机器共享状态。

## 数据模型

- `review_datasets`：逻辑数据集，不绑定某次挂载路径。
- `review_runs`：检测模型和阈值的一次版本化运行。
- `review_events`：当前队列状态和最新人工结论。
- `review_decisions`：append-only 的结论历史，用于审计和训练集回流。

`review_events.kind` 是稳定错误代码，`category` 是一级类别，`severity` 是验收
严重度。统一映射位于 `src/egoqc/review_taxonomy.py`。当前 RekaDaily 手部预筛实际
产出 `hand_absent` 和 `persistent_extra_hands` 两类；抖动、姿态冻结、MANO/SO(3)、
MPJPE、ATE、时间漂移、视频质量和 schema 错误已经预留 taxonomy，但只有相应检测器
真正产出事件后才会出现在复检队列，不用空壳指标冒充测量结果。

事件导入使用 `event_id` 幂等 upsert；重新计算模型指标不会抹掉已完成的人工结论。
审核员领取事件时获得不可猜测的 token，默认租约为 15 分钟。写入结论同时检查
token、审核员和事件版本，避免多人覆盖。过期租约在列表或统计请求时自动回收。

## 启动

安装可选依赖：

```bash
python3 -m venv .venv-review
.venv-review/bin/python -m pip install -e '.[postgres]'
```

数据库连接串只通过环境变量注入，不写进仓库：

```bash
export DATABASE_URL='postgresql://egoqc:REDACTED@127.0.0.1:5432/egoqc'
egoqc init-review-db
egoqc import-review-events /path/to/events.json \
  --dataset-name rekadaily-10k-raw \
  --run-name hand-screen-v2 \
  --source-root /srv/egoqc/samples/rekadaily
egoqc serve-postgres-review \
  --evidence-root /srv/egoqc/results/rekadaily-hand-anomaly-review-v1 \
  --host 127.0.0.1 --port 8767
```

本机访问开发机时使用 SSH 隧道，不开放数据库或 Web 端口到公网：

```bash
ssh -N -L 8767:127.0.0.1:8767 <user>@<dev-host> -p <ssh-port>
```

打开 `http://127.0.0.1:8767/`。页面每 3 秒更新队列和统计；结论实时写入
PostgreSQL。浏览器 localStorage 仅记住审核员名字，不保存审核结果。

生产开发机可由 Supervisor 托管数据库和 Web 服务，对应模板位于
`deploy/egoqc-postgres.supervisor.conf` 和 `deploy/egoqc-review.supervisor.conf`。
数据库与 Web 分别只监听 `127.0.0.1:5432` 和 `127.0.0.1:8767`。

## 百 TB 规模边界

数据库不存视频 blob、逐帧检测或 MANO 数组，只保存事件索引和小型 JSONB 指标。
MP4 证据片段继续放 CPFS/对象存储派生区。生产环境建议每日 `pg_dump`，每月清理
无效 run，但保留 `review_decisions` 审计表。若 Web 服务扩成多副本，可继续共用同一
PostgreSQL；领取接口已经具备并发冲突保护。
