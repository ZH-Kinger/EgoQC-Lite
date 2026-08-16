# EgoQC-Lite architecture

## Scope

The reference implementation targets LeRobot v3 datasets containing:

- aggregated frame Parquet shards;
- aggregated MP4 shards;
- per-episode routing metadata;
- ego RGB video;
- world-space wrist transforms;
- MANO joint rotations;
- camera-space compressed state.

It deliberately separates three questions:

1. Is the dataset structurally readable?
2. Are redundant geometric representations internally consistent?
3. Is the reconstructed hand visually aligned with the source video?

The first two are deterministic and run over all incoming data. The third is
sampled and must never block ingestion of otherwise useful video.

## Incremental model

Raw files are immutable. Every output is derived and versioned.

```text
raw LeRobot v3
  -> file fingerprint
  -> frame/episode deterministic checks
  -> shard result cache
  -> adaptive visual sample plan
  -> sampled evidence
  -> materialized training views
```

The shard cache signature includes:

- head/tail file fingerprint;
- the complete quality configuration hash;
- the EgoQC code version;
- episode IDs and expected lengths.

Changing the data, metadata, thresholds or standard version therefore creates a
new cache entry without rewriting the raw dataset.

## Mounted OSS / CPFS control plane

The first production control plane treats mounted OSS or CPFS paths as regular,
read-only POSIX paths. It does not require cloud credentials in the scanner.

```text
explicit dataset roots
  -> register: route-aware stat inventory
  -> registry.sqlite: dataset/file/run state
  -> plan: only unseen signature + standard pairs
  -> immutable JSONL manifest
  -> run-manifest: verify signature, scan, checkpoint
  -> versioned quality result
```

Registration follows files referenced by `meta/episodes`; it does not recursively
walk a multi-hundred-TB mount. Dataset identity is derived from a stable source
label and the path relative to a configured mount root. Dataset signatures use
relative path, type, size, mtime and existence. The scan layer can independently
use head/tail fingerprints when stronger verification is needed.

SQLite control state must live on local storage or CPFS with normal filesystem
locking. It must not live directly on an object-storage FUSE mount. Registry
snapshots and immutable JSONL manifests may be copied to OSS for durability.

Before a manifest task runs, the mounted files are inventoried again. A signature
change marks the task `stale`; it is never silently processed under the old
identity. Successful `(dataset, signature, standard_version, config_hash,
code_version)` tuples are skipped on later plans, while failed tasks remain
retryable. Workers atomically claim tasks with expiring leases and refresh a
heartbeat after each shard.

Parquet results and video probe metadata live in a shared, content-addressed
artifact cache, so unchanged shards survive dataset-level metadata revisions.
Within a Parquet shard, only validation columns are projected. `episode_index`
is converted once and contiguous row slices are built in one pass.

Local parallelism is across finalized dataset/batch tasks. Each task retains
single-open semantics for its aggregated MP4 shards. Operational commands expose
environment checks (`doctor`), registry state (`status`), and a raw-data-free
end-to-end synthetic run (`self-test`). Machine-readable outputs are emitted as
both JSONL and Zstandard-compressed Parquet, with per-shard latency, logical
bytes, fingerprints, and cache status.

## Scale-out boundary

The unit of work is a Parquet or MP4 shard, not an episode.

- A Parquet shard is read once and all contained episodes are evaluated.
- Sample targets from all episodes in an MP4 are merged.
- The MP4 is opened once and decoded sequentially until all targets are found.

This boundary can later be scheduled by Ray, Daft, Kubernetes Jobs or an
existing batch system without changing validation functions.

## Quality tiers

- `gold`: no warning or error.
- `silver`: warnings only; usable with an explicit sampling weight.
- `bronze`: non-fatal geometric/semantic error; limited-use views only.
- `quarantine`: structural corruption, non-finite data or invalid rotations.

The tier is a routing convenience. Individual issue codes and metrics remain
the source of truth.

## Visual layer contract

`sample_plan.jsonl` contains local episode frame indices. `extract-samples`
resolves them through episode video offsets and writes evidence images and
contact sheets. A later MANO renderer or overlay classifier can consume this
stable interface without coupling itself to the full scanner.
