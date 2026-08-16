from __future__ import annotations

import json
import hashlib
import time
from collections import Counter
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .artifacts import ArtifactCache
from .cache import Cache
from .provenance import code_version, config_hash
from .report import write_json, write_jsonl, write_parquet, write_report
from .types import EpisodeResult, Issue
from .validator import classify, load_episode_index, load_task_map, validate_dataset_structure, validate_episode
from .video import probe_video
from .decisions import acceptance_for, write_decision_manifests


def _value(row: Dict[str, Any], key: str, default: Any = None) -> Any:
    value = row.get(key, default)
    return value.as_py() if isinstance(value, pa.Scalar) else value


def _fps(info: Dict[str, Any], video_key: str) -> float:
    if "fps" in info:
        return float(info["fps"])
    return float(info["features"][video_key]["info"]["video.fps"])


def _episode_slices(table: pa.Table) -> Dict[int, pa.Table]:
    if "episode_index" not in table.column_names or not len(table):
        return {}
    values = np.asarray(
        table["episode_index"].combine_chunks().to_numpy(zero_copy_only=False),
        dtype=np.int64,
    )
    boundaries = np.flatnonzero(values[1:] != values[:-1]) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [len(values)]))
    segments: Dict[int, List[pa.Table]] = defaultdict(list)
    for start, end in zip(starts, ends):
        segments[int(values[start])].append(table.slice(int(start), int(end - start)))
    return {
        episode_index: parts[0] if len(parts) == 1 else pa.concat_tables(parts)
        for episode_index, parts in segments.items()
    }


def _cache_key(payload: Dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _failure_result(
    episode_index: int,
    length: int,
    code: str,
    message: str,
    file: str,
) -> EpisodeResult:
    result = EpisodeResult(episode_index, length, tier="quarantine")
    result.issues.append(Issue(code, "error", message, episode_index, file))
    return result


def _video_standard_issues(
    path: Path,
    metadata: Dict[str, Any],
    info: Dict[str, Any],
    expected_fps: float,
    video_key: str,
) -> List[Issue]:
    issues: List[Issue] = []
    feature = info.get("features", {}).get(video_key, {})
    expected = feature.get("info", {})
    checks = (
        ("width", "video.width", "video_width_mismatch"),
        ("height", "video.height", "video_height_mismatch"),
        ("pix_fmt", "video.pix_fmt", "video_pix_fmt_mismatch"),
    )
    for actual_key, expected_key, code in checks:
        expected_value = expected.get(expected_key)
        actual_value = metadata.get(actual_key)
        if expected_value is not None and actual_value is not None and actual_value != expected_value:
            issues.append(
                Issue(
                    code,
                    "error",
                    f"{actual_key}={actual_value}，info.json={expected_value}",
                    file=str(path),
                )
            )
    actual_fps = metadata.get("average_rate")
    if actual_fps is not None and abs(float(actual_fps) - expected_fps) > max(0.01, expected_fps * 0.001):
        issues.append(
            Issue(
                "video_fps_mismatch",
                "error",
                f"视频 fps={actual_fps:.6g}，期望 {expected_fps:.6g}",
                file=str(path),
            )
        )
    expected_codec = expected.get("video.codec")
    actual_codec = metadata.get("codec")
    codec_aliases = {"libx264": "h264", "avc1": "h264"}
    normalized_actual = codec_aliases.get(actual_codec, actual_codec)
    normalized_expected = codec_aliases.get(expected_codec, expected_codec)
    if normalized_expected and normalized_actual and normalized_actual != normalized_expected:
        issues.append(
            Issue(
                "video_codec_mismatch",
                "error",
                f"视频 codec={actual_codec}，info.json={expected_codec}",
                file=str(path),
            )
        )
    if metadata.get("audio_streams", 0):
        issues.append(
            Issue(
                "unexpected_audio_stream",
                "warning",
                f"视频包含 {metadata['audio_streams']} 条音轨",
                file=str(path),
            )
        )
    return issues


def run(
    dataset: Path,
    output: Path,
    config: Dict[str, Any],
    hash_mode: str = "headtail",
    cache_root: Optional[Path] = None,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    started_at = datetime.now(timezone.utc).isoformat()
    started_clock = time.perf_counter()
    dataset = dataset.resolve()
    output.mkdir(parents=True, exist_ok=True)
    configuration_hash = config_hash(config)
    live_run_id = hashlib.sha256(
        f"{started_at}\0{dataset}\0{configuration_hash}".encode("utf-8")
    ).hexdigest()[:16]
    live_root = output / "live" / "runs" / live_run_id
    write_json(
        output / "live" / "current.json",
        {
            "run_id": live_run_id,
            "dataset": str(dataset),
            "started_at": started_at,
            "status": "running",
            "completed": 0,
            "total": 0,
            "fraction": 0.0,
            "eta_s": None,
            "path": None,
        },
    )
    validation_hash = config_hash(
        {key: value for key, value in config.items() if key != "report"}
    )
    package_version = code_version()
    artifacts = ArtifactCache(cache_root or output / "artifact-cache")
    info, dataset_issues = validate_dataset_structure(dataset, config)
    structural_dataset_issues = list(dataset_issues)
    episode_table = load_episode_index(dataset)
    task_map = load_task_map(dataset)
    rows = episode_table.to_pylist()
    fps = _fps(info, config["video_key"])
    cache = Cache(output / "cache.sqlite")
    results: List[EpisodeResult] = []
    video_records: List[Dict[str, Any]] = []
    per_video_issues: Dict[Path, List[Issue]] = defaultdict(list)
    video_metadata_by_path: Dict[Path, Dict[str, Any]] = {}
    cache_stats = {
        "parquet_hits": 0,
        "parquet_misses": 0,
        "video_hits": 0,
        "video_misses": 0,
    }
    shard_records: List[Dict[str, Any]] = []
    input_bytes = 0

    data_groups: Dict[Path, List[Dict[str, Any]]] = defaultdict(list)
    route_failures: Dict[int, EpisodeResult] = {}
    for row in rows:
        ep = int(_value(row, "episode_index"))
        length = int(_value(row, "length"))
        data_chunk = _value(row, "data/chunk_index")
        data_file = _value(row, "data/file_index")
        if data_chunk is None or data_file is None:
            route_failures[ep] = _failure_result(
                ep,
                length,
                "missing_data_route",
                "episode metadata 缺少 data/chunk_index 或 data/file_index",
                "meta/episodes",
            )
            continue
        data_path = (
            dataset
            / "data"
            / f"chunk-{int(data_chunk):03d}"
            / f"file-{int(data_file):03d}.parquet"
        )
        data_groups[data_path].append(row)

    video_key = config["video_key"]
    planned_video_paths = set()
    for row in rows:
        video_chunk = _value(row, f"videos/{video_key}/chunk_index")
        video_file = _value(row, f"videos/{video_key}/file_index")
        if video_chunk is not None and video_file is not None:
            planned_video_paths.add(
                dataset
                / "videos"
                / video_key
                / f"chunk-{int(video_chunk):03d}"
                / f"file-{int(video_file):03d}.mp4"
            )
    total_work_units = len(data_groups) + len(planned_video_paths)
    completed_work_units = 0

    def progress(kind: str, path: Path) -> None:
        nonlocal completed_work_units
        completed_work_units += 1
        elapsed = max(0.0, time.perf_counter() - started_clock)
        rate = completed_work_units / elapsed if elapsed > 0 else 0.0
        remaining = max(0, total_work_units - completed_work_units)
        event = {
            "completed": completed_work_units,
            "total": total_work_units,
            "fraction": completed_work_units / max(1, total_work_units),
            "elapsed_s": elapsed,
            "eta_s": remaining / rate if rate > 0 else None,
            "logical_input_bytes": input_bytes,
            "kind": kind,
            "path": str(path),
        }
        write_json(
            output / "live" / "current.json",
            {
                "run_id": live_run_id,
                "dataset": str(dataset),
                "started_at": started_at,
                "status": "running",
                **event,
            },
        )
        if progress_callback:
            progress_callback(event)

    def live_shard(path: Path, shard_results: List[EpisodeResult]) -> None:
        name = hashlib.sha256(str(path).encode("utf-8")).hexdigest()[:20] + ".jsonl"
        write_jsonl(
            live_root / "shards" / name,
            [
                {
                    **result.to_dict(),
                    "acceptance": acceptance_for(
                        list(result.issues) + structural_dataset_issues, config
                    ),
                    "provisional": True,
                }
                for result in shard_results
            ],
        )

    shard_cache_dir = output / "shard_cache"
    shard_cache_dir.mkdir(exist_ok=True)
    for data_path, shard_rows in sorted(data_groups.items(), key=lambda item: str(item[0])):
        shard_started = time.perf_counter()
        if not data_path.exists():
            missing_results: List[EpisodeResult] = []
            for row in shard_rows:
                ep = int(_value(row, "episode_index"))
                length = int(_value(row, "length"))
                missing_results.append(
                    _failure_result(
                        ep,
                        length,
                        "missing_data_file",
                        "逐帧 parquet 不存在",
                        str(data_path),
                    )
                )
            results.extend(missing_results)
            live_shard(data_path, missing_results)
            progress("parquet", data_path)
            shard_records.append(
                {
                    "kind": "parquet",
                    "path": str(data_path),
                    "bytes": 0,
                    "episode_count": len(shard_rows),
                    "cache_status": "missing",
                    "elapsed_s": time.perf_counter() - shard_started,
                }
            )
            continue
        shard_bytes = data_path.stat().st_size
        input_bytes += shard_bytes
        fingerprint = cache.fingerprint(data_path, hash_mode)
        signature_payload = {
            "fingerprint": fingerprint,
            "validation_config_hash": validation_hash,
            "code_version": package_version,
            "fps": fps,
            "episodes": [
                [int(_value(row, "episode_index")), int(_value(row, "length"))]
                for row in shard_rows
            ],
        }
        signature = _cache_key(signature_payload)
        local_cached_path = shard_cache_dir / f"{signature}.json"
        cached = artifacts.read("parquet", signature)
        if cached is None and local_cached_path.exists():
            try:
                cached = json.loads(local_cached_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                cached = None
        if cached is not None:
            cache_stats["parquet_hits"] += 1
            cache_status = "hit"
            shard_results = [EpisodeResult.from_dict(value) for value in cached]
        else:
            cache_stats["parquet_misses"] += 1
            cache_status = "miss"
            shard_results: List[EpisodeResult] = []
            relative_path = str(data_path.relative_to(dataset))
            try:
                parquet = pq.ParquetFile(data_path)
                available = set(parquet.schema_arrow.names)
                requested = list(
                    dict.fromkeys(config["required_frame_columns"] + ["main_type"])
                )
                columns = [name for name in requested if name in available]
                table = parquet.read(columns=columns)
                episode_tables = _episode_slices(table)
                for row in shard_rows:
                    ep = int(_value(row, "episode_index"))
                    length = int(_value(row, "length"))
                    try:
                        ep_table = episode_tables.get(ep, table.slice(0, 0))
                        shard_results.append(
                            validate_episode(
                                ep_table,
                                ep,
                                length,
                                fps,
                                config,
                                relative_path,
                                filtered=True,
                                expected_from_index=_value(row, "dataset_from_index"),
                                expected_to_index=_value(row, "dataset_to_index"),
                                task_map=task_map,
                                expected_tasks=_value(row, "tasks", []),
                            )
                        )
                    except Exception as error:
                        shard_results.append(
                            _failure_result(
                                ep,
                                length,
                                "episode_validation_failed",
                                f"{type(error).__name__}: {error}",
                                relative_path,
                            )
                        )
            except Exception as error:
                shard_results = [
                    _failure_result(
                        int(_value(row, "episode_index")),
                        int(_value(row, "length")),
                        "data_read_failed",
                        f"{type(error).__name__}: {error}",
                        relative_path,
                    )
                    for row in shard_rows
                ]
            cached_values = [result.to_dict() for result in shard_results]
            shared_path = artifacts.write("parquet", signature, cached_values)
            ArtifactCache.materialize(shared_path, local_cached_path)
        data_limit = float(config.get("storage_limits", {}).get("data_file_size_mb", 0))
        if data_limit > 0 and shard_bytes > data_limit * 1024**2:
            for shard_result in shard_results:
                shard_result.issues.append(
                    Issue(
                        "parquet_file_too_large",
                        "error",
                        f"Parquet {shard_bytes / 1024**2:.2f} MiB，超过 {data_limit:g} MiB",
                        shard_result.episode_index,
                        str(data_path),
                    )
                )
                shard_result.tier = classify(shard_result.issues)
        results.extend(shard_results)
        live_shard(data_path, shard_results)
        cache.record(data_path, config["standard_version"], fingerprint)
        if not local_cached_path.exists():
            shared_path = artifacts.path("parquet", signature)
            if shared_path.exists():
                ArtifactCache.materialize(shared_path, local_cached_path)
        progress("parquet", data_path)
        shard_records.append(
            {
                "kind": "parquet",
                "path": str(data_path.relative_to(dataset)),
                "bytes": shard_bytes,
                "episode_count": len(shard_rows),
                "cache_status": cache_status,
                "elapsed_s": time.perf_counter() - shard_started,
                "fingerprint": fingerprint,
            }
        )

    results.extend(route_failures.values())

    seen_videos = set()
    by_episode = {result.episode_index: result for result in results}
    intervals: Dict[Path, List[tuple[float, float, int]]] = defaultdict(list)
    for row in rows:
        ep_index = int(_value(row, "episode_index"))
        result = by_episode[ep_index]
        chunk_size = int(config.get("storage_limits", {}).get("chunk_size", 0))
        if chunk_size > 0:
            expected_chunk = ep_index // chunk_size
            actual_data_chunk = _value(row, "data/chunk_index")
            actual_video_chunk = _value(row, f"videos/{video_key}/chunk_index")
            if (
                actual_data_chunk is not None
                and int(actual_data_chunk) != expected_chunk
            ) or (
                actual_video_chunk is not None
                and int(actual_video_chunk) != expected_chunk
            ):
                issue = Issue(
                    "chunk_index_mismatch",
                    "error",
                    f"episode {ep_index} 期望 chunk={expected_chunk}，data={actual_data_chunk} video={actual_video_chunk}",
                    ep_index,
                    "meta/episodes",
                )
                result.issues.append(issue)
                result.tier = classify(result.issues)
        chunk_key = f"videos/{video_key}/chunk_index"
        file_key = f"videos/{video_key}/file_index"
        video_chunk = _value(row, chunk_key)
        video_file = _value(row, file_key)
        if video_chunk is None or video_file is None:
            issue = Issue("missing_video_route", "error", f"episode metadata 缺少 {video_key} 路由", ep_index)
            dataset_issues.append(issue)
            result.issues.append(issue)
            result.tier = classify(result.issues)
            continue
        video_path = (
            dataset
            / "videos"
            / video_key
            / f"chunk-{int(video_chunk):03d}"
            / f"file-{int(video_file):03d}.mp4"
        )
        from_key = f"videos/{video_key}/from_timestamp"
        to_key = f"videos/{video_key}/to_timestamp"
        start_value = _value(row, from_key, 0.0)
        end_value = _value(row, to_key, 0.0)
        start = float(start_value) if start_value is not None else 0.0
        end = float(end_value) if end_value is not None else 0.0
        intervals[video_path].append((start, end, ep_index))
        expected_duration = result.length / fps
        if end <= start or abs((end - start) - expected_duration) > max(1.0 / fps, 0.002):
            issue = Issue(
                "video_interval_mismatch",
                "error",
                f"视频区间 [{start:.6f}, {end:.6f}) 与 length/fps={expected_duration:.6f}s 不一致",
                ep_index,
                str(video_path),
            )
            result.issues.append(issue)
            result.tier = classify(result.issues)
        if video_path in seen_videos:
            continue
        seen_videos.add(video_path)
        if not video_path.exists():
            dataset_issues.append(Issue("missing_video_file", "error", "视频文件不存在", file=str(video_path)))
            progress("video", video_path)
            continue
        video_started = time.perf_counter()
        video_bytes = video_path.stat().st_size
        input_bytes += video_bytes
        fingerprint = cache.fingerprint(video_path, hash_mode)
        video_signature = _cache_key(
            {
                "fingerprint": fingerprint,
                "code_version": package_version,
                "video_key": video_key,
                "video_check": config.get("video_check", {"mode": "header"}),
            }
        )
        cached_video = artifacts.read("video", video_signature)
        if cached_video is not None:
            cache_stats["video_hits"] += 1
            video_cache_status = "hit"
            metadata = cached_video["metadata"]
            issues = [Issue.from_dict(value) for value in cached_video["issues"]]
        else:
            cache_stats["video_misses"] += 1
            video_cache_status = "miss"
            video_check = config.get("video_check", {})
            metadata, issues = probe_video(
                video_path,
                str(video_check.get("mode", "header")),
                video_check,
            )
            artifacts.write(
                "video",
                video_signature,
                {
                    "metadata": metadata,
                    "issues": [issue.to_dict() for issue in issues],
                },
            )
        standard_issues = _video_standard_issues(
            video_path,
            metadata,
            info,
            fps,
            video_key,
        )
        issues = list(issues) + standard_issues
        video_limit = float(config.get("storage_limits", {}).get("video_file_size_mb", 0))
        if video_limit > 0 and video_bytes > video_limit * 1024**2:
            issues.append(
                Issue(
                    "video_file_too_large",
                    "error",
                    f"视频 {video_bytes / 1024**2:.2f} MiB，超过 {video_limit:g} MiB",
                    file=str(video_path),
                )
            )
        per_video_issues[video_path].extend(issues)
        video_metadata_by_path[video_path] = metadata
        dataset_issues.extend(issues)
        video_records.append({"path": str(video_path.relative_to(dataset)), **metadata})
        cache.record(video_path, config["standard_version"], fingerprint)
        progress("video", video_path)
        shard_records.append(
            {
                "kind": "video",
                "path": str(video_path.relative_to(dataset)),
                "bytes": video_bytes,
                "episode_count": len(intervals[video_path]),
                "cache_status": video_cache_status,
                "elapsed_s": time.perf_counter() - video_started,
                "fingerprint": fingerprint,
            }
        )

    for video_path, spans in intervals.items():
        spans.sort()
        metadata = video_metadata_by_path.get(video_path, {})
        actual_duration = metadata.get("duration")
        counted_frames = int(metadata.get("counted_frames") or 0)
        reported_frames = counted_frames or int(metadata.get("reported_frames") or 0)
        expected_end = max((end for _, end, _ in spans), default=0.0)
        if (
            actual_duration is not None
            and abs(float(actual_duration) - expected_end) > 2.0 / fps
        ):
            issue = Issue(
                "video_duration_mismatch",
                "error",
                f"视频 duration={actual_duration:.6f}s，episode 最大终点={expected_end:.6f}s",
                file=str(video_path),
            )
            dataset_issues.append(issue)
            per_video_issues[video_path].append(issue)
        expected_frames = int(round(expected_end * fps))
        if reported_frames > 0 and abs(reported_frames - expected_frames) > 2:
            issue = Issue(
                "video_frame_count_mismatch",
                "error",
                f"视频 {'counted_frames' if counted_frames else 'reported_frames'}={reported_frames}，episode 区间期望 {expected_frames}（容差 ±2）",
                file=str(video_path),
            )
            dataset_issues.append(issue)
            per_video_issues[video_path].append(issue)
        if len(spans) > 1 and all(abs(start) < 1e-9 for start, _, _ in spans):
            for _, _, ep_index in spans:
                issue = Issue(
                    "aggregated_video_zero_offsets",
                    "error",
                    "同一聚合 MP4 含多个 episode，但 from_timestamp 全为 0",
                    ep_index,
                    str(video_path),
                )
                by_episode[ep_index].issues.append(issue)
                by_episode[ep_index].tier = classify(by_episode[ep_index].issues)
        for previous, current in zip(spans, spans[1:]):
            if current[0] < previous[1] - 1e-6:
                for ep_index in (previous[2], current[2]):
                    issue = Issue(
                        "video_intervals_overlap",
                        "error",
                        f"聚合 MP4 episode 时间区间重叠: {previous[:2]} vs {current[:2]}",
                        ep_index,
                        str(video_path),
                    )
                    by_episode[ep_index].issues.append(issue)
                    by_episode[ep_index].tier = classify(by_episode[ep_index].issues)

        for _, _, ep_index in spans:
            for video_issue in per_video_issues.get(video_path, []):
                issue = Issue(
                    video_issue.code,
                    video_issue.severity,
                    video_issue.message,
                    ep_index,
                    video_issue.file,
                    video_issue.evidence,
                )
                by_episode[ep_index].issues.append(issue)
            by_episode[ep_index].tier = classify(by_episode[ep_index].issues)

    for row in rows:
        ep_index = int(_value(row, "episode_index"))
        chunk_key = f"videos/{video_key}/chunk_index"
        file_key = f"videos/{video_key}/file_index"
        video_chunk = _value(row, chunk_key)
        video_file = _value(row, file_key)
        if video_chunk is None or video_file is None:
            continue
        video_path = (
            dataset
            / "videos"
            / video_key
            / f"chunk-{int(video_chunk):03d}"
            / f"file-{int(video_file):03d}.mp4"
        )
        if not video_path.exists():
            issue = Issue("missing_video_file", "error", "视频文件不存在", ep_index, str(video_path))
            by_episode[ep_index].issues.append(issue)
            by_episode[ep_index].tier = classify(by_episode[ep_index].issues)
    cache.close()

    tier_counts = Counter(result.tier for result in results)
    elapsed_s = time.perf_counter() - started_clock
    parquet_total = cache_stats["parquet_hits"] + cache_stats["parquet_misses"]
    video_total = cache_stats["video_hits"] + cache_stats["video_misses"]
    decision_summary = write_decision_manifests(
        output, dataset, results, config, structural_dataset_issues
    )
    total_episode_duration_s = sum(result.length / fps for result in results)
    qualified_visible_duration_s = sum(
        float(result.metrics.get("qualified_visible_duration_s", 0.0)) for result in results
    )
    effective_video_duration_s = sum(
        float(result.metrics.get("effective_video_duration_s", 0.0)) for result in results
    )
    visibility_summary = {
        "total_episode_duration_s": total_episode_duration_s,
        "qualified_visible_duration_s": qualified_visible_duration_s,
        "effective_video_duration_s": effective_video_duration_s,
        "effective_video_hours": effective_video_duration_s / 3600.0,
        "effective_utilization_ratio": (
            effective_video_duration_s / total_episode_duration_s
            if total_episode_duration_s > 0
            else 0.0
        ),
        "episodes_hand_out_of_view_too_long": sum(
            any(issue.code == "hand_out_of_view_too_long" for issue in result.issues)
            for result in results
        ),
        "episodes_without_qualifying_visibility": sum(
            any(
                issue.code == "insufficient_continuous_hand_visibility"
                for issue in result.issues
            )
            for result in results
        ),
    }
    summary = {
        "dataset": str(dataset),
        "standard_version": config["standard_version"],
        "config_hash": configuration_hash,
        "validation_config_hash": validation_hash,
        "code_version": package_version,
        "fps": fps,
        "episode_count": len(results),
        "tier_counts": dict(tier_counts),
        "dataset_issue_count": len(dataset_issues),
        "cache": cache_stats,
        "decisions": decision_summary,
        "visibility": visibility_summary,
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_s": elapsed_s,
        "logical_input_bytes": input_bytes,
        "logical_throughput_mib_s": (
            input_bytes / (1024 * 1024) / elapsed_s if elapsed_s > 0 else 0.0
        ),
        "parquet_cache_hit_ratio": (
            cache_stats["parquet_hits"] / parquet_total if parquet_total else 0.0
        ),
        "video_cache_hit_ratio": (
            cache_stats["video_hits"] / video_total if video_total else 0.0
        ),
    }
    write_json(output / "summary.json", summary)
    write_jsonl(
        live_root / "episodes-final.jsonl",
        [
            {
                **result.to_dict(),
                "acceptance": acceptance_for(
                    list(result.issues) + structural_dataset_issues, config
                ),
                "provisional": False,
            }
            for result in results
        ],
    )
    write_json(
        output / "live" / "current.json",
        {
            "run_id": live_run_id,
            "dataset": str(dataset),
            "started_at": started_at,
            "finished_at": summary["finished_at"],
            "status": "succeeded",
            "completed": total_work_units,
            "total": total_work_units,
            "fraction": 1.0,
            "eta_s": 0.0,
            "path": None,
        },
    )
    episode_rows = [result.to_dict() for result in results]
    issue_rows = [
        issue.to_dict() for issue in dataset_issues
    ] + [
        issue.to_dict() for result in results for issue in result.issues
    ]
    write_jsonl(output / "episodes.jsonl", episode_rows)
    write_jsonl(output / "issues.jsonl", issue_rows)
    write_jsonl(output / "videos.jsonl", video_records)
    write_jsonl(output / "shards.jsonl", shard_records)
    write_parquet(
        output / "episodes.parquet",
        [
            {
                "episode_index": result.episode_index,
                "length": result.length,
                "tier": result.tier,
                **acceptance_for(list(result.issues) + structural_dataset_issues, config),
                "metrics_json": json.dumps(
                    result.metrics,
                    ensure_ascii=False,
                    allow_nan=True,
                ),
                "issues_json": json.dumps(
                    [issue.to_dict() for issue in result.issues],
                    ensure_ascii=False,
                    allow_nan=True,
                ),
                "sample_frames": result.sample_frames,
            }
            for result in results
        ],
    )
    write_parquet(
        output / "issues.parquet",
        [
            {
                "code": issue["code"],
                "severity": issue["severity"],
                "message": issue["message"],
                "episode_index": issue["episode_index"],
                "file": issue["file"],
                "evidence_json": json.dumps(
                    issue["evidence"],
                    ensure_ascii=False,
                    allow_nan=True,
                ),
            }
            for issue in issue_rows
        ],
    )
    write_parquet(output / "videos.parquet", video_records)
    write_parquet(output / "shards.parquet", shard_records)
    write_jsonl(output / "sample_plan.jsonl", [
        {"episode_index": result.episode_index, "frame_indices": result.sample_frames}
        for result in results
    ])
    write_report(
        output / "report.html",
        dataset,
        summary,
        results,
        dataset_issues,
        int(config.get("report", {}).get("max_episodes", 500)),
        shard_records,
    )
    return summary
