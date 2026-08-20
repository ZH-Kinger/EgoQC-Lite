from __future__ import annotations

import hashlib
import json
import platform
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .provenance import code_version
from .report import write_json, write_jsonl


SCHEMA_VERSION = "egoqc-few-b-vlm-benchmark-v1"


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.expanduser().open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: each row must be an object")
            rows.append(row)
    return rows


def select_benchmark_rows(
    rows: Sequence[Dict[str, Any]],
    maximum_clips: int,
    seed: int,
    strategy: str = "stable_random",
) -> List[Dict[str, Any]]:
    if maximum_clips <= 0:
        raise ValueError("maximum_clips must be positive")

    def rank(row: Dict[str, Any]) -> str:
        identity = str(
            row.get("record_id")
            or row.get("video_id")
            or row.get("request_id")
            or row.get("source_uri")
            or ""
        )
        return hashlib.sha256(f"{seed}:{identity}".encode("utf-8")).hexdigest()

    if strategy not in {"stable_random", "balanced_weak"}:
        raise ValueError("strategy must be stable_random or balanced_weak")
    usable = [
        row
        for row in rows
        if row.get("source_uri") and Path(str(row["source_uri"])).expanduser().is_file()
    ]
    if strategy == "stable_random":
        return sorted(usable, key=rank)[:maximum_clips]
    positives = []
    negatives = []
    for row in usable:
        targets = (row.get("distillation") or {}).get("targets") or {}
        if any(float(value) >= 0.5 for value in targets.values()):
            positives.append(row)
        else:
            negatives.append(row)
    positive_limit = (maximum_clips + 1) // 2
    negative_limit = maximum_clips // 2
    selected = sorted(positives, key=rank)[:positive_limit]
    selected.extend(sorted(negatives, key=rank)[:negative_limit])
    if len(selected) < maximum_clips:
        selected_ids = {id(row) for row in selected}
        remainder = [row for row in usable if id(row) not in selected_ids]
        selected.extend(sorted(remainder, key=rank)[: maximum_clips - len(selected)])
    return selected


def _clip_window(row: Dict[str, Any]) -> Tuple[float, float]:
    distillation = row.get("distillation") or {}
    sampler = (row.get("vla_pretraining") or {}).get("clip_sampler") or {}
    start = float(
        distillation.get("clip_start_s")
        if distillation.get("clip_start_s") is not None
        else row.get("clip_start_s")
        if row.get("clip_start_s") is not None
        else sampler.get("fixed_start_s") or 0.0
    )
    if distillation.get("clip_end_s") is not None:
        end = float(distillation["clip_end_s"])
    elif row.get("clip_end_s") is not None:
        end = float(row["clip_end_s"])
    else:
        duration = float(sampler.get("window_s") or min(8.0, float(row.get("duration_s") or 8.0)))
        end = start + duration
    source_duration = float(row.get("duration_s") or end)
    start = max(0.0, min(start, source_duration))
    end = max(start, min(end, source_duration))
    if end - start <= 0:
        raise ValueError(f"invalid clip window {start:.3f}..{end:.3f}")
    return start, end


def _sample_video_frames(
    path: Path, start_s: float, end_s: float, frame_count: int
) -> List[Any]:
    try:
        import av
    except ImportError as error:
        raise RuntimeError("few-B benchmark requires PyAV") from error
    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    if frame_count == 1:
        targets = [(start_s + end_s) / 2.0]
    else:
        step = (end_s - start_s) / frame_count
        targets = [start_s + (index + 0.5) * step for index in range(frame_count)]
    images: List[Any] = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate) if stream.average_rate else 30.0
        tolerance = 0.5 / max(fps, 1.0)
        seek_s = max(0.0, start_s - 1.0)
        if stream.time_base is not None and start_s > 0:
            container.seek(
                int(seek_s / float(stream.time_base)),
                stream=stream,
                any_frame=False,
                backward=True,
            )
        target_index = 0
        decoded_index = 0
        last_image = None
        for frame in container.decode(stream):
            frame_time = (
                float(frame.time)
                if frame.time is not None
                else seek_s + decoded_index / max(fps, 1e-6)
            )
            decoded_index += 1
            if frame_time + tolerance < targets[target_index]:
                continue
            image = frame.to_image().convert("RGB")
            if last_image is not None:
                last_image.close()
            last_image = image.copy()
            while target_index < len(targets) and frame_time >= targets[target_index] - tolerance:
                images.append(image.copy())
                target_index += 1
            image.close()
            if target_index >= len(targets):
                break
        while len(images) < frame_count and last_image is not None:
            images.append(last_image.copy())
        if last_image is not None:
            last_image.close()
        if len(images) != frame_count:
            raise RuntimeError(
                f"failed to decode {frame_count} frames from {path} at {start_s:.3f}..{end_s:.3f}"
            )
    return images


def _decode_benchmark_row(
    row: Dict[str, Any], frame_count: int
) -> Tuple[Path, float, float, List[Any], float]:
    source = Path(str(row["source_uri"])).expanduser().resolve()
    start_s, end_s = _clip_window(row)
    started = time.perf_counter()
    frames = _sample_video_frames(source, start_s, end_s, frame_count)
    return source, start_s, end_s, frames, time.perf_counter() - started


def _prefetched_decodes(
    rows: Sequence[Dict[str, Any]], frame_count: int, workers: int
) -> Iterable[Tuple[Dict[str, Any], Path, float, float, List[Any], float]]:
    if workers <= 0:
        raise ValueError("decode workers must be positive")
    if workers == 1:
        for row in rows:
            yield row, *_decode_benchmark_row(row, frame_count)
        return
    iterator = iter(rows)
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="egoqc-decode") as executor:
        pending = deque()
        for _ in range(workers):
            try:
                row = next(iterator)
            except StopIteration:
                break
            pending.append((row, executor.submit(_decode_benchmark_row, row, frame_count)))
        while pending:
            row, future = pending.popleft()
            yield row, *future.result()
            try:
                next_row = next(iterator)
            except StopIteration:
                continue
            pending.append(
                (next_row, executor.submit(_decode_benchmark_row, next_row, frame_count))
            )


def freeze_few_b_samples(
    manifest: Path,
    output: Path,
    *,
    maximum_clips: int = 200,
    seed: int = 31,
    selection_strategy: str = "balanced_weak",
) -> Dict[str, Any]:
    source_path = manifest.expanduser().resolve()
    selected = select_benchmark_rows(
        _read_jsonl(source_path),
        maximum_clips,
        seed,
        strategy=selection_strategy,
    )
    if not selected:
        raise ValueError("manifest has no locally readable source_uri rows")
    positive_tasks: Dict[str, int] = {}
    positive_rows = 0
    source_counts: Dict[str, int] = {}
    for row in selected:
        source = str(row.get("source_dataset") or "unknown")
        source_counts[source] = source_counts.get(source, 0) + 1
        targets = (row.get("distillation") or {}).get("targets") or {}
        active = [str(task) for task, value in targets.items() if float(value) >= 0.5]
        positive_rows += int(bool(active))
        for task in active:
            positive_tasks[task] = positive_tasks.get(task, 0) + 1
    output = output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    frozen = output / "samples.jsonl"
    write_jsonl(frozen, selected)
    digest = hashlib.sha256(frozen.read_bytes()).hexdigest()
    summary = {
        "schema_version": "egoqc-few-b-frozen-samples-v1",
        "status": "frozen_not_evaluated",
        "input_manifest": str(source_path),
        "input_manifest_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "samples": len(selected),
        "maximum_clips": maximum_clips,
        "selection_seed": seed,
        "selection_strategy": selection_strategy,
        "positive_rows_by_weak_teacher": positive_rows,
        "negative_rows_by_weak_teacher": len(selected) - positive_rows,
        "positive_task_counts": positive_tasks,
        "source_counts": source_counts,
        "split_groups": len(
            {
                str(row.get("split_group") or (row.get("distillation") or {}).get("split_group") or row.get("video_id") or row.get("request_id"))
                for row in selected
            }
        ),
        "teacher_labels_are_human_gold": False,
        "formal_accuracy_claim_authorized": False,
        "samples_sha256": digest,
        "frozen_manifest": str(frozen),
    }
    write_json(output / "summary.json", summary)
    return summary


def parse_structured_response(text: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    value = text.strip()
    if value.startswith("```"):
        lines = value.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        value = "\n".join(lines).strip()
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as error:
        return None, str(error)
    if not isinstance(parsed, dict):
        return None, "response JSON is not an object"
    return parsed, None


def normalize_sparse_findings(
    parsed: Dict[str, Any], task_order: Sequence[str]
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    findings = parsed.get("f")
    if not isinstance(findings, list):
        return None, "f must be a list"
    allowed = set(task_order)
    normalized = dict(parsed)
    normalized_findings = []
    for finding in findings:
        if not isinstance(finding, list) or len(finding) != 6:
            return None, "every finding must have 6 fields"
        code_or_index = finding[0]
        code: Optional[str] = None
        if isinstance(code_or_index, int):
            if 0 <= code_or_index < len(task_order):
                code = task_order[code_or_index]
        elif isinstance(code_or_index, str):
            if code_or_index in allowed:
                code = code_or_index
            elif code_or_index.isdigit():
                index = int(code_or_index)
                if 0 <= index < len(task_order):
                    code = task_order[index]
        if code is None:
            return None, f"unknown finding code or index: {code_or_index}"
        normalized_findings.append([code, *finding[1:]])
    normalized["f"] = normalized_findings
    return normalized, None


def _percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _artifact_fingerprint(model_path: Path) -> Dict[str, Any]:
    files = sorted(path for path in model_path.rglob("*") if path.is_file())
    digest = hashlib.sha256()
    total = 0
    for path in files:
        relative = path.relative_to(model_path).as_posix()
        size = path.stat().st_size
        total += size
        digest.update(relative.encode("utf-8"))
        digest.update(str(size).encode("ascii"))
        with path.open("rb") as handle:
            while block := handle.read(8 * 1024 * 1024):
                digest.update(block)
    return {"sha256": digest.hexdigest(), "bytes": total, "files": len(files)}


def _load_resumable_results(
    output: Path,
    rows: Sequence[Dict[str, Any]],
    *,
    frame_count: int,
    maximum_edge: int,
) -> List[Dict[str, Any]]:
    partial = output / "predictions.partial.jsonl"
    if not partial.is_file():
        return []
    allowed = {
        str(row.get("video_id") or row.get("request_id") or row.get("record_id") or "")
        for row in rows
    }
    resumed: List[Dict[str, Any]] = []
    seen = set()
    for result in _read_jsonl(partial):
        identity = str(result.get("video_id") or result.get("record_id") or "")
        if (
            identity
            and identity in allowed
            and identity not in seen
            and int(result.get("frame_count") or 0) == frame_count
            and int(result.get("maximum_edge") or 0) == maximum_edge
        ):
            resumed.append(result)
            seen.add(identity)
    return resumed


def _prompt(task_config: Dict[str, Any], activity: str, frame_count: int) -> str:
    tasks = list(task_config.get("model_tasks", {}).items())
    task_lines = [f"{index}={code}({spec.get('label', code)})" for index, (code, spec) in enumerate(tasks)]
    return (
        "你是第一人称具身数据的视觉质检器。下面是一个按时间均匀抽取的短视频片段，"
        f"共有 {frame_count} 帧，候选任务为 {activity or 'unknown'}。\n"
        "问题索引固定为：" + ";".join(task_lines)
        + "\n只输出一行紧凑 JSON，不要 Markdown、解释或代码块。格式必须是："
        '{"f":[[问题索引整数,概率0到1,严重度整数0到3,开始时间0到1,结束时间0到1,'
        '[证据帧索引]]],"c":总体置信度0到1,"a":是否拒答}。'
        "f只写概率大于等于0.05的问题，完全没有问题时必须写空数组f=[]；"
        "严重度0/1/2/3对应none/minor/major/critical。看不清或证据不足时a=true，不得臆测。"
    )


def _prepare_model_inputs(
    processor: Any,
    frames: Sequence[Any],
    prompt: str,
    maximum_edge: int,
    sample_fps: float,
) -> Any:
    from qwen_vl_utils import process_vision_info

    video_item = {
        "type": "video",
        "video": list(frames),
        "sample_fps": sample_fps,
        "max_pixels": maximum_edge * maximum_edge,
        "total_pixels": len(frames) * maximum_edge * maximum_edge,
    }
    messages = [
        {"role": "user", "content": [video_item, {"type": "text", "text": prompt}]}
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    images, videos, video_kwargs = process_vision_info(
        messages,
        image_patch_size=processor.image_processor.patch_size,
        return_video_kwargs=True,
        return_video_metadata=True,
    )
    if videos is not None:
        videos, metadata = zip(*videos)
        videos = list(videos)
        metadata = list(metadata)
    else:
        metadata = None
    return processor(
        text=text,
        images=images,
        videos=videos,
        video_metadata=metadata,
        return_tensors="pt",
        do_resize=False,
        **video_kwargs,
    )


def benchmark_few_b_vlm(
    model_path: Path,
    manifest: Path,
    task_config_path: Path,
    output: Path,
    *,
    model_id: Optional[str] = None,
    maximum_clips: int = 3,
    frame_count: int = 8,
    maximum_edge: int = 448,
    device: str = "cuda",
    precision: str = "bf16",
    max_new_tokens: int = 256,
    seed: int = 17,
    selection_strategy: str = "stable_random",
    resume: bool = False,
    decode_workers: int = 2,
) -> Dict[str, Any]:
    """Run a small, fully traced inference benchmark. It never computes accuracy."""

    if precision not in {"bf16", "fp16", "fp32"}:
        raise ValueError("precision must be bf16, fp16 or fp32")
    if device not in {"cuda", "cpu"}:
        raise ValueError("device must be cuda or cpu")
    if decode_workers <= 0:
        raise ValueError("decode_workers must be positive")
    try:
        import torch
        import transformers
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    except ImportError as error:
        raise RuntimeError(
            "few-B benchmark requires torch, transformers and qwen-vl-utils"
        ) from error
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[precision]
    model_path = model_path.expanduser().resolve()
    output = output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = select_benchmark_rows(
        _read_jsonl(manifest), maximum_clips, seed, strategy=selection_strategy
    )
    if not rows:
        raise ValueError("manifest has no locally readable source_uri rows")
    task_config = json.loads(task_config_path.expanduser().read_text(encoding="utf-8"))
    results = (
        _load_resumable_results(
            output,
            rows,
            frame_count=frame_count,
            maximum_edge=maximum_edge,
        )
        if resume
        else []
    )
    resumed_ids = {
        str(result.get("video_id") or result.get("record_id") or "")
        for result in results
    }
    pending_rows = [
        row
        for row in rows
        if str(row.get("video_id") or row.get("request_id") or row.get("record_id") or "")
        not in resumed_ids
    ]
    fingerprint_started = time.perf_counter()
    artifact = _artifact_fingerprint(model_path)
    artifact["fingerprint_seconds"] = time.perf_counter() - fingerprint_started
    load_started = time.perf_counter()
    processor = AutoProcessor.from_pretrained(str(model_path), local_files_only=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        str(model_path),
        dtype=dtype,
        device_map="auto" if device == "cuda" else "cpu",
        local_files_only=True,
    )
    model.eval()
    if device == "cuda":
        torch.cuda.synchronize()
    load_seconds = time.perf_counter() - load_started
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    model_memory_mb = (
        torch.cuda.memory_allocated() / 1024**2 if device == "cuda" else None
    )
    inference_started = time.perf_counter()
    newly_completed = 0
    for row, source, start_s, end_s, frames, decode_seconds in _prefetched_decodes(
        pending_rows, frame_count, decode_workers
    ):
        clip_started = time.perf_counter()
        activity = str(row.get("task_id") or (row.get("activities") or [""])[0])
        prompt = _prompt(task_config, activity, frame_count)
        preprocess_started = time.perf_counter()
        try:
            inputs = _prepare_model_inputs(
                processor,
                frames,
                prompt,
                maximum_edge,
                frame_count / max(end_s - start_s, 1e-6),
            )
        finally:
            for frame in frames:
                frame.close()
        inputs = inputs.to(model.device)
        if device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        preprocess_seconds = time.perf_counter() - preprocess_started
        generation_started = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        if device == "cuda":
            torch.cuda.synchronize()
        generation_seconds = time.perf_counter() - generation_started
        trimmed = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs.input_ids, generated)
        ]
        response = processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        parsed, parse_error = parse_structured_response(response)
        task_order = list(task_config.get("model_tasks", {}))
        if parsed is not None:
            parsed, parse_error = normalize_sparse_findings(parsed, task_order)
        results.append(
            {
                "schema_version": SCHEMA_VERSION,
                "status": "succeeded",
                "video_id": row.get("video_id") or row.get("request_id"),
                "record_id": row.get("record_id") or row.get("request_id"),
                "source_dataset": row.get("source_dataset"),
                "selection_source": row.get("selection_source"),
                "event_codes": row.get("event_codes") or [],
                "source_uri_sha256": hashlib.sha256(str(source).encode("utf-8")).hexdigest(),
                "clip_start_s": start_s,
                "clip_end_s": end_s,
                "frame_count": frame_count,
                "maximum_edge": maximum_edge,
                "decode_seconds": decode_seconds,
                "preprocess_seconds": preprocess_seconds,
                "generation_seconds": generation_seconds,
                "total_seconds": decode_seconds + time.perf_counter() - clip_started,
                "input_tokens": int(inputs.input_ids.shape[-1]),
                "output_tokens": int(generated.shape[-1] - inputs.input_ids.shape[-1]),
                "peak_inference_vram_mb": (
                    torch.cuda.max_memory_allocated() / 1024**2
                    if device == "cuda"
                    else None
                ),
                "structured_json_valid": parsed is not None,
                "parse_error": parse_error,
                "parsed_response": parsed,
                "raw_response": response,
                "label_role": "unscored_prediction_not_gold",
            }
        )
        newly_completed += 1
        elapsed = time.perf_counter() - inference_started
        completed = len(results)
        write_jsonl(output / "predictions.partial.jsonl", results)
        write_json(
            output / "progress.json",
            {
                "schema_version": SCHEMA_VERSION,
                "status": "running",
                "completed_clips": completed,
                "total_clips": len(rows),
                "resumed_clips": len(resumed_ids),
                "elapsed_seconds": elapsed,
                "average_seconds_per_new_clip": elapsed / newly_completed,
                "eta_seconds": elapsed / newly_completed * (len(rows) - completed),
                "structured_json_valid_clips": sum(
                    int(item["structured_json_valid"]) for item in results
                ),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
        )
    current_session_wall_seconds = time.perf_counter() - inference_started
    total_seconds = sum(float(item["total_seconds"]) for item in results)
    total_video_seconds = sum(
        float(item["clip_end_s"] - item["clip_start_s"]) for item in results
    )
    new_results = results[len(resumed_ids) :]
    current_session_video_seconds = sum(
        float(item["clip_end_s"] - item["clip_start_s"]) for item in new_results
    )
    source_counts: Dict[str, int] = {}
    selection_counts: Dict[str, int] = {}
    for row in rows:
        source = str(row.get("source_dataset") or "unknown")
        selection = str(row.get("selection_source") or "unknown")
        source_counts[source] = source_counts.get(source, 0) + 1
        selection_counts[selection] = selection_counts.get(selection, 0) + 1
    report = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "preliminary_speed_and_structured_output_only",
        "accuracy_measured": False,
        "accuracy_claim_authorized": False,
        "model_id": model_id or model_path.name,
        "model_path_sha256": hashlib.sha256(str(model_path).encode("utf-8")).hexdigest(),
        "model_artifact": artifact,
        "parameter_count": parameter_count,
        "precision": precision,
        "device": device,
        "model_load_seconds": load_seconds,
        "model_memory_mb": model_memory_mb,
        "clips": len(results),
        "resumed_clips": len(resumed_ids),
        "decode_workers": decode_workers,
        "source_counts": source_counts,
        "selection_counts": selection_counts,
        "successful_clips": len(results),
        "structured_json_valid_clips": sum(
            int(item["structured_json_valid"]) for item in results
        ),
        "video_seconds": total_video_seconds,
        "aggregate_service_seconds": total_seconds,
        "wall_seconds_excluding_model_load": current_session_wall_seconds,
        "current_session_video_seconds": current_session_video_seconds,
        "video_hours_per_wall_hour": (
            current_session_video_seconds / current_session_wall_seconds
            if current_session_wall_seconds else 0.0
        ),
        "latency_seconds": {
            "total_p50": _percentile([item["total_seconds"] for item in results], 0.50),
            "total_p95": _percentile([item["total_seconds"] for item in results], 0.95),
            "decode_p50": _percentile([item["decode_seconds"] for item in results], 0.50),
            "preprocess_p50": _percentile([item["preprocess_seconds"] for item in results], 0.50),
            "generation_p50": _percentile([item["generation_seconds"] for item in results], 0.50),
        },
        "code_version": code_version(),
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda_device": torch.cuda.get_device_name() if device == "cuda" else None,
        },
        "input_protocol": {
            "frame_count": frame_count,
            "maximum_edge": maximum_edge,
            "selection_seed": seed,
            "selection_strategy": selection_strategy,
            "maximum_clips": maximum_clips,
            "resume": resume,
            "decode_workers": decode_workers,
            "max_new_tokens": max_new_tokens,
            "wire_output_schema": "compact_sparse_findings_v1",
            "task_order": list(task_config.get("model_tasks", {})),
        },
    }
    write_jsonl(output / "predictions.jsonl", results)
    write_json(output / "benchmark.json", report)
    write_json(
        output / "progress.json",
        {
            "schema_version": SCHEMA_VERSION,
            "status": "completed",
            "completed_clips": len(results),
            "total_clips": len(results),
            "resumed_clips": len(resumed_ids),
            "elapsed_seconds": total_seconds,
            "eta_seconds": 0.0,
            "structured_json_valid_clips": report["structured_json_valid_clips"],
            "updated_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    return report
