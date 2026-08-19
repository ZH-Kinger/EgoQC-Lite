from __future__ import annotations

import base64
import io
import json
import math
import os
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlparse

import av
import numpy as np
from PIL import Image

from .report import write_json, write_jsonl


SCHEMA_VERSION = "egoqc-teacher-api-run-v1"

BAILIAN_SHARED_BASE_URLS = {
    "beijing": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "singapore": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    "virginia": "https://dashscope-us.aliyuncs.com/compatible-mode/v1",
}

BAILIAN_WORKSPACE_REGIONS = {
    "beijing": "cn-beijing",
    "singapore": "ap-southeast-1",
    "virginia": "us-east-1",
}

COST_PROFILES = {
    "low": {"sample_fps": 1.5, "max_frames": 12, "max_edge": 448, "jpeg_quality": 72},
    "balanced": {"sample_fps": 2.0, "max_frames": 16, "max_edge": 640, "jpeg_quality": 78},
    "quality": {"sample_fps": 4.0, "max_frames": 32, "max_edge": 768, "jpeg_quality": 82},
}


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path} 第 {line_number} 行不是合法 JSON") from error
        if not isinstance(row, dict):
            raise ValueError(f"{path} 第 {line_number} 行必须是对象")
        rows.append(row)
    return rows


def _target_times(start_s: float, end_s: float, sample_fps: float, max_frames: int) -> np.ndarray:
    duration = end_s - start_s
    if start_s < 0 or duration <= 0:
        raise ValueError(f"clip 时间范围非法: [{start_s}, {end_s}]")
    count = min(max_frames, max(2, int(math.ceil(duration * sample_fps))))
    margin = min(duration / 4.0, 0.5 / sample_fps)
    return np.linspace(start_s + margin, end_s - margin, count)


def extract_clip_frames(
    source_uri: str,
    start_s: float,
    end_s: float,
    *,
    sample_fps: float = 2.0,
    max_frames: int = 16,
    max_edge: int = 768,
    jpeg_quality: int = 80,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Decode a bounded ordered frame sequence without materializing a video clip."""

    if sample_fps <= 0 or max_frames < 2 or max_edge <= 0:
        raise ValueError("sample_fps/max_frames/max_edge 参数非法")
    if not 1 <= jpeg_quality <= 95:
        raise ValueError("jpeg_quality 必须在 1..95")
    path = Path(source_uri).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"视频不存在: {path}")
    targets = _target_times(start_s, end_s, sample_fps, max_frames)
    selected: List[Tuple[float, Image.Image]] = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        if stream.time_base is not None and start_s > 0:
            seek_s = max(0.0, start_s - 1.0)
            container.seek(
                int(seek_s / float(stream.time_base)),
                stream=stream,
                backward=True,
                any_frame=False,
            )
        target_index = 0
        rate = float(stream.average_rate) if stream.average_rate else 30.0
        decoded_index = 0
        for frame in container.decode(stream):
            frame_time = (
                float(frame.time)
                if frame.time is not None
                else start_s + decoded_index / max(rate, 1e-6)
            )
            decoded_index += 1
            if frame_time + 0.5 / max(rate, 1e-6) < start_s:
                continue
            while target_index < len(targets) and frame_time >= float(targets[target_index]):
                selected.append((frame_time, frame.to_image().convert("RGB")))
                target_index += 1
            if target_index >= len(targets) or frame_time > end_s:
                break
    if not selected:
        raise ValueError(f"clip 未解码到帧: {path} [{start_s}, {end_s}]")

    frames: List[Dict[str, Any]] = []
    encoded_bytes = 0
    original_sizes = []
    encoded_sizes = []
    try:
        for frame_time, image in selected:
            original_sizes.append([image.width, image.height])
            if max(image.size) > max_edge:
                image.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
            encoded_sizes.append([image.width, image.height])
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=jpeg_quality, optimize=True)
            payload = buffer.getvalue()
            encoded_bytes += len(payload)
            frames.append({
                "time_s": frame_time,
                "relative_time_s": frame_time - start_s,
                "data_url": "data:image/jpeg;base64," + base64.b64encode(payload).decode("ascii"),
            })
    finally:
        for _, image in selected:
            image.close()
    return frames, {
        "frame_count": len(frames),
        "encoded_bytes": encoded_bytes,
        "original_sizes": original_sizes,
        "encoded_sizes": encoded_sizes,
        "sample_fps": sample_fps,
    }


def _prompt(request: Dict[str, Any]) -> str:
    task_names = list(request.get("candidate_tasks") or [])
    dimensions = request.get("assessment_dimensions") or {}
    required = request.get("required_response") or {}
    return (
        "你是具身智能第一视角数据质量审核员。按时间顺序检查所有图像；规则触发项只是召回提示，"
        "不能限制你发现其他问题。区分确定事实与不确定推断。不要因为片段进入审核队列就默认有错。\n"
        f"规则事件: {json.dumps(request.get('event_codes') or [], ensure_ascii=False)}\n"
        f"触发任务: {json.dumps(request.get('trigger_tasks') or [], ensure_ascii=False)}\n"
        f"数据任务标签: {json.dumps(request.get('task_context') or {}, ensure_ascii=False)}\n"
        f"数据能力清单: {json.dumps(request.get('capability_context') or {}, ensure_ascii=False)}\n"
        f"视觉证据类型: {request.get('visual_evidence') or 'raw_video'}\n"
        f"固定任务: {json.dumps(task_names, ensure_ascii=False)}\n"
        f"审查维度: {json.dumps(dimensions, ensure_ascii=False)}\n"
        "每个固定任务必须输出 probability 和 confidence。开放问题写入 findings，并给出相对 clip 的"
        "时间范围。信息不足时降低 confidence，不要编造不可见的标注。只输出一个 JSON 对象，不要 Markdown。\n"
        f"输出结构: {json.dumps(required, ensure_ascii=False)}"
    )


def build_chat_payload(
    request: Dict[str, Any],
    model: str,
    frames: Iterable[Dict[str, Any]],
    *,
    response_format: bool = True,
    media_mode: str = "ordered_images",
    sample_fps: float = 2.0,
) -> Dict[str, Any]:
    content: List[Dict[str, Any]] = [{"type": "text", "text": _prompt(request)}]
    frame_rows = list(frames)
    if media_mode == "bailian_video_frames":
        content.append({
            "type": "text",
            "text": "frame_relative_times_s=" + json.dumps(
                [round(float(frame["relative_time_s"]), 4) for frame in frame_rows]
            ),
        })
        content.append({
            "type": "video",
            "video": [frame["data_url"] for frame in frame_rows],
            "fps": sample_fps,
        })
    elif media_mode == "ordered_images":
        for index, frame in enumerate(frame_rows):
            content.append({
                "type": "text",
                "text": f"frame={index:02d}, relative_time={float(frame['relative_time_s']):.3f}s",
            })
            content.append({
                "type": "image_url",
                "image_url": {"url": frame["data_url"], "detail": "low"},
            })
    else:
        raise ValueError(f"不支持 media_mode={media_mode}")
    payload: Dict[str, Any] = {
        "model": model,
        "temperature": 0,
        "messages": [
            {
                "role": "system",
                "content": "输出严格、可解析、基于证据的 JSON。",
            },
            {"role": "user", "content": content},
        ],
    }
    if response_format:
        payload["response_format"] = {"type": "json_object"}
    return payload


def _content(response: Dict[str, Any]) -> str:
    choices = response.get("choices") or []
    if not choices:
        raise ValueError("API 响应缺少 choices")
    content = choices[0].get("message", {}).get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(item.get("text", ""))
            for item in content
            if isinstance(item, dict) and item.get("type") in {"text", "output_text"}
        )
    raise ValueError("API 响应 message.content 类型不支持")


def _parse_json_content(content: str) -> Dict[str, Any]:
    value = content.strip()
    if value.startswith("```"):
        lines = value.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        value = "\n".join(lines)
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("教师输出必须是 JSON 对象")
    return parsed


def normalize_teacher_label(label: Dict[str, Any], request: Dict[str, Any]) -> None:
    """Clamp non-authoritative finding timestamps while preserving an audit trail.

    Findings are free-text evidence, not training targets. Some VLM responses use
    the source-video clock or round a clip endpoint upward. Rejecting an otherwise
    valid structured label would trigger paid retries, so normalize only these
    timestamps and retain the original values in the artifact.
    """

    clip_duration = float(request["clip_end_s"]) - float(request["clip_start_s"])
    findings = label.get("findings", [])
    if not isinstance(findings, list):
        return
    for finding in findings:
        if not isinstance(finding, dict):
            continue
        try:
            start_s = float(finding.get("start_s"))
            end_s = float(finding.get("end_s"))
        except (TypeError, ValueError):
            continue
        if not math.isfinite(start_s) or not math.isfinite(end_s) or end_s < start_s:
            continue
        normalized_start = min(clip_duration, max(0.0, start_s))
        normalized_end = min(clip_duration, max(normalized_start, end_s))
        if normalized_start != start_s or normalized_end != end_s:
            finding["time_normalization"] = {
                "original_start_s": start_s,
                "original_end_s": end_s,
                "reason": "clamped_to_reviewed_clip",
            }
            finding["start_s"] = normalized_start
            finding["end_s"] = normalized_end


def validate_teacher_label(label: Dict[str, Any], request: Dict[str, Any]) -> None:
    if label.get("schema_version") != "egoqc-visual-teacher-v1":
        raise ValueError("教师输出 schema_version 不兼容")
    tasks = label.get("tasks")
    if not isinstance(tasks, dict):
        raise ValueError("教师输出缺少 tasks")
    expected_tasks = set(request.get("candidate_tasks") or [])
    unknown_tasks = sorted(set(tasks) - expected_tasks)
    if unknown_tasks:
        raise ValueError(f"教师输出包含未知 tasks: {unknown_tasks}")
    for task in expected_tasks:
        value = tasks.get(task)
        if not isinstance(value, dict):
            raise ValueError(f"教师输出缺少 task={task}")
        probability = float(value.get("probability"))
        confidence = float(value.get("confidence"))
        if not 0 <= probability <= 1 or not 0 <= confidence <= 1:
            raise ValueError(f"task={task} probability/confidence 越界")
    overall = label.get("overall")
    if not isinstance(overall, dict):
        raise ValueError("教师输出缺少 overall")
    if not isinstance(overall.get("training_usable"), bool):
        raise ValueError("overall.training_usable 必须是 boolean")
    if overall.get("recommended_route") not in {"accept", "human_review", "reject"}:
        raise ValueError("overall.recommended_route 非法")
    overall_confidence = float(overall.get("confidence"))
    if not 0 <= overall_confidence <= 1:
        raise ValueError("overall.confidence 越界")
    if not isinstance(overall.get("allowed_uses"), list):
        raise ValueError("overall.allowed_uses 必须是列表")
    findings = label.get("findings", [])
    if not isinstance(findings, list):
        raise ValueError("findings 必须是列表")
    clip_duration = float(request["clip_end_s"]) - float(request["clip_start_s"])
    for index, finding in enumerate(findings):
        if not isinstance(finding, dict) or not str(finding.get("category", "")).strip():
            raise ValueError(f"findings[{index}] 缺少 category")
        if finding.get("severity") not in {"info", "warning", "error"}:
            raise ValueError(f"findings[{index}].severity 非法")
        start_s = float(finding.get("start_s"))
        end_s = float(finding.get("end_s"))
        if start_s < 0 or end_s < start_s or end_s > clip_duration + 1e-3:
            raise ValueError(f"findings[{index}] 时间范围越界")
    if not isinstance(label.get("missing_annotations", []), list):
        raise ValueError("missing_annotations 必须是列表")


def _endpoint(base_url: str) -> str:
    value = base_url.rstrip("/")
    if value.endswith("/chat/completions"):
        return value
    return value + "/chat/completions"


def _post_json(
    endpoint: str,
    api_key: str,
    payload: Dict[str, Any],
    timeout_s: float,
) -> Dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    http_request = urllib.request.Request(
        endpoint,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(http_request, timeout=timeout_s) as response:
        value = json.loads(response.read().decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("API 响应必须是 JSON 对象")
    return value


def _existing_label(request: Dict[str, Any]) -> bool:
    path = Path(request["output_path"])
    if not path.is_file():
        return False
    try:
        label = json.loads(path.read_text(encoding="utf-8"))
        validate_teacher_label(label, request)
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False
    return True


def _execute_one(
    request: Dict[str, Any],
    *,
    model: str,
    endpoint: str,
    api_key: Optional[str],
    dry_run: bool,
    overwrite: bool,
    response_format: bool,
    media_mode: str,
    sample_fps: float,
    max_frames: int,
    max_edge: int,
    jpeg_quality: int,
    timeout_s: float,
    max_retries: int,
) -> Dict[str, Any]:
    request_id = str(request["request_id"])
    started = time.perf_counter()
    if not overwrite and _existing_label(request):
        return {"request_id": request_id, "status": "cached", "elapsed_s": 0.0}
    try:
        frames, media = extract_clip_frames(
            str(request["source_uri"]),
            float(request["clip_start_s"]),
            float(request["clip_end_s"]),
            sample_fps=sample_fps,
            max_frames=max_frames,
            max_edge=max_edge,
            jpeg_quality=jpeg_quality,
        )
        payload = build_chat_payload(
            request,
            model,
            frames,
            response_format=response_format,
            media_mode=media_mode,
            sample_fps=sample_fps,
        )
        if dry_run:
            return {
                "request_id": request_id,
                "status": "dry_run",
                "media": media,
                "payload_bytes": len(json.dumps(payload, ensure_ascii=False).encode("utf-8")),
                "elapsed_s": time.perf_counter() - started,
            }
        if not api_key:
            raise ValueError("API key 为空")
        last_error: Optional[Exception] = None
        response: Dict[str, Any] = {}
        for attempt in range(max_retries + 1):
            try:
                response = _post_json(endpoint, api_key, payload, timeout_s)
                last_error = None
                break
            except (urllib.error.URLError, TimeoutError, ValueError, json.JSONDecodeError) as error:
                last_error = error
                if attempt < max_retries:
                    time.sleep(min(8.0, 0.5 * (2 ** attempt)))
        if last_error is not None:
            raise last_error
        label = _parse_json_content(_content(response))
        normalize_teacher_label(label, request)
        validate_teacher_label(label, request)
        label.update({
            "teacher_model": model,
            "prompt_version": request.get("prompt_version"),
            "request_id": request_id,
            "reviewed_clip": {
                "source_uri": request["source_uri"],
                "clip_start_s": request["clip_start_s"],
                "clip_end_s": request["clip_end_s"],
                "sampled_frames": media["frame_count"],
            },
            "api_provenance": {
                "endpoint_host": urlparse(endpoint).netloc,
                "response_id": response.get("id"),
                "usage": response.get("usage") or {},
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
        })
        write_json(Path(request["output_path"]), label)
        return {
            "request_id": request_id,
            "status": "succeeded",
            "media": media,
            "usage": response.get("usage") or {},
            "elapsed_s": time.perf_counter() - started,
        }
    except Exception as error:
        return {
            "request_id": request_id,
            "status": "failed",
            "error_type": type(error).__name__,
            "error": str(error),
            "elapsed_s": time.perf_counter() - started,
        }


def run_teacher_api(
    queue: Path,
    output: Path,
    *,
    provider: str = "openai-compatible",
    region: Optional[str] = None,
    workspace_id: Optional[str] = None,
    base_url: Optional[str],
    model: Optional[str],
    api_key_env: Optional[str] = None,
    dry_run: bool = False,
    overwrite: bool = False,
    response_format: bool = True,
    concurrency: int = 2,
    cost_profile: str = "low",
    sample_fps: Optional[float] = None,
    max_frames: Optional[int] = None,
    max_edge: Optional[int] = None,
    jpeg_quality: Optional[int] = None,
    timeout_s: float = 120.0,
    max_retries: int = 3,
    max_requests: Optional[int] = None,
    allow_external_supplier_data: bool = False,
    input_price_per_million: Optional[float] = None,
    output_price_per_million: Optional[float] = None,
) -> Dict[str, Any]:
    if concurrency < 1 or max_retries < 0 or timeout_s <= 0:
        raise ValueError("concurrency/max_retries/timeout 参数非法")
    if cost_profile not in COST_PROFILES:
        raise ValueError(f"不支持 cost_profile={cost_profile}")
    profile = COST_PROFILES[cost_profile]
    sample_fps = float(sample_fps if sample_fps is not None else profile["sample_fps"])
    max_frames = int(max_frames if max_frames is not None else profile["max_frames"])
    max_edge = int(max_edge if max_edge is not None else profile["max_edge"])
    jpeg_quality = int(
        jpeg_quality if jpeg_quality is not None else profile["jpeg_quality"]
    )
    requests = _read_jsonl(queue.expanduser().resolve())
    if max_requests is not None:
        requests = requests[: max(0, int(max_requests))]
    supplier_sources = sorted({
        str(request.get("source_dataset") or "unknown_dataset")
        for request in requests
        if request.get("source_class") == "supplier_dataset"
    })
    if supplier_sources and not dry_run and not allow_external_supplier_data:
        raise ValueError(
            "队列包含供应商数据，会将抽样帧发送给外部教师 API；"
            "经明确授权后传入 allow_external_supplier_data=True。"
            f" sources={supplier_sources}"
        )
    request_ids = [str(request.get("request_id")) for request in requests]
    if len(set(request_ids)) != len(request_ids):
        raise ValueError("教师队列存在重复 request_id，拒绝重复计费")
    if provider not in {"openai-compatible", "bailian"}:
        raise ValueError(f"不支持 provider={provider}")
    effective_model = model or os.environ.get("TEACHER_API_MODEL")
    effective_base_url = base_url or os.environ.get("TEACHER_API_BASE_URL")
    effective_key_env = api_key_env
    media_mode = "ordered_images"
    if provider == "bailian":
        effective_model = effective_model or "qwen3-vl-plus"
        effective_key_env = effective_key_env or "DASHSCOPE_API_KEY"
        effective_region = region or os.environ.get("BAILIAN_REGION") or "beijing"
        if effective_region not in BAILIAN_SHARED_BASE_URLS:
            raise ValueError(f"不支持百炼地域: {effective_region}")
        if not effective_base_url:
            if workspace_id:
                region_code = BAILIAN_WORKSPACE_REGIONS[effective_region]
                effective_base_url = (
                    f"https://{workspace_id}.{region_code}.maas.aliyuncs.com/compatible-mode/v1"
                )
            else:
                effective_base_url = BAILIAN_SHARED_BASE_URLS[effective_region]
        media_mode = "bailian_video_frames"
        region = effective_region
    else:
        effective_key_env = effective_key_env or "TEACHER_API_KEY"
    if dry_run:
        effective_model = effective_model or "dry-run-model"
        effective_base_url = effective_base_url or "https://dry-run.invalid/v1"
    if not effective_model or not effective_base_url:
        raise ValueError("需要 --model/TEACHER_API_MODEL 和 --base-url/TEACHER_API_BASE_URL")
    api_key = None if dry_run else os.environ.get(effective_key_env)
    if not dry_run and not api_key:
        raise ValueError(f"环境变量 {effective_key_env} 未设置")

    output = output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    endpoint = _endpoint(effective_base_url)
    results: List[Dict[str, Any]] = []
    kwargs = {
        "model": effective_model,
        "endpoint": endpoint,
        "api_key": api_key,
        "dry_run": dry_run,
        "overwrite": overwrite,
        "response_format": response_format,
        "media_mode": media_mode,
        "sample_fps": sample_fps,
        "max_frames": max_frames,
        "max_edge": max_edge,
        "jpeg_quality": jpeg_quality,
        "timeout_s": timeout_s,
        "max_retries": max_retries,
    }
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {
            executor.submit(_execute_one, request, **kwargs): request
            for request in requests
        }
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            write_json(output / "requests" / f"{result['request_id']}.json", result)
    results.sort(key=lambda row: row["request_id"])
    write_jsonl(output / "results.jsonl", results)
    status_counts = {
        status: sum(row["status"] == status for row in results)
        for status in ("succeeded", "cached", "dry_run", "failed")
    }
    input_tokens = sum(int(row.get("usage", {}).get("prompt_tokens", 0)) for row in results)
    output_tokens = sum(int(row.get("usage", {}).get("completion_tokens", 0)) for row in results)
    estimated_cost = None
    if input_price_per_million is not None and output_price_per_million is not None:
        estimated_cost = (
            input_tokens * input_price_per_million
            + output_tokens * output_price_per_million
        ) / 1_000_000.0
    summary = {
        "schema_version": SCHEMA_VERSION,
        "queue": str(queue.expanduser().resolve()),
        "requests": len(requests),
        "status_counts": status_counts,
        "model": effective_model,
        "provider": provider,
        "region": region,
        "endpoint_host": urlparse(endpoint).netloc,
        "dry_run": dry_run,
        "cost_profile": cost_profile,
        "credentials_stored": False,
        "external_supplier_data_authorized": bool(
            supplier_sources and allow_external_supplier_data
        ),
        "supplier_sources": supplier_sources,
        "media_mode": media_mode,
        "sample_fps": sample_fps,
        "max_frames": max_frames,
        "max_edge": max_edge,
        "jpeg_quality": jpeg_quality,
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
        "estimated_cost": estimated_cost,
        "results": str(output / "results.jsonl"),
    }
    write_json(output / "summary.json", summary)
    return summary
