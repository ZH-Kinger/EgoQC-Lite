from __future__ import annotations

from pathlib import Path
from typing import Any, BinaryIO, Dict, List, Optional, Tuple, Union

import numpy as np

from .types import Issue


def _frame_quality(frame: Any) -> Dict[str, float]:
    rgb = frame.to_ndarray(format="rgb24").astype(np.float32)
    gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    if min(gray.shape) >= 3:
        laplacian = (
            -4.0 * gray[1:-1, 1:-1]
            + gray[:-2, 1:-1]
            + gray[2:, 1:-1]
            + gray[1:-1, :-2]
            + gray[1:-1, 2:]
        )
        blur_variance = float(np.var(laplacian))
    else:
        blur_variance = 0.0
    return {
        "time_s": float(frame.time) if frame.time is not None else float("nan"),
        "blur_laplacian_variance": blur_variance,
        "luma_mean": float(np.mean(gray)),
        "dark_clip_ratio": float(np.mean(gray <= 5.0)),
        "bright_clip_ratio": float(np.mean(gray >= 250.0)),
        "key_frame": bool(frame.key_frame),
    }


def _quality_issues(
    path: Union[Path, str], samples: List[Dict[str, float]], options: Dict[str, Any]
) -> List[Issue]:
    if not samples:
        return [Issue("video_sample_decode_failed", "warning", "未能解码质量抽样帧", file=str(path))]
    blur_min = float(options.get("blur_laplacian_variance_min", 25.0))
    blurry_ratio = float(np.mean([sample["blur_laplacian_variance"] < blur_min for sample in samples]))
    dark_mean = float(options.get("luma_mean_min", 20.0))
    bright_mean = float(options.get("luma_mean_max", 235.0))
    clip_max = float(options.get("clipped_pixel_ratio_max", 0.85))
    exposure_bad = [
        sample["luma_mean"] < dark_mean
        or sample["luma_mean"] > bright_mean
        or sample["dark_clip_ratio"] > clip_max
        or sample["bright_clip_ratio"] > clip_max
        for sample in samples
    ]
    issues: List[Issue] = []
    ratio_limit = float(options.get("bad_sample_ratio_warning", 0.5))
    if blurry_ratio > ratio_limit:
        issues.append(
            Issue(
                "video_blur_candidate",
                "warning",
                f"质量抽样中 {blurry_ratio:.1%} 帧低于清晰度阈值 {blur_min:g}",
                file=str(path),
                evidence={"sample_count": len(samples), "bad_ratio": blurry_ratio},
            )
        )
    exposure_ratio = float(np.mean(exposure_bad))
    if exposure_ratio > ratio_limit:
        issues.append(
            Issue(
                "video_exposure_candidate",
                "warning",
                f"质量抽样中 {exposure_ratio:.1%} 帧疑似过暗、过亮或大面积截断",
                file=str(path),
                evidence={"sample_count": len(samples), "bad_ratio": exposure_ratio},
            )
        )
    return issues


def probe_video(
    path: Union[Path, BinaryIO],
    mode: str = "header",
    options: Optional[Dict[str, Any]] = None,
    source_name: Optional[str] = None,
) -> Tuple[Dict[str, Any], List[Issue]]:
    """Probe a video at one of three explicit cost levels.

    ``header`` never decodes pixels, ``count`` fully decodes for an exact frame
    count, and ``sample-quality`` seeks to a small stratified set of frames.
    """

    if mode not in {"header", "count", "sample-quality"}:
        raise ValueError(f"unknown video check mode: {mode}")
    options = options or {}
    display_path = source_name or str(path)
    issues: List[Issue] = []
    try:
        import av
    except ImportError:
        return {}, [Issue("video_probe_unavailable", "warning", "未安装 PyAV，跳过视频探测", file=display_path)]
    try:
        source = str(path) if isinstance(path, Path) else path
        with av.open(source) as container:
            stream = container.streams.video[0]
            rate = float(stream.average_rate) if stream.average_rate else None
            duration = float(stream.duration * stream.time_base) if stream.duration else None
            metadata: Dict[str, Any] = {
                "check_mode": mode,
                "container_format": container.format.name,
                "codec": stream.codec_context.name,
                "width": stream.codec_context.width,
                "height": stream.codec_context.height,
                "pix_fmt": stream.codec_context.pix_fmt,
                "average_rate": rate,
                "time_base": str(stream.time_base),
                "duration": duration,
                "reported_frames": int(stream.frames or 0),
                "audio_streams": len(container.streams.audio),
            }
            if mode == "count":
                count = 0
                first_key_frame = None
                previous_time = None
                gap_count = 0
                jitter_abs_sum_s = 0.0
                jitter_abs_max_s = 0.0
                non_monotonic_timestamps = 0
                nominal_interval_s = 1.0 / rate if rate and rate > 0 else None
                jitter_mean_limit = float(options.get("frame_interval_jitter_mean_ms_max", 2.0))
                jitter_max_limit = float(options.get("frame_interval_jitter_max_ms_max", 5.0))
                jitter_event_limit = max(0, int(options.get("frame_interval_jitter_event_limit", 32)))
                jitter_event_count = 0
                jitter_events: List[Dict[str, float]] = []
                for frame in container.decode(stream):
                    if first_key_frame is None:
                        first_key_frame = bool(frame.key_frame)
                    current_time = float(frame.time) if frame.time is not None else None
                    if current_time is not None and previous_time is not None:
                        gap = current_time - previous_time
                        if gap <= 0:
                            non_monotonic_timestamps += 1
                        elif nominal_interval_s is not None:
                            jitter = abs(gap - nominal_interval_s)
                            jitter_abs_sum_s += jitter
                            jitter_abs_max_s = max(jitter_abs_max_s, jitter)
                            gap_count += 1
                            jitter_ms = 1000.0 * jitter
                            if jitter_ms > jitter_max_limit:
                                jitter_event_count += 1
                                if len(jitter_events) < jitter_event_limit:
                                    jitter_events.append({
                                        "frame_index": count,
                                        "time_s": current_time,
                                        "gap_ms": 1000.0 * gap,
                                        "jitter_ms": jitter_ms,
                                    })
                    if current_time is not None:
                        previous_time = current_time
                    count += 1
                metadata["counted_frames"] = count
                metadata["first_decoded_frame_keyframe"] = first_key_frame
                metadata["timestamp_gap_count"] = gap_count
                metadata["non_monotonic_timestamps"] = non_monotonic_timestamps
                metadata["frame_interval_jitter_mean_ms"] = (
                    1000.0 * jitter_abs_sum_s / gap_count if gap_count else None
                )
                metadata["frame_interval_jitter_max_ms"] = (
                    1000.0 * jitter_abs_max_s if gap_count else None
                )
                metadata["frame_interval_jitter_event_count"] = jitter_event_count
                metadata["frame_interval_jitter_events"] = jitter_events
                if stream.frames and count != int(stream.frames):
                    issues.append(
                        Issue(
                            "video_reported_frame_mismatch",
                            "warning",
                            f"容器报告 {int(stream.frames)} 帧，实际解码 {count} 帧",
                            file=display_path,
                            evidence={"reported_frames": int(stream.frames), "counted_frames": count},
                        )
                    )
                jitter_mean = metadata["frame_interval_jitter_mean_ms"]
                jitter_max = metadata["frame_interval_jitter_max_ms"]
                if (
                    jitter_mean is not None
                    and (jitter_mean > jitter_mean_limit or jitter_max > jitter_max_limit)
                ):
                    issues.append(
                        Issue(
                            "video_frame_interval_jitter",
                            "warning",
                            f"帧间隔抖动 mean={jitter_mean:.3f}ms max={jitter_max:.3f}ms",
                            file=display_path,
                            evidence={
                                "mean_ms": jitter_mean,
                                "max_ms": jitter_max,
                                "mean_limit_ms": jitter_mean_limit,
                                "max_limit_ms": jitter_max_limit,
                                "event_count": jitter_event_count,
                                "events_truncated": jitter_event_count > len(jitter_events),
                            },
                        )
                    )
                if non_monotonic_timestamps:
                    issues.append(
                        Issue(
                            "video_non_monotonic_timestamps",
                            "error",
                            f"发现 {non_monotonic_timestamps} 个非递增视频时间戳",
                            file=display_path,
                        )
                    )
                if first_key_frame is False:
                    issues.append(
                        Issue(
                            "video_first_frame_not_keyframe",
                            "warning",
                            "首个解码帧不是关键帧，随机访问和分段审核可能变慢",
                            file=display_path,
                        )
                    )
            elif mode == "sample-quality":
                sample_count = max(1, int(options.get("sample_frames", 8)))
                usable_duration = duration
                if usable_duration is None and rate and stream.frames:
                    usable_duration = float(stream.frames) / rate
                if not usable_duration or usable_duration <= 0:
                    targets = [0.0]
                else:
                    targets = np.linspace(0.0, max(0.0, usable_duration - 1.0 / (rate or 30.0)), sample_count)
                samples: List[Dict[str, float]] = []
                for target in targets:
                    container.seek(
                        max(0, int(float(target) * av.time_base)),
                        any_frame=False,
                        backward=True,
                    )
                    selected = None
                    for frame in container.decode(stream):
                        selected = frame
                        if frame.time is None or float(frame.time) >= float(target) - 0.5 / (rate or 30.0):
                            break
                    if selected is not None:
                        samples.append(_frame_quality(selected))
                metadata["quality_samples"] = samples
                metadata["quality_sample_count"] = len(samples)
                if samples:
                    metadata["blur_laplacian_variance_min"] = min(
                        sample["blur_laplacian_variance"] for sample in samples
                    )
                    metadata["luma_mean_min"] = min(sample["luma_mean"] for sample in samples)
                    metadata["luma_mean_max"] = max(sample["luma_mean"] for sample in samples)
                issues.extend(_quality_issues(display_path, samples, options))
            return metadata, issues
    except Exception as exc:
        issues.append(Issue("video_open_failed", "error", f"视频无法打开或解码: {exc}", file=display_path))
        return {}, issues
