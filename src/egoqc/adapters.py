from __future__ import annotations

import json
import tarfile
from dataclasses import dataclass
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc

from .math3d import axis_angle_to_matrix, matrix_to_euler_xyz, transform_points
from .validator import load_episode_index
from .canonical import (
    CanonicalEpisode,
    CapabilityManifest,
    HandTrack,
    VideoReference,
)
from .video import probe_video
from .types import Issue
from .generic_ego import inspect_generic_ego_video


LEGACY_COLUMNS = [
    "timestamp",
    "frame_index",
    "episode_index",
    "index",
    "task_index",
    "observation.mano.left.present",
    "observation.mano.left.state",
    "observation.mano.left.global_orient",
    "observation.mano.left.body_pose",
    "observation.mano.left.cam_trans",
    "observation.mano.left.betas",
    "observation.mano.right.present",
    "observation.mano.right.state",
    "observation.mano.right.global_orient",
    "observation.mano.right.body_pose",
    "observation.mano.right.cam_trans",
    "observation.mano.right.betas",
    "camera.intrinsic",
    "camera.extrinsic.T_world_camera",
]


@dataclass
class UnifiedHandRecord:
    """In-memory standard view shared by validators, repair and render stages."""

    values: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return self.values


@dataclass
class AdaptedEpisode:
    records: List[UnifiedHandRecord]
    route: Dict[str, Any]
    provenance: Dict[str, Any]


def detect_adapter(dataset: Path) -> str:
    dataset = dataset.expanduser().resolve()
    if dataset.is_file() and dataset.suffix.lower() in {".mp4", ".mov", ".avi"}:
        return "raw_video"
    if dataset.is_file() and dataset.suffix.lower() in {".hdf5", ".h5"}:
        return "egodex_hdf5"
    reka_index = dataset / "metadata" / "index.parquet"
    if reka_index.exists():
        names = set(pq.ParquetFile(reka_index).schema_arrow.names)
        if {"video_id", "project", "duration_s", "fps", "num_frames"} <= names:
            return "rekadaily_raw"
    info_path = dataset / "meta" / "info.json"
    if not info_path.exists():
        if next(dataset.glob("*.hdf5"), None) is not None:
            return "egodex_hdf5"
        if any((dataset / name).is_dir() for name in ("part1", "part2", "test", "extra")):
            return "egodex_collection"
        return "unsupported"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    if info.get("robot_type") == "mano_hamer":
        return "mano_hamer"
    features = info.get("features", {})
    if "extrinsics_w2c" in features and "state_mask" in features:
        return "standard_v3"
    return "unsupported"


def _route(dataset: Path, episode: int, video_key: str) -> Dict[str, Any]:
    matches = [
        row for row in load_episode_index(dataset).to_pylist()
        if int(row["episode_index"]) == episode
    ]
    if len(matches) != 1:
        raise ValueError(f"episode {episode} 路由数量应为 1，实际 {len(matches)}")
    row = matches[0]
    return {
        "metadata": row,
        "data_path": dataset / "data" / f"chunk-{int(row['data/chunk_index']):03d}" / f"file-{int(row['data/file_index']):03d}.parquet",
        "video_path": dataset / "videos" / video_key / f"chunk-{int(row[f'videos/{video_key}/chunk_index']):03d}" / f"file-{int(row[f'videos/{video_key}/file_index']):03d}.mp4",
        "start_time": float(row[f"videos/{video_key}/from_timestamp"]),
        "stop_time": float(row[f"videos/{video_key}/to_timestamp"]),
        "length": int(row["length"]),
    }


class ManoHamerAdapter:
    """Readonly adapter for the observed LeRobot v3 ``mano_hamer`` layout."""

    name = "mano_hamer"
    video_key = "front_camera"

    @staticmethod
    def capabilities() -> CapabilityManifest:
        return CapabilityManifest(
            video=True,
            camera_intrinsics=True,
            camera_trajectory=True,
            mano_parameters=True,
            prediction_confidence=False,
            task_labels=True,
            independent_timestamps=True,
        )

    def load_episode(self, dataset: Path, episode: int) -> AdaptedEpisode:
        dataset = dataset.expanduser().resolve()
        info = json.loads((dataset / "meta" / "info.json").read_text(encoding="utf-8"))
        route = _route(dataset, episode, self.video_key)
        parquet = pq.ParquetFile(route["data_path"])
        missing = sorted(set(LEGACY_COLUMNS) - set(parquet.schema_arrow.names))
        if missing:
            raise ValueError(f"mano_hamer 缺少字段 {missing}: {route['data_path']}")
        table = parquet.read(columns=LEGACY_COLUMNS)
        episode_values = np.asarray(table["episode_index"].to_pylist(), dtype=np.int64)
        table = table.filter(pa.array(episode_values == episode))
        rows = table.to_pylist()
        rows.sort(key=lambda row: int(row["frame_index"]))
        records = self.normalize_rows(rows, info)
        if len(records) != route["length"]:
            raise ValueError(
                f"episode {episode} metadata length={route['length']}，Parquet rows={len(records)}"
            )
        provenance = {
            "adapter": self.name,
            "readonly": True,
            "source_dtype": "float32",
            "normalized_dtype": "float64",
            "stored_camera_extrinsic": info.get("stored_camera_extrinsic"),
            "normalized_camera_extrinsic": "world_to_camera",
            "state_semantics": info.get("state_semantics"),
            "left_pose_convention": "assumed_already_canonical; requires dataset-owner confirmation",
            "acceptance_scope": "compatibility_preview_only_until_left_pose_convention_is_confirmed",
        }
        return AdaptedEpisode(records, route, provenance)

    def normalize_rows(
        self, rows: Sequence[Dict[str, Any]], info: Dict[str, Any]
    ) -> List[UnifiedHandRecord]:
        video_shape = info.get("features", {}).get(self.video_key, {}).get("shape", [0, 0, 3])
        height, width = int(video_shape[0]), int(video_shape[1])
        output: List[UnifiedHandRecord] = []
        for source in rows:
            intrinsic = np.asarray(source["camera.intrinsic"], dtype=np.float64).reshape(3, 3)
            stored_c2w = np.asarray(
                source["camera.extrinsic.T_world_camera"], dtype=np.float64
            ).reshape(4, 4)
            extrinsic_w2c = np.linalg.inv(stored_c2w)
            state = np.zeros(122, dtype=np.float64)
            present = []
            normalized: Dict[str, Any] = {}
            for side_index, side in enumerate(("left", "right")):
                prefix = f"observation.mano.{side}"
                valid = bool(source[f"{prefix}.present"])
                present.append(valid)
                legacy_state = np.asarray(source[f"{prefix}.state"], dtype=np.float64)
                world_position = legacy_state[48:51]
                world_wrist = axis_angle_to_matrix(legacy_state[0:3])
                local_pose = axis_angle_to_matrix(legacy_state[3:48].reshape(15, 3))
                offset = side_index * 61
                if valid:
                    state[offset : offset + 3] = transform_points(extrinsic_w2c, world_position)
                    state[offset + 3 : offset + 6] = matrix_to_euler_xyz(
                        extrinsic_w2c[:3, :3] @ world_wrist
                    )
                    state[offset + 6 : offset + 51] = matrix_to_euler_xyz(local_pose).reshape(-1)
                state[offset + 51 : offset + 61] = legacy_state[51:61]
                normalized[f"{side}_transl_world"] = world_position.tolist()
                normalized[f"{side}_orient_world"] = world_wrist.reshape(-1).tolist()
                normalized[f"{side}_hand_pose"] = local_pose.reshape(-1).tolist()
                normalized[f"{side}_kept"] = valid
                normalized[f"{side}_seg_start"] = -1
                normalized[f"{side}_seg_end"] = -1
            if present == [True, False]:
                main_type = 0
            elif present == [False, True]:
                main_type = 1
            else:
                main_type = -1
            fov = [
                2.0 * np.arctan(width / (2.0 * intrinsic[0, 0])),
                2.0 * np.arctan(height / (2.0 * intrinsic[1, 1])),
            ]
            normalized.update(
                {
                    "index": int(source["index"]),
                    "frame_index": int(source["frame_index"]),
                    "episode_index": int(source["episode_index"]),
                    "task_index": int(source["task_index"]),
                    "main_type": main_type,
                    "timestamp": float(source["timestamp"]),
                    "state_mask": present,
                    "observation.state": state.tolist(),
                    "fov": np.asarray(fov, dtype=np.float64).tolist(),
                    "intrinsics": intrinsic.reshape(-1).tolist(),
                    "extrinsics_w2c": extrinsic_w2c.reshape(-1).tolist(),
                }
            )
            output.append(UnifiedHandRecord(normalized))
        return output


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    return value


class EgoDexHDF5Adapter:
    """Readonly adapter for paired EgoDex ``episode.hdf5 + episode.mp4`` files."""

    name = "egodex_hdf5"

    @staticmethod
    def capabilities() -> CapabilityManifest:
        return CapabilityManifest(
            video=True,
            camera_intrinsics=True,
            camera_trajectory=True,
            hand_joint_transforms=True,
            mano_parameters=False,
            prediction_confidence=True,
            task_labels=True,
            subtask_labels=False,
            tactile=False,
            hand_ground_truth=False,
            trajectory_ground_truth=False,
            independent_timestamps=False,
        )


    @staticmethod
    def _resolve_episode(dataset: Path, episode: Union[int, str]) -> tuple[Path, Path, str]:
        dataset = dataset.expanduser().resolve()
        if dataset.is_file():
            hdf5_path = dataset
            root = dataset.parent
        else:
            root = dataset
            reference = Path(str(episode))
            hdf5_path = reference if reference.is_absolute() else dataset / reference
            if hdf5_path.suffix.lower() not in {".hdf5", ".h5"}:
                hdf5_path = hdf5_path.with_suffix(".hdf5")
        hdf5_path = hdf5_path.resolve()
        try:
            hdf5_path.relative_to(root.resolve())
        except ValueError as error:
            raise ValueError(f"episode 路径越出数据集根目录: {hdf5_path}") from error
        video_path = hdf5_path.with_suffix(".mp4")
        if not hdf5_path.exists():
            raise FileNotFoundError(hdf5_path)
        if not video_path.exists():
            raise FileNotFoundError(video_path)
        episode_id = hdf5_path.relative_to(root.resolve()).with_suffix("").as_posix()
        return hdf5_path, video_path, episode_id

    @staticmethod
    def _hand_track(handle: Any, side: str, frame_count: int, threshold: float) -> HandTrack:
        transform_group = handle["transforms"]
        confidence_group = handle["confidences"]
        prefix = side.lower()
        root_name = f"{prefix}Hand"
        # EgoDex uses ``leftThumb...`` but ``leftIndexFinger...`` for the other
        # digits. Use an explicit source-schema list so a missing digit cannot be
        # silently accepted as a different hand model.
        finger_names = [
            f"{prefix}Thumb{segment}"
            for segment in ("Knuckle", "IntermediateBase", "IntermediateTip", "Tip")
        ]
        finger_names.extend(
            f"{prefix}{finger}Finger{segment}"
            for finger in ("Index", "Middle", "Ring", "Little")
            for segment in (
                "Metacarpal",
                "Knuckle",
                "IntermediateBase",
                "IntermediateTip",
                "Tip",
            )
        )
        joint_names = [root_name, *finger_names]
        missing_transforms = [name for name in joint_names if name not in transform_group]
        if missing_transforms:
            raise ValueError(f"{side} 缺少 transform: {missing_transforms}")
        missing_confidence = [name for name in joint_names if name not in confidence_group]
        if missing_confidence:
            raise ValueError(f"{side} 缺少 confidence: {missing_confidence}")
        transforms = np.stack(
            [np.asarray(transform_group[name], dtype=np.float64) for name in joint_names],
            axis=1,
        )
        confidences = np.stack(
            [np.asarray(confidence_group[name], dtype=np.float64) for name in joint_names],
            axis=1,
        )
        if transforms.shape[0] != frame_count or confidences.shape[0] != frame_count:
            raise ValueError(
                f"{side} 帧数不一致: transforms={transforms.shape[0]} "
                f"confidences={confidences.shape[0]} camera={frame_count}"
            )
        valid = np.isfinite(transforms).all(axis=(1, 2, 3)) & (
            confidences[:, 0] >= float(threshold)
        )
        return HandTrack(
            side=side,
            joint_names=joint_names,
            transforms=transforms,
            confidences=confidences,
            valid=valid,
            local_origin="wrist_root_transform; non-MANO source joint hierarchy",
            source_model="EgoDex joint SE(3)",
            confidence_threshold=float(threshold),
        )

    def load_episode(
        self,
        dataset: Path,
        episode: Union[int, str],
        confidence_threshold: float = 0.5,
    ) -> CanonicalEpisode:
        try:
            import h5py
        except ImportError as error:
            raise RuntimeError(
                "EgoDex adapter 需要 h5py；安装 `python -m pip install -e '.[egodex]'`"
            ) from error
        try:
            import av
        except ImportError as error:
            raise RuntimeError("EgoDex adapter 需要 PyAV") from error

        hdf5_path, video_path, episode_id = self._resolve_episode(dataset, episode)
        with h5py.File(hdf5_path, "r") as handle:
            required = (
                "camera/intrinsic",
                "transforms/camera",
                "transforms/leftHand",
                "transforms/rightHand",
                "confidences/leftHand",
                "confidences/rightHand",
            )
            missing = [name for name in required if name not in handle]
            if missing:
                raise ValueError(f"EgoDex HDF5 缺少字段: {missing}")
            intrinsic = np.asarray(handle["camera/intrinsic"], dtype=np.float64)
            camera = np.asarray(handle["transforms/camera"], dtype=np.float64)
            frame_count = int(camera.shape[0])
            hands = {
                side: self._hand_track(handle, side, frame_count, confidence_threshold)
                for side in ("left", "right")
            }
            attrs = {str(key): _json_safe(value) for key, value in handle.attrs.items()}

        with av.open(str(video_path)) as container:
            if not container.streams.video:
                raise ValueError(f"MP4 没有视频流: {video_path}")
            stream = container.streams.video[0]
            if not stream.average_rate:
                raise ValueError(f"MP4 缺少 FPS: {video_path}")
            fps = float(stream.average_rate)
            reported_frames = int(stream.frames or 0)
            if reported_frames and reported_frames != frame_count:
                raise ValueError(
                    f"HDF5 frames={frame_count}，MP4 reported_frames={reported_frames}"
                )
            video = VideoReference(
                path=video_path,
                fps=fps,
                frame_count=frame_count,
                width=int(stream.codec_context.width),
                height=int(stream.codec_context.height),
                codec=str(stream.codec_context.name),
                pix_fmt=str(stream.codec_context.pix_fmt or "") or None,
                audio_streams=len(container.streams.audio),
            )

        labels = {
            key: attrs.get(key)
            for key in (
                "task",
                "llm_description",
                "llm_description2",
                "llm_objects",
                "llm_verbs",
                "llm_type",
            )
            if attrs.get(key) is not None
        }
        metadata = {
            key: value
            for key, value in attrs.items()
            if key not in labels
        }
        canonical = CanonicalEpisode(
            episode_id=episode_id,
            source_format=self.name,
            timestamps=np.arange(frame_count, dtype=np.float64) / fps,
            video=video,
            capabilities=self.capabilities(),
            camera_intrinsics=intrinsic,
            camera_transforms=camera,
            hands=hands,
            labels=labels,
            metadata=metadata,
            provenance={
                "adapter": self.name,
                "readonly": True,
                "hdf5_path": str(hdf5_path),
                "video_path": str(video_path),
                "confidence_threshold": float(confidence_threshold),
                "timestamp_source": "derived_from_frame_index_and_video_fps",
                "camera_transform_semantics": "vendor_transform; direction_requires_owner_confirmation",
                "mano_parameters": "not_present",
                "video_reported_frames": reported_frames,
            },
        )
        canonical.validate()
        return canonical


class RekaDailyRawAdapter:
    """Metadata-first adapter for a RekaDaily raw WebDataset snapshot."""

    name = "rekadaily_raw"
    required_columns = {
        "video_id", "project", "duration_s", "fps", "width", "height",
        "num_frames", "codec", "file_size_bytes", "src_ext", "collector",
    }

    @staticmethod
    def capabilities() -> CapabilityManifest:
        return CapabilityManifest(video=True, coarse_activity_labels=True)

    @classmethod
    def _index_path(cls, dataset: Path) -> Path:
        path = dataset.expanduser().resolve() / "metadata" / "index.parquet"
        if not path.exists():
            raise FileNotFoundError(path)
        missing = sorted(cls.required_columns - set(pq.ParquetFile(path).schema_arrow.names))
        if missing:
            raise ValueError(f"RekaDaily metadata 缺少字段: {missing}")
        return path

    @staticmethod
    def _project_counts(projects: pa.ChunkedArray) -> List[Dict[str, Any]]:
        values = pc.value_counts(projects).to_pylist()
        return [
            {"project": item["values"], "videos": int(item["counts"])}
            for item in sorted(values, key=lambda item: item["counts"], reverse=True)
        ]

    def summarize_index(self, dataset: Path) -> Dict[str, Any]:
        table = pq.read_table(
            self._index_path(dataset),
            columns=[
                "video_id", "project", "duration_s", "fps", "width", "height",
                "file_size_bytes", "src_ext",
            ],
        )
        fps_ok = pc.fill_null(pc.greater_equal(table["fps"], 29.9), False)
        short_edge = pc.min_element_wise(table["width"], table["height"])
        resolution_ok = pc.fill_null(pc.greater_equal(short_edge, 720), False)
        src_ext = pc.utf8_lower(table["src_ext"])
        expensive_stage_candidate = pc.and_(fps_ok, resolution_ok)
        return {
            "dataset": str(dataset.expanduser().resolve()),
            "detected_adapter": self.name,
            "compatible": True,
            "scope": "metadata_only; no tar shard opened",
            "videos": int(table.num_rows),
            "unique_video_ids": int(pc.count_distinct(table["video_id"]).as_py()),
            "duration_hours": float(pc.sum(table["duration_s"]).as_py() / 3600.0),
            "logical_video_tb": float(pc.sum(table["file_size_bytes"]).as_py() / 1e12),
            "screening_counts": {
                "fps_below_29_9": int(pc.sum(pc.invert(fps_ok)).as_py()),
                "short_edge_below_720": int(pc.sum(pc.invert(resolution_ok)).as_py()),
                "mov_container": int(pc.sum(pc.equal(src_ext, "mov")).as_py()),
                "metadata_candidates_for_expensive_stage": int(
                    pc.sum(expensive_stage_candidate).as_py()
                ),
            },
            "metadata_candidate_ratio": float(pc.mean(expensive_stage_candidate).as_py()),
            "projects": self._project_counts(table["project"]),
            "capabilities": self.capabilities().to_dict(),
            "unavailable_acceptance_metrics": self.unavailable_metrics(),
        }

    @staticmethod
    def unavailable_metrics() -> List[str]:
        return [
            "hand_visibility", "MANO", "MPJPE", "ODSR", "ATE", "Et",
            "2D_reprojection", "hand_motion_jitter",
        ]

    def _row(self, dataset: Path, video_id: str) -> Dict[str, Any]:
        table = pq.read_table(
            self._index_path(dataset), filters=[("video_id", "=", video_id)]
        )
        rows = table.to_pylist()
        if len(rows) != 1:
            raise ValueError(f"video_id={video_id} 应匹配 1 行，实际 {len(rows)}")
        return _json_safe(rows[0])

    @staticmethod
    def _loose_video(dataset: Path, row: Dict[str, Any]) -> Optional[Path]:
        video_id = str(row["video_id"])
        extension = str(row.get("src_ext") or "").lower()
        candidates: List[Path] = []
        if extension:
            candidates.extend((dataset / "sample").glob(f"*/{video_id}.{extension}"))
            candidates.append(dataset / f"{video_id}.{extension}")
        candidates.extend((dataset / "sample").glob(f"*/{video_id}.*"))
        return next((path.resolve() for path in candidates if path.exists()), None)

    @staticmethod
    def _tar_video(dataset: Path, row: Dict[str, Any]) -> Optional[Tuple[Path, str]]:
        """Locate a video in an uncompressed WebDataset shard without extracting it."""

        project = str(row.get("project") or "")
        video_id = str(row["video_id"])
        extension = str(row.get("src_ext") or "").lower()
        if not project or not extension:
            return None
        expected = f"{video_id}.{extension}"
        shard_root = dataset / "data" / project
        for shard in sorted(shard_root.glob("*.tar")):
            try:
                with tarfile.open(shard, "r") as archive:
                    try:
                        member = archive.getmember(expected)
                    except KeyError:
                        member = next(
                            (
                                candidate for candidate in archive.getmembers()
                                if candidate.isfile()
                                and PurePosixPath(candidate.name).name == expected
                            ),
                            None,
                        )
                    if member is not None:
                        return shard.resolve(), member.name
            except (tarfile.TarError, OSError):
                continue
        return None

    def inspect_video(
        self,
        dataset: Path,
        video_id: str,
        mode: str = "header",
        video_options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        dataset = dataset.expanduser().resolve()
        row = self._row(dataset, video_id)
        video_path = self._loose_video(dataset, row)
        tar_video = None if video_path else self._tar_video(dataset, row)
        source_uri = (
            str(video_path) if video_path
            else f"tar://{tar_video[0]}!/{tar_video[1]}" if tar_video
            else None
        )
        report: Dict[str, Any] = {
            "dataset": str(dataset),
            "detected_adapter": self.name,
            "compatible": True,
            "video_id": video_id,
            "metadata": row,
            "capabilities": self.capabilities().to_dict(),
            "video_path": str(video_path) if video_path else None,
            "video_uri": source_uri,
            "source_access": (
                "loose_sample" if video_path
                else "webdataset_tar_member" if tar_video
                else "webdataset_shard_not_found"
            ),
            "unavailable_acceptance_metrics": self.unavailable_metrics(),
        }
        if video_path is None and tar_video is None:
            report["video_probe"] = None
            report["issues"] = [{
                "code": "raw_video_not_materialized",
                "severity": "info",
                "message": "索引存在，但 loose sample 未下载；未打开大 tar shard",
            }]
            return report
        if video_path is not None:
            probed, issues = probe_video(video_path, mode, video_options)
            issue_source = str(video_path)
        else:
            shard_path, member_name = tar_video
            issue_source = f"tar://{shard_path}!/{member_name}"
            with tarfile.open(shard_path, "r") as archive:
                member_stream = archive.extractfile(member_name)
                if member_stream is None:
                    raise ValueError(f"tar 成员不是普通文件: {issue_source}")
                probed, issues = probe_video(
                    member_stream, mode, video_options, source_name=issue_source
                )
        cheap_acceptance = {
            "fps_at_least_30": float(row["fps"]) >= 29.9,
            "resolution_at_least_720p": min(int(row["width"]), int(row["height"])) >= 720,
            "container_mp4_or_avi": str(row["src_ext"]).lower() in {"mp4", "avi"},
        }
        if not cheap_acceptance["fps_at_least_30"]:
            issues.append(Issue(
                "video_fps_below_requirement", "error",
                f"FPS={float(row['fps']):.3f}，低于 30 FPS（29.9 容差）",
                file=issue_source,
            ))
        if not cheap_acceptance["resolution_at_least_720p"]:
            issues.append(Issue(
                "video_resolution_below_requirement", "error",
                f"分辨率 {int(row['width'])}x{int(row['height'])} 低于 720p",
                file=issue_source,
            ))
        if not cheap_acceptance["container_mp4_or_avi"]:
            issues.append(Issue(
                "video_container_requires_conversion", "warning",
                f"容器 .{row['src_ext']} 需要转换为验收格式 MP4/AVI",
                file=issue_source,
            ))
        hard_failure = any(issue.severity == "error" for issue in issues)
        review_signal = any(issue.severity == "warning" for issue in issues)
        report.update({
            "video_probe": probed,
            "metadata_comparison": {
                "width_match": probed.get("width") == int(row["width"]),
                "height_match": probed.get("height") == int(row["height"]),
                "fps_delta": (
                    abs(float(probed["average_rate"]) - float(row["fps"]))
                    if probed.get("average_rate") is not None else None
                ),
                "reported_frame_delta": (
                    int(probed["reported_frames"]) - int(row["num_frames"])
                    if probed.get("reported_frames") else None
                ),
            },
            "cheap_acceptance": cheap_acceptance,
            "screening_decision": (
                "screen_out_before_hand_annotation" if hard_failure
                else "review_before_hand_annotation" if review_signal
                else "candidate_for_hand_annotation"
            ),
            "issues": [issue.to_dict() for issue in issues],
        })
        return report


def inspect_adapter(
    dataset: Path,
    episode: Optional[Union[int, str]],
    confidence_threshold: float = 0.5,
    video_check: str = "header",
    video_options: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    adapter_name = detect_adapter(dataset)
    if adapter_name == "raw_video":
        return inspect_generic_ego_video(
            dataset,
            mode=video_check,
            video_options=video_options,
        )
    if adapter_name == "rekadaily_raw":
        adapter = RekaDailyRawAdapter()
        if episode is None:
            return adapter.summarize_index(dataset)
        return adapter.inspect_video(dataset, str(episode), video_check, video_options)
    if adapter_name in {"egodex_hdf5", "egodex_collection"}:
        if episode is None:
            raise ValueError("EgoDex inspect-adapter 需要 --episode")
        canonical = EgoDexHDF5Adapter().load_episode(
            dataset, episode, confidence_threshold=confidence_threshold
        )
        return {
            "dataset": str(dataset.expanduser().resolve()),
            "detected_adapter": "egodex_hdf5",
            "compatible": True,
            "canonical": canonical.summary(),
        }
    if adapter_name != "mano_hamer":
        return {
            "dataset": str(dataset.expanduser().resolve()),
            "detected_adapter": adapter_name,
            "compatible": adapter_name == "standard_v3",
            "capabilities": (
                CapabilityManifest(
                    video=True,
                    camera_intrinsics=True,
                    camera_trajectory=True,
                    mano_parameters=True,
                    task_labels=True,
                    independent_timestamps=True,
                ).to_dict()
                if adapter_name == "standard_v3" else {}
            ),
        }
    numeric_episode = int(episode)
    adapted = ManoHamerAdapter().load_episode(dataset, numeric_episode)
    first = adapted.records[0].to_dict() if adapted.records else {}
    return {
        "dataset": str(dataset.expanduser().resolve()),
        "episode_index": numeric_episode,
        "detected_adapter": adapter_name,
        "compatible": True,
        "records": len(adapted.records),
        "route": {key: str(value) if isinstance(value, Path) else value for key, value in adapted.route.items()},
        "normalized_shapes": {
            "observation.state": len(first.get("observation.state", [])),
            "state_mask": len(first.get("state_mask", [])),
            "left_hand_pose": len(first.get("left_hand_pose", [])),
            "right_hand_pose": len(first.get("right_hand_pose", [])),
            "extrinsics_w2c": len(first.get("extrinsics_w2c", [])),
        },
        "provenance": adapted.provenance,
        "capabilities": ManoHamerAdapter.capabilities().to_dict(),
    }
