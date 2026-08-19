from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass(frozen=True)
class CapabilityManifest:
    """Declares measurable inputs without pretending missing ground truth exists."""

    video: bool = False
    coarse_activity_labels: bool = False
    camera_intrinsics: bool = False
    camera_trajectory: bool = False
    hand_joint_transforms: bool = False
    mano_parameters: bool = False
    prediction_confidence: bool = False
    task_labels: bool = False
    subtask_labels: bool = False
    tactile: bool = False
    hand_ground_truth: bool = False
    trajectory_ground_truth: bool = False
    independent_timestamps: bool = False
    video_timestamps: bool = False
    camera_distortion: bool = False
    hand_2d_keypoints: bool = False
    hand_bounding_boxes: bool = False
    depth: bool = False
    imu: bool = False
    audio: bool = False
    multiple_cameras: bool = False
    stereo: bool = False
    camera_extrinsics: bool = False
    gaze: bool = False
    robot_state: bool = False
    robot_action: bool = False
    glove_pose: bool = False
    privacy_annotations: bool = False
    object_annotations: bool = False

    def to_dict(self) -> Dict[str, bool]:
        return {
            name: bool(value)
            for name, value in self.__dict__.items()
        }

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "CapabilityManifest":
        known = cls.__dataclass_fields__
        return cls(**{name: bool(raw) for name, raw in value.items() if name in known})


def route_capabilities(
    capabilities: CapabilityManifest,
    *,
    has_mano_overlay: bool = False,
) -> Dict[str, Any]:
    """Route a source by evidence it actually has, not by dataset brand.

    Missing optional modalities disable only dependent metrics. They never make an
    otherwise readable raw ego video invalid.
    """

    enabled = {
        "video_header_qc": capabilities.video,
        "sparse_visual_qc": capabilities.video,
        "temporal_visual_qc": capabilities.video,
        "hand_observability_model": capabilities.video,
        "task_semantic_qc": capabilities.video and capabilities.task_labels,
        "subtask_boundary_qc": capabilities.video and capabilities.subtask_labels,
        "camera_trajectory_qc": capabilities.camera_trajectory,
        "mano_numeric_qc": capabilities.mano_parameters,
        "mano_visual_reprojection_qc": capabilities.video and (
            has_mano_overlay
            or (capabilities.mano_parameters and capabilities.camera_intrinsics)
        ),
        "hand_joint_qc": capabilities.hand_joint_transforms,
        "tactile_qc": capabilities.tactile,
        "sensor_sync_qc": capabilities.independent_timestamps,
        "depth_qc": capabilities.depth,
        "imu_qc": capabilities.imu,
        "audio_qc": capabilities.audio,
        "gaze_qc": capabilities.gaze,
        "multiview_sync_qc": capabilities.multiple_cameras,
        "robot_action_sync_qc": capabilities.robot_action,
        "glove_sync_qc": capabilities.glove_pose,
        "privacy_visual_screening": capabilities.video,
        "ego_viewpoint_screening": capabilities.video,
        "near_duplicate_detection": capabilities.video,
    }
    objectives = {
        "video_representation": capabilities.video,
        "temporal_prediction": capabilities.video,
        "video_text_alignment": capabilities.video and (
            capabilities.task_labels or capabilities.coarse_activity_labels
        ),
        "hand_presence_auxiliary": capabilities.video and (
            capabilities.hand_2d_keypoints
            or capabilities.hand_bounding_boxes
            or capabilities.prediction_confidence
        ),
        "mano_motion": capabilities.mano_parameters,
        "camera_pose": capabilities.camera_trajectory,
        "tactile": capabilities.tactile,
        "robot_action": capabilities.robot_action,
        "robot_state": capabilities.robot_state,
        "multiview_consistency": capabilities.multiple_cameras,
        "gaze_conditioning": capabilities.gaze,
    }
    metric_requirements = {
        "MPJPE": capabilities.hand_ground_truth and (
            capabilities.hand_joint_transforms or capabilities.mano_parameters
        ),
        "ATE": capabilities.camera_trajectory and capabilities.trajectory_ground_truth,
        "MANO_reprojection": enabled["mano_visual_reprojection_qc"],
        "hand_motion_jitter": capabilities.hand_joint_transforms or capabilities.mano_parameters,
        "camera_motion_jitter": capabilities.camera_trajectory,
        "sensor_drift_Et": capabilities.independent_timestamps,
        "task_label_match": enabled["task_semantic_qc"],
        "subtask_boundary": enabled["subtask_boundary_qc"],
        "multiview_calibration": capabilities.multiple_cameras
        and capabilities.camera_intrinsics
        and capabilities.camera_extrinsics,
        "robot_action_sync": capabilities.robot_action
        and capabilities.independent_timestamps,
        "glove_video_sync": capabilities.glove_pose
        and capabilities.independent_timestamps,
    }
    completion = []
    if capabilities.video and not capabilities.video_timestamps:
        completion.append({
            "field": "timestamps",
            "method": "derive_from_frame_index_and_container_fps",
            "authority": "deterministic_if_constant_frame_rate",
        })
    if capabilities.video and not capabilities.camera_intrinsics:
        completion.append({
            "field": "camera_intrinsics",
            "method": "estimate_or_import_calibration",
            "authority": "silver_never_ground_truth",
        })
    if capabilities.video and not capabilities.hand_2d_keypoints:
        completion.append({
            "field": "hand_2d_keypoints_or_boxes",
            "method": "offline_hand_detector",
            "authority": "silver_model_prediction",
        })
    if capabilities.video and not capabilities.camera_trajectory:
        completion.append({
            "field": "camera_trajectory",
            "method": "vslam_when_scene_is_observable",
            "authority": "silver_model_prediction",
        })
    if capabilities.video and not capabilities.mano_parameters:
        completion.append({
            "field": "mano_parameters",
            "method": "offline_hand_reconstruction",
            "authority": "silver_model_prediction",
        })
    return {
        "enabled_stages": enabled,
        "training_objectives": objectives,
        "available_metrics": [name for name, available in metric_requirements.items() if available],
        "unavailable_metrics": [name for name, available in metric_requirements.items() if not available],
        "completion_candidates": completion,
        "missing_optional_modalities_are_failures": False,
    }


def plan_use_cases(capabilities: CapabilityManifest) -> Dict[str, Any]:
    """Describe what a partially observed ego source can safely be used for."""

    values = capabilities.to_dict()
    values["text_labels"] = bool(
        capabilities.task_labels or capabilities.coarse_activity_labels
    )
    definitions = {
        "video_self_supervised_pretraining": {
            "required": ("video",),
            "recommended": ("video_timestamps",),
        },
        "video_language_pretraining": {
            "required": ("video", "text_labels"),
            "recommended": ("task_labels",),
        },
        "visual_qc_model_training": {
            "required": ("video",),
            "recommended": ("task_labels", "hand_bounding_boxes", "video_timestamps"),
        },
        "hand_reconstruction_inference": {
            "required": ("video",),
            "recommended": ("camera_intrinsics", "camera_distortion"),
        },
        "hand_reconstruction_supervised_training": {
            "required": ("video", "hand_ground_truth"),
            "recommended": ("camera_intrinsics", "mano_parameters", "independent_timestamps"),
        },
        "vla_observation_pretraining": {
            "required": ("video", "task_labels"),
            "recommended": ("hand_2d_keypoints", "camera_intrinsics", "subtask_labels"),
        },
        "robot_imitation_learning": {
            "required": ("video", "robot_action", "independent_timestamps"),
            "recommended": ("robot_state", "task_labels", "camera_intrinsics"),
        },
        "multiview_representation_learning": {
            "required": ("video", "multiple_cameras"),
            "recommended": ("camera_intrinsics", "camera_extrinsics", "independent_timestamps"),
        },
        "stereo_depth_learning": {
            "required": ("video", "stereo", "camera_intrinsics", "camera_extrinsics"),
            "recommended": ("depth", "independent_timestamps"),
        },
        "teleoperation_glove_learning": {
            "required": ("video", "glove_pose", "independent_timestamps"),
            "recommended": ("tactile", "task_labels", "robot_action"),
        },
        "gaze_conditioned_interaction": {
            "required": ("video", "gaze", "independent_timestamps"),
            "recommended": ("task_labels", "hand_2d_keypoints"),
        },
        "camera_motion_or_slam_learning": {
            "required": ("video", "camera_trajectory"),
            "recommended": ("camera_intrinsics", "imu", "trajectory_ground_truth"),
        },
        "public_release_after_privacy_review": {
            "required": ("video", "privacy_annotations"),
            "recommended": (),
        },
        "supplier_acceptance": {
            "required": ("video",),
            "recommended": (
                "task_labels", "camera_intrinsics", "camera_trajectory",
                "mano_parameters", "independent_timestamps",
            ),
        },
    }
    result = {}
    for name, definition in definitions.items():
        missing_required = [key for key in definition["required"] if not values.get(key, False)]
        missing_recommended = [
            key for key in definition["recommended"] if not values.get(key, False)
        ]
        status = "blocked" if missing_required else "partial" if missing_recommended else "ready"
        result[name] = {
            "status": status,
            "missing_required": missing_required,
            "missing_recommended": missing_recommended,
            "ground_truth_policy": (
                "predicted_silver_modalities_may_train_auxiliary_heads_but_may_not_score_accuracy"
            ),
        }
    return result


@dataclass
class VideoReference:
    path: Path
    fps: float
    frame_count: int
    width: int
    height: int
    codec: str
    pix_fmt: Optional[str] = None
    audio_streams: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": str(self.path),
            "fps": self.fps,
            "frame_count": self.frame_count,
            "width": self.width,
            "height": self.height,
            "codec": self.codec,
            "pix_fmt": self.pix_fmt,
            "audio_streams": self.audio_streams,
        }


@dataclass
class HandTrack:
    side: str
    joint_names: List[str]
    transforms: np.ndarray
    confidences: np.ndarray
    valid: np.ndarray
    local_origin: str
    source_model: str
    confidence_threshold: Optional[float] = None

    def validate(self, frame_count: int) -> None:
        expected = (frame_count, len(self.joint_names), 4, 4)
        if self.transforms.shape != expected:
            raise ValueError(
                f"{self.side} transforms shape={self.transforms.shape}，期望 {expected}"
            )
        if self.confidences.shape != (frame_count, len(self.joint_names)):
            raise ValueError(
                f"{self.side} confidences shape={self.confidences.shape}"
            )
        if self.valid.shape != (frame_count,):
            raise ValueError(f"{self.side} valid shape={self.valid.shape}")

    def summary(self) -> Dict[str, Any]:
        finite = np.isfinite(self.transforms).all(axis=(1, 2, 3))
        root_confidence = self.confidences[:, 0] if self.confidences.size else np.array([])
        finite_confidence = self.confidences[np.isfinite(self.confidences)]
        all_joints_confident_ratio = None
        joint_values_confident_ratio = None
        if self.confidence_threshold is not None and self.confidences.size:
            all_joints_confident_ratio = float(
                np.mean(np.all(self.confidences >= self.confidence_threshold, axis=1))
            )
            joint_values_confident_ratio = float(
                np.mean(self.confidences >= self.confidence_threshold)
            )
        return {
            "side": self.side,
            "joint_count": len(self.joint_names),
            "joint_names": self.joint_names,
            "valid_frames": int(np.count_nonzero(self.valid)),
            "valid_ratio": float(np.mean(self.valid)) if len(self.valid) else 0.0,
            "valid_semantics": "root_presence_above_confidence_threshold",
            "finite_frames": int(np.count_nonzero(finite)),
            "root_confidence_mean": (
                float(np.mean(root_confidence)) if root_confidence.size else None
            ),
            "joint_confidence_mean": (
                float(np.mean(finite_confidence)) if finite_confidence.size else None
            ),
            "joint_confidence_p05": (
                float(np.quantile(finite_confidence, 0.05))
                if finite_confidence.size else None
            ),
            "confidence_threshold": self.confidence_threshold,
            "all_joints_confident_ratio": all_joints_confident_ratio,
            "joint_values_confident_ratio": joint_values_confident_ratio,
            "local_origin": self.local_origin,
            "source_model": self.source_model,
        }


@dataclass
class CanonicalEpisode:
    episode_id: str
    source_format: str
    timestamps: np.ndarray
    video: VideoReference
    capabilities: CapabilityManifest
    camera_intrinsics: Optional[np.ndarray] = None
    camera_transforms: Optional[np.ndarray] = None
    hands: Dict[str, HandTrack] = field(default_factory=dict)
    labels: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    provenance: Dict[str, Any] = field(default_factory=dict)

    @property
    def frame_count(self) -> int:
        return int(len(self.timestamps))

    def validate(self) -> None:
        if self.timestamps.shape != (self.video.frame_count,):
            raise ValueError(
                f"timestamp count={len(self.timestamps)}，video frames={self.video.frame_count}"
            )
        if self.frame_count and not np.isfinite(self.timestamps).all():
            raise ValueError("timestamps 包含 NaN/Inf")
        if self.frame_count > 1 and np.any(np.diff(self.timestamps) <= 0):
            raise ValueError("timestamps 必须严格递增")
        if self.camera_intrinsics is not None and self.camera_intrinsics.shape != (3, 3):
            raise ValueError(f"camera_intrinsics shape={self.camera_intrinsics.shape}")
        if self.camera_transforms is not None:
            expected = (self.frame_count, 4, 4)
            if self.camera_transforms.shape != expected:
                raise ValueError(
                    f"camera_transforms shape={self.camera_transforms.shape}，期望 {expected}"
                )
        for hand in self.hands.values():
            hand.validate(self.frame_count)

    def summary(self) -> Dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "source_format": self.source_format,
            "frame_count": self.frame_count,
            "duration_s": (
                float(self.timestamps[-1] + 1.0 / self.video.fps)
                if self.frame_count else 0.0
            ),
            "video": self.video.to_dict(),
            "capabilities": self.capabilities.to_dict(),
            "camera_intrinsics": (
                self.camera_intrinsics.tolist()
                if self.camera_intrinsics is not None else None
            ),
            "hands": {side: hand.summary() for side, hand in self.hands.items()},
            "labels": self.labels,
            "metadata": self.metadata,
            "provenance": self.provenance,
        }
