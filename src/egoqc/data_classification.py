from __future__ import annotations

from typing import Any, Dict


def classify_capability_profile(capabilities: Dict[str, Any]) -> str:
    video = bool(capabilities.get("video"))
    text = bool(capabilities.get("task_labels") or capabilities.get("coarse_activity_labels"))
    mano = bool(capabilities.get("mano_parameters"))
    calibrated = bool(
        capabilities.get("camera_intrinsics")
        and capabilities.get("camera_trajectory")
    )
    robot = bool(capabilities.get("robot_action") and capabilities.get("robot_state"))
    if video and mano and calibrated and robot:
        return "rgb_calibrated_mano_robot_action"
    if video and mano and calibrated:
        return "rgb_calibrated_mano"
    if video and mano:
        return "rgb_mano"
    if video and calibrated:
        return "rgb_calibrated"
    if video and text:
        return "rgb_text"
    if video:
        return "rgb_only"
    return "metadata_only"


def annotation_provenance(
    *,
    task_present: bool,
    source_mano_present: bool = False,
    source_mano_ground_truth: bool = False,
    derived_hand_screen_present: bool = False,
    derived_mano_present: bool = False,
    alignment_human_approved: bool = False,
) -> Dict[str, Any]:
    if source_mano_present:
        mano_status = (
            "source_ground_truth"
            if source_mano_ground_truth
            else "source_annotation_unverified"
        )
    elif derived_mano_present and alignment_human_approved:
        mano_status = "derived_silver_prediction_human_approved"
    elif derived_mano_present:
        mano_status = "derived_model_prediction_unapproved"
    else:
        mano_status = "missing"
    return {
        "schema_version": "egoqc-annotation-provenance-v1",
        "rgb": "source_observation",
        "task_text": "source_metadata" if task_present else "missing",
        "hand_screen": (
            "derived_model_prediction" if derived_hand_screen_present else "missing"
        ),
        "mano": mano_status,
        "mano_is_ground_truth": mano_status == "source_ground_truth",
        "mano_training_eligible": mano_status in {
            "source_ground_truth",
            "derived_silver_prediction_human_approved",
        },
    }
