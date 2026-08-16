from __future__ import annotations

from typing import Any, Dict


ERROR_TAXONOMY: Dict[str, Dict[str, str]] = {
    "hand_absent": {
        "label": "手连续离画",
        "category": "hand_visibility",
        "category_label": "手部可见性",
        "severity": "reject",
    },
    "persistent_extra_hands": {
        "label": "疑似第二人手",
        "category": "multi_person",
        "category_label": "多人/干扰",
        "severity": "reject",
    },
    "camera_jitter": {
        "label": "相机频繁抖动",
        "category": "motion_quality",
        "category_label": "运动质量",
        "severity": "review",
    },
    "hand_jitter": {
        "label": "手部轨迹抖动",
        "category": "motion_quality",
        "category_label": "运动质量",
        "severity": "review",
    },
    "pose_freeze": {
        "label": "姿态异常冻结",
        "category": "annotation_geometry",
        "category_label": "标注与几何",
        "severity": "review",
    },
    "mask_flicker": {
        "label": "手部有效标记闪烁",
        "category": "annotation_geometry",
        "category_label": "标注与几何",
        "severity": "review",
    },
    "mano_invalid_so3": {
        "label": "MANO 旋转不合法",
        "category": "annotation_geometry",
        "category_label": "标注与几何",
        "severity": "reject",
    },
    "wrist_inconsistent": {
        "label": "手腕坐标不一致",
        "category": "annotation_geometry",
        "category_label": "标注与几何",
        "severity": "reject",
    },
    "mpjpe_exceeded": {
        "label": "手部 MPJPE 超标",
        "category": "annotation_geometry",
        "category_label": "标注与几何",
        "severity": "reject",
    },
    "ate_exceeded": {
        "label": "相机 ATE 超标",
        "category": "camera_trajectory",
        "category_label": "相机与 SLAM",
        "severity": "reject",
    },
    "timestamp_drift": {
        "label": "时间同步漂移",
        "category": "temporal_sync",
        "category_label": "时序与同步",
        "severity": "reject",
    },
    "frame_count_mismatch": {
        "label": "视频与标注帧数不一致",
        "category": "temporal_sync",
        "category_label": "时序与同步",
        "severity": "reject",
    },
    "video_decode_error": {
        "label": "视频无法解码",
        "category": "video_quality",
        "category_label": "视频质量",
        "severity": "reject",
    },
    "low_visual_quality": {
        "label": "模糊/曝光异常",
        "category": "video_quality",
        "category_label": "视频质量",
        "severity": "review",
    },
    "schema_error": {
        "label": "数据结构不符合标准",
        "category": "data_integrity",
        "category_label": "结构与完整性",
        "severity": "reject",
    },
}


SEVERITY_LABELS = {
    "reject": "拒收级",
    "review": "需复核",
    "warning": "提示",
}


def describe_error(kind: str) -> Dict[str, Any]:
    description = ERROR_TAXONOMY.get(kind, {})
    severity = description.get("severity", "review")
    return {
        "error_label": description.get("label", kind),
        "category": description.get("category", "unclassified"),
        "category_label": description.get("category_label", "未分类"),
        "severity": severity,
        "severity_label": SEVERITY_LABELS.get(severity, severity),
    }
