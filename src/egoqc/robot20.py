from __future__ import annotations

import hashlib
import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _vector(element: Optional[ET.Element], attribute: str, default: str) -> List[float]:
    value = element.get(attribute, default) if element is not None else default
    return [float(part) for part in value.split()]


def inspect_urdf(path: Path, expected_dof: Optional[int] = None) -> Dict[str, Any]:
    path = path.expanduser().resolve()
    root = ET.parse(path).getroot()
    if root.tag != "robot":
        raise ValueError(f"URDF root 必须是 <robot>: {path}")
    links = [element.get("name", "") for element in root.findall("link")]
    link_set = set(links)
    errors: List[str] = []
    warnings: List[str] = []
    if len(link_set) != len(links):
        errors.append("link name 不唯一")

    joints: List[Dict[str, Any]] = []
    child_links = set()
    for element in root.findall("joint"):
        name = element.get("name", "")
        joint_type = element.get("type", "")
        parent_element = element.find("parent")
        child_element = element.find("child")
        parent = parent_element.get("link", "") if parent_element is not None else ""
        child = child_element.get("link", "") if child_element is not None else ""
        axis = _vector(element.find("axis"), "xyz", "1 0 0")
        origin_element = element.find("origin")
        origin_xyz = _vector(origin_element, "xyz", "0 0 0")
        origin_rpy = _vector(origin_element, "rpy", "0 0 0")
        limit_element = element.find("limit")
        limit = None
        if limit_element is not None:
            limit = {
                key: float(limit_element.get(key))
                for key in ("lower", "upper", "effort", "velocity")
                if limit_element.get(key) is not None
            }
        if not name:
            errors.append("存在空 joint name")
        if parent not in link_set or child not in link_set:
            errors.append(f"{name}: parent/child link 不存在")
        if child in child_links:
            errors.append(f"{name}: child link {child} 有多个 parent")
        child_links.add(child)
        if joint_type in {"revolute", "continuous", "prismatic"}:
            norm = math.sqrt(sum(value * value for value in axis))
            if not math.isfinite(norm) or abs(norm - 1.0) > 1e-5:
                errors.append(f"{name}: axis 非单位向量 {axis}")
        if joint_type == "revolute":
            if limit is None or "lower" not in limit or "upper" not in limit:
                errors.append(f"{name}: revolute joint 缺少 lower/upper")
            elif limit["lower"] >= limit["upper"]:
                errors.append(f"{name}: lower >= upper")
        joints.append(
            {
                "name": name,
                "type": joint_type,
                "parent": parent,
                "child": child,
                "axis": axis,
                "origin_xyz": origin_xyz,
                "origin_rpy": origin_rpy,
                "limit": limit,
            }
        )

    movable = [joint for joint in joints if joint["type"] != "fixed"]
    fixed = [joint for joint in joints if joint["type"] == "fixed"]
    if expected_dof is not None and len(movable) != expected_dof:
        errors.append(f"movable joint={len(movable)}，期望 {expected_dof}")
    joint_names = [joint["name"] for joint in joints]
    if len(set(joint_names)) != len(joint_names):
        errors.append("joint name 不唯一")
    roots = sorted(link_set - child_links)
    if len(roots) != 1:
        errors.append(f"URDF 应只有一个 root link，实际 {roots}")

    mesh_records = []
    for mesh in root.findall(".//mesh"):
        filename = mesh.get("filename", "")
        if filename.startswith("package://"):
            resolved = None
            exists = False
        else:
            candidate = (path.parent / filename).resolve()
            resolved = str(candidate)
            exists = candidate.exists()
        record = {"filename": filename, "resolved": resolved, "exists": exists}
        if exists and resolved:
            mesh_path = Path(resolved)
            record["size_bytes"] = mesh_path.stat().st_size
            record["sha256"] = _sha256(mesh_path)
        mesh_records.append(record)
    missing_meshes = sorted({record["filename"] for record in mesh_records if not record["exists"]})
    unique_meshes = {
        record["resolved"]: record
        for record in mesh_records
        if record["exists"] and record["resolved"]
    }
    if missing_meshes:
        warnings.append(f"缺少 {len(missing_meshes)} 个 mesh 资源；FK 可用，渲染/碰撞不可用")

    return {
        "path": str(path),
        "sha256": _sha256(path),
        "robot_name": root.get("name"),
        "root_links": roots,
        "link_count": len(links),
        "joint_count": len(joints),
        "movable_joint_count": len(movable),
        "fixed_joint_count": len(fixed),
        "mimic_joint_count": len(root.findall(".//mimic")),
        "joint_order": [joint["name"] for joint in movable],
        "joints": joints,
        "mesh_reference_count": len(mesh_records),
        "mesh_file_count": len(unique_meshes),
        "mesh_total_bytes": sum(record.get("size_bytes", 0) for record in unique_meshes.values()),
        "meshes": mesh_records,
        "missing_meshes": missing_meshes,
        "errors": errors,
        "warnings": warnings,
        "ok": not errors,
    }


def inspect_robot_pair(left: Path, right: Path, expected_dof: int = 20) -> Dict[str, Any]:
    left_report = inspect_urdf(left, expected_dof)
    right_report = inspect_urdf(right, expected_dof)
    errors: List[str] = []
    warnings: List[str] = []
    if len(left_report["joints"]) != len(right_report["joints"]):
        errors.append("左右手 joint 数量不同")
    else:
        for index, (left_joint, right_joint) in enumerate(
            zip(left_report["joints"], right_report["joints"])
        ):
            if left_joint["type"] != right_joint["type"]:
                errors.append(f"joint[{index}] 左右类型不同")
            if left_joint["limit"] != right_joint["limit"]:
                errors.append(f"joint[{index}] 左右 limit 不同")
            left_axis = [abs(value) for value in left_joint["axis"]]
            right_axis = [abs(value) for value in right_joint["axis"]]
            if any(abs(a - b) > 1e-6 for a, b in zip(left_axis, right_axis)):
                errors.append(f"joint[{index}] 左右 axis 绝对方向不同")
    if left_report["missing_meshes"] or right_report["missing_meshes"]:
        warnings.append("URDF zip 未包含 meshes")
    errors.extend(f"left: {value}" for value in left_report["errors"])
    errors.extend(f"right: {value}" for value in right_report["errors"])
    return {
        "expected_dof_per_hand": expected_dof,
        "left": left_report,
        "right": right_report,
        "pair_errors": errors,
        "pair_warnings": warnings,
        "ok": not errors,
    }
