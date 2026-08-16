from __future__ import annotations

import math
import struct
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw


def _numbers(value: Optional[str], default: str = "0 0 0") -> np.ndarray:
    return np.asarray([float(part) for part in (value or default).split()], dtype=np.float64)


def _origin(element: Optional[ET.Element]) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    if element is None:
        return transform
    xyz = _numbers(element.get("xyz"))
    roll, pitch, yaw = _numbers(element.get("rpy"))
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rotation = np.asarray(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )
    transform[:3, :3] = rotation
    transform[:3, 3] = xyz
    return transform


def _axis_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    c, s = math.cos(angle), math.sin(angle)
    cross = 1.0 - c
    rotation = np.asarray(
        [
            [c + x * x * cross, x * y * cross - z * s, x * z * cross + y * s],
            [y * x * cross + z * s, c + y * y * cross, y * z * cross - x * s],
            [z * x * cross - y * s, z * y * cross + x * s, c + z * z * cross],
        ]
    )
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    return transform


def load_stl(path: Path) -> np.ndarray:
    """Load binary or ASCII STL as (triangle, vertex, xyz) float64 coordinates."""
    raw = path.read_bytes()
    if len(raw) >= 84:
        triangle_count = struct.unpack_from("<I", raw, 80)[0]
        if 84 + triangle_count * 50 == len(raw):
            records = np.frombuffer(
                raw,
                dtype=np.dtype(
                    [
                        ("normal", "<f4", (3,)),
                        ("vertices", "<f4", (3, 3)),
                        ("attribute", "<u2"),
                    ]
                ),
                count=triangle_count,
                offset=84,
            )
            return np.asarray(records["vertices"], dtype=np.float64)
    vertices: List[List[float]] = []
    for line in raw.decode("ascii", errors="ignore").splitlines():
        parts = line.strip().split()
        if len(parts) == 4 and parts[0].lower() == "vertex":
            vertices.append([float(value) for value in parts[1:]])
    if not vertices or len(vertices) % 3:
        raise ValueError(f"无法解析 STL: {path}")
    return np.asarray(vertices, dtype=np.float64).reshape(-1, 3, 3)


def _resolve_mesh(urdf_path: Path, filename: str) -> Path:
    if filename.startswith("package://"):
        raise ValueError(f"轻量渲染器暂不支持 package URI: {filename}")
    return (urdf_path.parent / filename).resolve()


def load_robot(urdf_path: Path) -> Dict[str, Any]:
    urdf_path = urdf_path.expanduser().resolve()
    root = ET.parse(urdf_path).getroot()
    visuals: Dict[str, List[Dict[str, Any]]] = {}
    links = [element.get("name", "") for element in root.findall("link")]
    for link in root.findall("link"):
        records = []
        for visual in link.findall("visual"):
            mesh = visual.find("geometry/mesh")
            if mesh is None or not mesh.get("filename"):
                continue
            scale = _numbers(mesh.get("scale"), "1 1 1")
            color_element = visual.find("material/color")
            rgba = _numbers(
                color_element.get("rgba") if color_element is not None else None,
                "0.63 0.67 0.70 1",
            )
            records.append(
                {
                    "path": _resolve_mesh(urdf_path, mesh.get("filename", "")),
                    "origin": _origin(visual.find("origin")),
                    "scale": scale,
                    "color": tuple(int(np.clip(value, 0, 1) * 255) for value in rgba[:3]),
                }
            )
        visuals[link.get("name", "")] = records

    joints = []
    children = set()
    movable_order = []
    limits = []
    for joint in root.findall("joint"):
        joint_type = joint.get("type", "fixed")
        name = joint.get("name", "")
        parent = joint.find("parent").get("link", "")
        child = joint.find("child").get("link", "")
        children.add(child)
        movable_index = None
        if joint_type != "fixed":
            movable_index = len(movable_order)
            movable_order.append(name)
            limit = joint.find("limit")
            limits.append(
                (
                    float(limit.get("lower", "-3.141592653589793")),
                    float(limit.get("upper", "3.141592653589793")),
                )
            )
        joints.append(
            {
                "name": name,
                "type": joint_type,
                "parent": parent,
                "child": child,
                "origin": _origin(joint.find("origin")),
                "axis": _numbers(joint.find("axis").get("xyz") if joint.find("axis") is not None else None, "1 0 0"),
                "movable_index": movable_index,
            }
        )
    roots = sorted(set(links) - children)
    if len(roots) != 1:
        raise ValueError(f"URDF 必须有且仅有一个 root link，实际为 {roots}")
    return {
        "path": urdf_path,
        "name": root.get("name", urdf_path.stem),
        "root_link": roots[0],
        "links": links,
        "joints": joints,
        "visuals": visuals,
        "joint_order": movable_order,
        "limits": limits,
    }


def pose_from_preset(robot: Dict[str, Any], preset: str) -> np.ndarray:
    limits = np.asarray(robot["limits"], dtype=np.float64)
    if preset == "neutral":
        return np.clip(np.zeros(len(limits)), limits[:, 0], limits[:, 1])
    if preset == "midrange":
        return limits[:, 0] + 0.5 * (limits[:, 1] - limits[:, 0])
    raise ValueError(f"未知 pose preset: {preset}")


def forward_kinematics(robot: Dict[str, Any], q: Sequence[float]) -> Dict[str, np.ndarray]:
    q_array = np.asarray(q, dtype=np.float64)
    if q_array.shape != (len(robot["joint_order"]),):
        raise ValueError(f"q 需要 {len(robot['joint_order'])} 维，实际 {q_array.shape}")
    transforms = {robot["root_link"]: np.eye(4, dtype=np.float64)}
    pending = list(robot["joints"])
    while pending:
        progressed = False
        for joint in pending[:]:
            if joint["parent"] not in transforms:
                continue
            transform = transforms[joint["parent"]] @ joint["origin"]
            if joint["type"] in {"revolute", "continuous"}:
                transform = transform @ _axis_rotation(
                    joint["axis"], float(q_array[joint["movable_index"]])
                )
            elif joint["type"] == "prismatic":
                displacement = np.eye(4)
                displacement[:3, 3] = joint["axis"] * q_array[joint["movable_index"]]
                transform = transform @ displacement
            transforms[joint["child"]] = transform
            pending.remove(joint)
            progressed = True
        if not progressed:
            raise ValueError("URDF joint graph 不连通或含环")
    return transforms


def _transform_triangles(triangles: np.ndarray, transform: np.ndarray) -> np.ndarray:
    flat = np.ascontiguousarray(triangles.reshape(-1, 3), dtype=np.float64)
    result = np.einsum("ij,kj->ki", transform[:3, :3], flat) + transform[:3, 3]
    return result.reshape(triangles.shape)


def _view_basis() -> np.ndarray:
    forward = np.asarray([0.42, -1.0, 0.28], dtype=np.float64)
    forward /= np.linalg.norm(forward)
    world_up = np.asarray([0.0, 0.0, 1.0])
    right = np.cross(forward, world_up)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    return np.stack([right, up, forward])


def render_robot(
    urdf_path: Path,
    output: Path,
    q: Optional[Sequence[float]] = None,
    preset: str = "neutral",
    width: int = 900,
    height: int = 900,
    max_triangles: Optional[int] = None,
) -> Dict[str, Any]:
    robot = load_robot(urdf_path)
    q_array = pose_from_preset(robot, preset) if q is None else np.asarray(q, dtype=np.float64)
    transforms = forward_kinematics(robot, q_array)
    triangle_groups: List[np.ndarray] = []
    color_groups: List[np.ndarray] = []
    mesh_files = []
    for link in robot["links"]:
        for visual in robot["visuals"].get(link, []):
            triangles = load_stl(visual["path"]).copy()
            triangles *= visual["scale"].reshape(1, 1, 3)
            triangles = _transform_triangles(triangles, transforms[link] @ visual["origin"])
            triangle_groups.append(triangles)
            color_groups.append(np.tile(np.asarray(visual["color"]), (len(triangles), 1)))
            mesh_files.append(str(visual["path"]))
    if not triangle_groups:
        raise ValueError(f"URDF 没有可渲染 mesh: {urdf_path}")
    triangles = np.concatenate(triangle_groups)
    colors = np.concatenate(color_groups)
    source_triangles = len(triangles)
    if max_triangles and len(triangles) > max_triangles:
        indices = np.linspace(0, len(triangles) - 1, max_triangles, dtype=np.int64)
        triangles, colors = triangles[indices], colors[indices]

    camera = triangles @ _view_basis().T
    xy = camera[:, :, :2]
    low = xy.reshape(-1, 2).min(axis=0)
    high = xy.reshape(-1, 2).max(axis=0)
    span = np.maximum(high - low, 1e-9)
    padding = 0.07 * min(width, height)
    scale = min((width - 2 * padding) / span[0], (height - 2 * padding) / span[1])
    center = (low + high) / 2
    screen = (xy - center) * scale
    screen[:, :, 0] += width / 2
    screen[:, :, 1] = height / 2 - screen[:, :, 1]

    edge1 = camera[:, 1] - camera[:, 0]
    edge2 = camera[:, 2] - camera[:, 0]
    normals = np.cross(edge1, edge2)
    norm = np.linalg.norm(normals, axis=1)
    valid = norm > 1e-12
    normals[valid] /= norm[valid, None]
    light = np.asarray([-0.25, 0.45, 0.86])
    light /= np.linalg.norm(light)
    intensity = 0.48 + 0.52 * np.abs(np.einsum("ij,j->i", normals, light))
    shaded = np.clip(colors * intensity[:, None], 0, 255).astype(np.uint8)
    depth = camera[:, :, 2].mean(axis=1)
    order = np.argsort(depth)

    image = Image.new("RGB", (width, height), (244, 247, 250))
    draw = ImageDraw.Draw(image)
    for index in order:
        polygon = [tuple(value) for value in screen[index]]
        color = tuple(int(value) for value in shaded[index])
        draw.polygon(polygon, fill=color)
    draw.rounded_rectangle((18, 18, 345, 76), radius=10, fill=(255, 255, 255), outline=(205, 213, 220))
    draw.text((34, 29), robot["name"], fill=(20, 28, 36))
    draw.text((34, 51), f"20DoF mesh | pose={preset if q is None else 'custom'}", fill=(65, 77, 88))
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)
    return {
        "output": str(output),
        "robot_name": robot["name"],
        "joint_count": len(robot["joint_order"]),
        "joint_order": robot["joint_order"],
        "q": q_array.tolist(),
        "pose": preset if q is None else "custom",
        "mesh_file_count": len(mesh_files),
        "source_triangle_count": source_triangles,
        "rendered_triangle_count": len(triangles),
        "width": width,
        "height": height,
    }
