from __future__ import annotations

import hashlib
import inspect
import sys
import warnings
import builtins
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw


MANO21_SKELETON_EDGES = tuple(
    edge
    for start in (1, 5, 9, 13, 17)
    for edge in ((0, start), (start, start + 1), (start + 1, start + 2), (start + 2, start + 3))
)

# Native MANO order before HaWoR appends fingertips and reorders to OpenPose.
MANO16_SKELETON_EDGES = tuple(
    edge
    for chain in ((1, 2, 3), (4, 5, 6), (7, 8, 9), (10, 11, 12), (13, 14, 15))
    for edge in ((0, chain[0]), (chain[0], chain[1]), (chain[1], chain[2]))
)


def _enable_chumpy_compatibility() -> None:
    """Provide removed Python/NumPy aliases required while unpickling MANO.

    Official MANO model files contain legacy ``chumpy`` objects.  Chumpy 0.70
    still imports ``inspect.getargspec`` and NumPy scalar aliases that were
    removed in Python 3.11 and NumPy 2.0.  Keep the workaround isolated to the
    optional MANO backend instead of pinning the whole QC pipeline to an old
    Python/NumPy environment.
    """

    if not hasattr(inspect, "getargspec"):
        inspect.getargspec = inspect.getfullargspec  # type: ignore[attr-defined]
    aliases = {
        "bool": np.bool_,
        "int": builtins.int,
        "float": builtins.float,
        "complex": builtins.complex,
        "object": builtins.object,
        "unicode": builtins.str,
        "str": builtins.str,
    }
    for name, value in aliases.items():
        if name not in np.__dict__:
            setattr(np, name, value)


def mano_skeleton_edges(joint_count: int) -> Tuple[Tuple[int, int], ...]:
    if joint_count >= 21:
        return MANO21_SKELETON_EDGES
    if joint_count == 16:
        return MANO16_SKELETON_EDGES
    return ()


class ManoBackend(Protocol):
    """Small interface that keeps the quality pipeline independent of HaWoR."""

    def j0_canon(self, betas: np.ndarray, is_right: bool) -> np.ndarray:
        ...

    def forward(
        self,
        transl: np.ndarray,
        orient_rotmat: np.ndarray,
        hand_pose_rotmat: np.ndarray,
        betas: np.ndarray,
        is_right: bool,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return world-space vertices, joints and triangle faces."""
        ...

    def provenance(self) -> Dict[str, Any]:
        ...


def restore_left_pose(rotations: np.ndarray) -> np.ndarray:
    """Restore real left-hand local rotations from right-hand canonical storage.

    The dataset stores left-hand rotations as ``M R M^-1`` where
    ``M = diag(1, -1, -1)``.  M is its own inverse, so the restoration uses the
    same conjugation and is numerically lossless.
    """

    rotations = np.asarray(rotations, dtype=np.float64)
    mirror = np.diag([1.0, -1.0, -1.0])
    return np.einsum("ij,...jk,kl->...il", mirror, rotations, mirror)


def camera_intrinsics(
    width: int,
    height: int,
    fov: Optional[np.ndarray] = None,
    intrinsics: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Return a 3x3 K, preferring a valid stored matrix over FOV."""

    if intrinsics is not None:
        candidate = np.asarray(intrinsics, dtype=np.float64).reshape(3, 3)
        if np.isfinite(candidate).all() and candidate[0, 0] > 0 and candidate[1, 1] > 0:
            return candidate
    if fov is None:
        raise ValueError("缺少有效 intrinsics 和 fov，无法投影 MANO")
    fov = np.asarray(fov, dtype=np.float64).reshape(2)
    if not np.isfinite(fov).all() or np.any(fov <= 0) or np.any(fov >= np.pi):
        raise ValueError(f"非法 FOV: {fov.tolist()}")
    return np.array(
        [
            [(width / 2.0) / np.tan(fov[0] / 2.0), 0.0, width / 2.0],
            [0.0, (height / 2.0) / np.tan(fov[1] / 2.0), height / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def world_to_pixels(
    points_world: np.ndarray,
    extrinsics_w2c: np.ndarray,
    intrinsics: np.ndarray,
    minimum_depth_m: float = 0.01,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project world-space points using the dataset's OpenCV camera convention."""

    points = np.asarray(points_world, dtype=np.float64)
    extrinsics = np.asarray(extrinsics_w2c, dtype=np.float64).reshape(4, 4)
    k = np.asarray(intrinsics, dtype=np.float64).reshape(3, 3)
    camera = points @ extrinsics[:3, :3].T + extrinsics[:3, 3]
    depth = camera[:, 2]
    valid = np.isfinite(camera).all(axis=1) & (depth > minimum_depth_m)
    pixels = np.full((len(points), 2), np.nan, dtype=np.float64)
    projected = camera @ k.T
    pixels[valid] = projected[valid, :2] / projected[valid, 2:3]
    return pixels, depth, valid


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


class HaworManoBackend:
    """Adapter for the exact MANO wrapper used by the HaWoR inference code."""

    def __init__(self, hawor_root: Path, data_root: Optional[Path] = None):
        self.hawor_root = hawor_root.expanduser().resolve()
        if not (self.hawor_root / "lib" / "models" / "mano_wrapper.py").exists():
            raise FileNotFoundError(
                f"HaWoR MANO wrapper 不存在: {self.hawor_root / 'lib/models/mano_wrapper.py'}"
            )
        self.data_root = (
            data_root.expanduser().resolve()
            if data_root
            else self.hawor_root / "_DATA"
        )
        self.model_files = {
            "right": self.data_root / "data" / "mano" / "MANO_RIGHT.pkl",
            "left": self.data_root / "data_left" / "mano_left" / "MANO_LEFT.pkl",
        }
        missing = [str(path) for path in self.model_files.values() if not path.exists()]
        if missing:
            raise FileNotFoundError("缺少 MANO 模型文件: " + ", ".join(missing))
        if str(self.hawor_root) not in sys.path:
            sys.path.insert(0, str(self.hawor_root))
        _enable_chumpy_compatibility()
        from lib.models.mano_wrapper import MANO  # type: ignore

        self._mano_class = MANO
        self._models: Dict[str, Any] = {}

    def _model(self, is_right: bool) -> Any:
        key = "right" if is_right else "left"
        if key in self._models:
            return self._models[key]
        if is_right:
            options = dict(
                data_dir=str(self.data_root / "data") + "/",
                model_path=str(self.data_root / "data" / "mano"),
                gender="neutral",
                num_hand_joints=15,
                create_body_pose=False,
            )
        else:
            options = dict(
                data_dir=str(self.data_root / "data_left") + "/",
                model_path=str(self.data_root / "data_left" / "mano_left"),
                gender="neutral",
                num_hand_joints=15,
                create_body_pose=False,
                is_rhand=False,
            )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = self._mano_class(**options)
        if not is_right:
            # HaWoR applies this fix to the official left MANO shapedirs.
            model.shapedirs[:, 0, :] *= -1
        self._models[key] = model
        return model

    def j0_canon(self, betas: np.ndarray, is_right: bool) -> np.ndarray:
        import torch

        betas = np.asarray(betas, dtype=np.float32).reshape(-1, 10)
        count = len(betas)
        model = self._model(is_right)
        global_orient = torch.eye(3).view(1, 1, 3, 3).expand(count, 1, 3, 3).contiguous()
        hand_pose = torch.eye(3).view(1, 1, 3, 3).expand(count, 15, 3, 3).contiguous()
        with torch.inference_mode():
            output = model(
                global_orient=global_orient,
                hand_pose=hand_pose,
                betas=torch.from_numpy(np.nan_to_num(betas)),
                transl=torch.zeros(count, 3),
                pose2rot=False,
            )
        return output.joints[:, 0, :].detach().cpu().numpy().astype(np.float32)

    def forward(
        self,
        transl: np.ndarray,
        orient_rotmat: np.ndarray,
        hand_pose_rotmat: np.ndarray,
        betas: np.ndarray,
        is_right: bool,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        import torch

        model = self._model(is_right)
        transl = np.asarray(transl, dtype=np.float32).reshape(-1, 3)
        count = len(transl)
        with torch.inference_mode():
            output = model(
                global_orient=torch.from_numpy(np.nan_to_num(orient_rotmat)).float().reshape(
                    count, 1, 3, 3
                ),
                hand_pose=torch.from_numpy(np.nan_to_num(hand_pose_rotmat)).float().reshape(
                    count, 15, 3, 3
                ),
                betas=torch.from_numpy(np.nan_to_num(betas)).float().reshape(count, 10),
                transl=torch.from_numpy(np.nan_to_num(transl)).float(),
                pose2rot=False,
            )
        faces = np.asarray(model.faces, dtype=np.int64)
        if not is_right:
            faces = faces[:, [0, 2, 1]]
        return (
            output.vertices.detach().cpu().numpy().astype(np.float32),
            output.joints.detach().cpu().numpy().astype(np.float32),
            faces,
        )

    def provenance(self) -> Dict[str, Any]:
        return {
            "backend": "hawor",
            "hawor_root": str(self.hawor_root),
            "mano_models": {
                side: {"path": str(path), "sha256": _sha256(path)}
                for side, path in self.model_files.items()
            },
        }


class ManoOverlayRenderer:
    def __init__(self, backend: ManoBackend, alpha: float = 0.48):
        self.backend = backend
        self.alpha = float(np.clip(alpha, 0.0, 1.0))

    def render(self, image: Image.Image, record: Dict[str, Any]) -> Tuple[Image.Image, Dict[str, Any]]:
        images, metrics = self.render_many([image], [record])
        return images[0], metrics[0]

    def render_many(
        self,
        images: Sequence[Image.Image],
        records: Sequence[Dict[str, Any]],
    ) -> Tuple[List[Image.Image], List[Dict[str, Any]]]:
        """Batch MANO inference while preserving per-frame raster overlays."""
        if len(images) != len(records):
            raise ValueError("images 与 records 长度必须一致")
        rgb_images = [image.convert("RGB") for image in images]
        overlays = [Image.new("RGBA", image.size, (0, 0, 0, 0)) for image in rgb_images]
        draws = [ImageDraw.Draw(overlay, "RGBA") for overlay in overlays]
        metrics: List[Dict[str, Any]] = [
            {"hands_rendered": 0, "sides": {}} for _ in records
        ]

        for side, is_right, mask_index, beta_slice, color_rgb in (
            ("left", False, 0, slice(51, 61), (45, 208, 190)),
            ("right", True, 1, slice(112, 122), (255, 170, 60)),
        ):
            selected = [
                index
                for index, record in enumerate(records)
                if np.asarray(record["state_mask"], dtype=bool).reshape(2)[mask_index]
            ]
            if not selected:
                continue
            states = np.stack(
                [np.asarray(records[index]["observation.state"], dtype=np.float32) for index in selected]
            ).reshape(-1, 122)
            betas = states[:, beta_slice]
            transl_world = np.stack(
                [np.asarray(records[index][f"{side}_transl_world"], dtype=np.float32) for index in selected]
            ).reshape(-1, 3)
            orient = np.stack(
                [np.asarray(records[index][f"{side}_orient_world"], dtype=np.float32) for index in selected]
            ).reshape(-1, 3, 3)
            pose = np.stack(
                [np.asarray(records[index][f"{side}_hand_pose"], dtype=np.float32) for index in selected]
            ).reshape(-1, 15, 3, 3)
            if not is_right:
                pose = restore_left_pose(pose).astype(np.float32)
            j0 = self.backend.j0_canon(betas, is_right)
            vertices_batch, joints_batch, faces = self.backend.forward(
                transl_world - j0, orient, pose, betas, is_right
            )
            for batch_index, output_index in enumerate(selected):
                record = records[output_index]
                width, height = rgb_images[output_index].size
                k = camera_intrinsics(width, height, record.get("fov"), record.get("intrinsics"))
                extrinsics = np.asarray(record["extrinsics_w2c"], dtype=np.float64).reshape(4, 4)
                pixels, depth, valid = world_to_pixels(vertices_batch[batch_index], extrinsics, k)
                joint_pixels, _, joint_valid = world_to_pixels(
                    joints_batch[batch_index], extrinsics, k
                )
                inside = (
                    valid
                    & (pixels[:, 0] >= 0)
                    & (pixels[:, 0] < width)
                    & (pixels[:, 1] >= 0)
                    & (pixels[:, 1] < height)
                )
                visible_faces = faces[np.all(valid[faces], axis=1)]
                draw = draws[output_index]
                if len(visible_faces):
                    face_color = color_rgb + (round(255 * self.alpha),)
                    order = np.argsort(-depth[visible_faces].mean(axis=1))
                    for face in visible_faces[order]:
                        draw.polygon([tuple(pixels[index]) for index in face], fill=face_color)
                skeleton_edges = mano_skeleton_edges(len(joint_pixels))
                line_width = max(3, min(width, height) // 240)
                rendered_edges = 0
                for start, end in skeleton_edges:
                    if joint_valid[start] and joint_valid[end]:
                        draw.line(
                            [tuple(joint_pixels[start]), tuple(joint_pixels[end])],
                            fill=color_rgb + (255,),
                            width=line_width,
                        )
                        rendered_edges += 1
                for x, y in joint_pixels[joint_valid]:
                    radius = line_width + 1
                    draw.ellipse(
                        (x - radius, y - radius, x + radius, y + radius),
                        fill=(255, 255, 255, 245),
                        outline=color_rgb + (255,),
                        width=max(1, line_width // 2),
                    )
                if np.any(inside):
                    bounds = pixels[inside]
                    x0, y0 = bounds.min(axis=0)
                    x1, y1 = bounds.max(axis=0)
                    draw.rectangle((x0, y0, x1, y1), outline=color_rgb + (240,), width=3)
                metrics[output_index]["hands_rendered"] += 1
                metrics[output_index]["sides"][side] = {
                    "out_of_frame_ratio": float(1.0 - np.mean(inside)),
                    "visible_vertex_ratio": float(np.mean(inside)),
                    "wrist_pixel": (
                        [float(value) for value in joint_pixels[0]]
                        if len(joint_pixels) and joint_valid[0]
                        else None
                    ),
                    "skeleton_edge_count": rendered_edges,
                }
        composed = [
            Image.alpha_composite(rgb.convert("RGBA"), overlay).convert("RGB")
            for rgb, overlay in zip(rgb_images, overlays)
        ]
        return composed, metrics

    def provenance(self) -> Dict[str, Any]:
        return {"alpha": self.alpha, **self.backend.provenance()}
