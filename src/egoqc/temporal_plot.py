from __future__ import annotations

import html
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .temporal import _nanmax_rows, _position_residual, _rotation_residual
from .validator import _arrays, load_episode_index


def _route_value(row: Dict[str, Any], key: str) -> Any:
    value = row[key]
    return value.as_py() if isinstance(value, pa.Scalar) else value


def _points(values: np.ndarray, left: float, top: float, width: float, height: float, maximum: float) -> str:
    values = np.asarray(values, dtype=np.float64)
    if len(values) > 1000:
        indices = np.unique(np.linspace(0, len(values) - 1, 1000).astype(int))
    else:
        indices = np.arange(len(values))
    finite = np.isfinite(values[indices])
    if not np.any(finite):
        return ""
    x = left + indices[finite] / max(1, len(values) - 1) * width
    y = top + height - np.clip(values[indices][finite], 0, maximum) / maximum * height
    return " ".join(f"{xv:.2f},{yv:.2f}" for xv, yv in zip(x, y))


def write_temporal_plot(
    dataset: Path,
    episode_index: int,
    output: Path,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    dataset = dataset.expanduser().resolve()
    rows = load_episode_index(dataset).to_pylist()
    row = next((item for item in rows if int(_route_value(item, "episode_index")) == episode_index), None)
    if row is None:
        raise ValueError(f"episode {episode_index} 不存在")
    data_path = dataset / "data" / f"chunk-{int(_route_value(row, 'data/chunk_index')):03d}" / f"file-{int(_route_value(row, 'data/file_index')):03d}.parquet"
    columns = [
        "episode_index", "state_mask", "left_transl_world", "right_transl_world",
        "left_orient_world", "right_orient_world", "left_hand_pose", "right_hand_pose",
        "extrinsics_w2c",
    ]
    table = pq.read_table(data_path, columns=columns)
    episodes = _arrays(table, "episode_index", np.int64).reshape(-1)
    table = table.filter(pa.array(episodes == episode_index))
    length = len(table)
    mask = _arrays(table, "state_mask", bool).reshape(length, 2)
    left = _arrays(table, "left_transl_world").reshape(length, 3)
    right = _arrays(table, "right_transl_world").reshape(length, 3)
    left_orient = _arrays(table, "left_orient_world").reshape(length, 3, 3)
    right_orient = _arrays(table, "right_orient_world").reshape(length, 3, 3)
    left_pose = _arrays(table, "left_hand_pose").reshape(length, 15, 3, 3)
    right_pose = _arrays(table, "right_hand_pose").reshape(length, 15, 3, 3)
    extr = _arrays(table, "extrinsics_w2c").reshape(length, 4, 4)
    camera_valid = np.isfinite(extr).all(axis=(1, 2))
    series = {
        "position": (
            _position_residual(left, mask[:, 0]) * 1000.0,
            _position_residual(right, mask[:, 1]) * 1000.0,
            _position_residual(extr[:, :3, 3], camera_valid) * 1000.0,
        ),
        "wrist": (
            _rotation_residual(left_orient, mask[:, 0]),
            _rotation_residual(right_orient, mask[:, 1]),
            _rotation_residual(extr[:, :3, :3], camera_valid),
        ),
        "joints": (
            _nanmax_rows(_rotation_residual(left_pose, mask[:, 0])),
            _nanmax_rows(_rotation_residual(right_pose, mask[:, 1])),
        ),
    }
    thresholds = config.get("thresholds", {})
    panels: Iterable[Tuple[str, str, Tuple[np.ndarray, ...], float]] = (
        ("Position local residual", "mm", series["position"], float(thresholds.get("position_jitter_error_m", 0.020)) * 1000.0),
        ("Wrist / camera SO(3) residual", "deg", series["wrist"], float(thresholds.get("rotation_jitter_error_deg", 15.0))),
        ("Finger joint SO(3) residual", "deg", series["joints"], float(thresholds.get("joint_jitter_error_deg", 20.0))),
    )
    width, height = 1280, 820
    plot_left, plot_width, panel_height = 86, 1140, 174
    colors = ("#2dcfbe", "#f4a340", "#637083")
    blocks = []
    peaks: Dict[str, float] = {}
    for panel_index, (title, unit, values, threshold) in enumerate(panels):
        top = 118 + panel_index * 220
        finite_values = np.concatenate([value[np.isfinite(value)] for value in values])
        observed = float(np.percentile(finite_values, 99.5)) if finite_values.size else 0.0
        maximum = max(threshold * 1.25, observed * 1.1, 1e-6)
        peaks[title] = observed
        blocks.append(f'<text x="{plot_left}" y="{top - 18}" class="panel-title">{html.escape(title)}</text>')
        blocks.append(f'<text x="{plot_left + plot_width}" y="{top - 18}" text-anchor="end" class="unit">p99.5 {observed:.2f} {unit}</text>')
        blocks.append(f'<rect x="{plot_left}" y="{top}" width="{plot_width}" height="{panel_height}" class="plot"/>')
        threshold_y = top + panel_height - min(1.0, threshold / maximum) * panel_height
        blocks.append(f'<line x1="{plot_left}" y1="{threshold_y:.2f}" x2="{plot_left + plot_width}" y2="{threshold_y:.2f}" class="threshold"/>')
        blocks.append(f'<text x="{plot_left + 8}" y="{threshold_y - 5:.2f}" class="threshold-label">error {threshold:g} {unit}</text>')
        for index, value in enumerate(values):
            points = _points(value, plot_left, top, plot_width, panel_height, maximum)
            if points:
                blocks.append(f'<polyline points="{points}" stroke="{colors[index]}" class="signal"/>')
        blocks.append(f'<text x="{plot_left - 12}" y="{top + 6}" text-anchor="end" class="axis">{maximum:.1f}</text>')
        blocks.append(f'<text x="{plot_left - 12}" y="{top + panel_height}" text-anchor="end" class="axis">0</text>')
    document = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<style>
text{{font-family:Inter,system-ui,sans-serif;fill:#17201b}} .title{{font-size:28px;font-weight:650}}
.subtitle,.unit,.axis{{font-size:12px;fill:#68736c}} .panel-title{{font-size:15px;font-weight:600}}
.plot{{fill:#f7f9f7;stroke:#dfe5df}} .signal{{fill:none;stroke-width:1.6;stroke-linejoin:round}}
.threshold{{stroke:#c84b45;stroke-width:1;stroke-dasharray:5 4}} .threshold-label{{font-size:10px;fill:#a53d38}}
.legend{{font-size:12px}}
</style>
<rect width="100%" height="100%" fill="#ffffff"/>
<text x="54" y="48" class="title">Temporal QC · episode {episode_index}</text>
<text x="54" y="74" class="subtitle">{html.escape(dataset.name)} · {length} frames · local interpolation residual</text>
<circle cx="830" cy="48" r="5" fill="#2dcfbe"/><text x="842" y="52" class="legend">left</text>
<circle cx="905" cy="48" r="5" fill="#f4a340"/><text x="917" y="52" class="legend">right</text>
<circle cx="988" cy="48" r="5" fill="#637083"/><text x="1000" y="52" class="legend">camera</text>
{''.join(blocks)}
<text x="{plot_left}" y="798" class="axis">frame 0</text><text x="{plot_left + plot_width}" y="798" text-anchor="end" class="axis">frame {max(0, length - 1)}</text>
</svg>'''
    output = output.expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document, encoding="utf-8")
    return {"episode_index": episode_index, "frames": length, "output": str(output.resolve()), "p99_5": peaks}
