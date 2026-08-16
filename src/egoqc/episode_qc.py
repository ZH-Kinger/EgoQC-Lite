from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .decisions import acceptance_for
from .validator import (
    load_episode_index,
    load_task_map,
    validate_dataset_structure,
    validate_episode,
)


def inspect_episode(dataset: Path, episode: int, config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate one routed episode without scanning a multi-thousand-episode root."""
    dataset = dataset.expanduser().resolve()
    _, dataset_issues = validate_dataset_structure(dataset, config)
    info = json.loads((dataset / "meta" / "info.json").read_text(encoding="utf-8"))
    video_key = config["video_key"]
    fps = float(
        info["fps"]
        if "fps" in info
        else info["features"][video_key]["info"]["video.fps"]
    )
    rows = load_episode_index(dataset).to_pylist()
    matches = [row for row in rows if int(row["episode_index"]) == episode]
    if len(matches) != 1:
        raise ValueError(f"episode {episode} 路由数量应为 1，实际 {len(matches)}")
    row = matches[0]
    data_path = (
        dataset
        / "data"
        / f"chunk-{int(row['data/chunk_index']):03d}"
        / f"file-{int(row['data/file_index']):03d}.parquet"
    )
    parquet = pq.ParquetFile(data_path)
    available = set(parquet.schema_arrow.names)
    requested = list(dict.fromkeys(config["required_frame_columns"] + ["main_type"]))
    table = parquet.read(columns=[name for name in requested if name in available])
    values = np.asarray(table["episode_index"].to_pylist(), dtype=np.int64)
    episode_table = table.filter(pa.array(values == episode))
    result = validate_episode(
        episode_table,
        episode,
        int(row["length"]),
        fps,
        config,
        str(data_path.relative_to(dataset)),
        filtered=True,
        expected_from_index=row.get("dataset_from_index"),
        expected_to_index=row.get("dataset_to_index"),
        task_map=load_task_map(dataset),
        expected_tasks=row.get("tasks", []),
    )
    return {
        "dataset": str(dataset),
        "episode_index": episode,
        "task": row.get("tasks"),
        "data_file": str(data_path.relative_to(dataset)),
        "shard_rows": len(table),
        "dataset_issues": [issue.to_dict() for issue in dataset_issues],
        "acceptance": acceptance_for(list(result.issues) + dataset_issues, config),
        "result": result.to_dict(),
    }
