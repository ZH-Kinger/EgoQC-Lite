from __future__ import annotations

import html
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import pyarrow as pa
import pyarrow.parquet as pq

from .types import EpisodeResult, Issue


def write_json(path: Path, value: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=True) + "\n")
    temporary.replace(path)


def write_parquet(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    values = list(rows)
    table = pa.Table.from_pylist(values) if values else pa.table({})
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    pq.write_table(table, temporary, compression="zstd")
    temporary.replace(path)


def write_report(
    path: Path,
    dataset: Path,
    summary: Dict[str, Any],
    episodes: list[EpisodeResult],
    dataset_issues: list[Issue],
    max_episodes: int = 500,
    shard_records: Optional[list[Dict[str, Any]]] = None,
) -> None:
    shard_records = shard_records or []
    tier_counts = Counter(ep.tier for ep in episodes)
    issue_counts = Counter(issue.code for ep in episodes for issue in ep.issues)
    issue_counts.update(issue.code for issue in dataset_issues)
    tier_order = {"quarantine": 0, "bronze": 1, "silver": 2, "gold": 3}
    visible_episodes = sorted(
        episodes,
        key=lambda episode: (
            tier_order.get(episode.tier, 4),
            episode.episode_index,
        ),
    )[:max(0, max_episodes)]
    total = max(1, len(episodes))
    gold_rate = tier_counts["gold"] / total
    issue_total = sum(issue_counts.values())
    issue_max = max(issue_counts.values(), default=1)
    issue_bars = "".join(
        f"""<div class="bar-row"><code>{html.escape(code)}</code>
        <div class="bar-track"><span style="width:{count / issue_max * 100:.2f}%"></span></div>
        <strong>{count}</strong></div>"""
        for code, count in issue_counts.most_common(12)
    ) or '<p class="empty">没有发现问题</p>'
    tier_segments = "".join(
        f"""<span class="segment {tier}" style="width:{tier_counts[tier] / total * 100:.4f}%"
        title="{tier}: {tier_counts[tier]}"></span>"""
        for tier in ("gold", "silver", "bronze", "quarantine")
        if tier_counts[tier]
    )
    cache = summary.get("cache", {})
    parquet_hit = float(summary.get("parquet_cache_hit_ratio", 0.0))
    video_hit = float(summary.get("video_cache_hit_ratio", 0.0))
    slow_shards = sorted(
        shard_records,
        key=lambda record: float(record.get("elapsed_s", 0.0)),
        reverse=True,
    )[:15]
    shard_max = max(
        (float(record.get("elapsed_s", 0.0)) for record in slow_shards),
        default=1.0,
    )
    shard_row_values = []
    for record in slow_shards:
        latency_pct = float(record.get("elapsed_s", 0.0)) / shard_max * 100
        shard_row_values.append(
            f"""<tr><td><span class="kind">{html.escape(str(record.get("kind", "")))}</span></td>
        <td class="path-cell" title="{html.escape(str(record.get("path", "")))}">
        {html.escape(str(record.get("path", "")))}</td>
        <td>{float(record.get("elapsed_s", 0.0)):.3f}s</td>
        <td><div class="latency-track"><span style="width:{latency_pct:.2f}%"></span></div></td>
        <td>{html.escape(str(record.get("cache_status", "—")))}</td></tr>"""
        )
    shard_rows = "".join(shard_row_values) or (
        '<tr><td colspan="5" class="empty">没有 shard 指标</td></tr>'
    )
    episode_payload = [
        {
            "episode_index": episode.episode_index,
            "length": episode.length,
            "tier": episode.tier,
            "sample_frames": episode.sample_frames,
            "issues": [
                {"code": issue.code, "message": issue.message}
                for issue in episode.issues
            ],
            "metrics": {
                key: value
                for key, value in episode.metrics.items()
                if key
                in {
                    "left_valid_ratio",
                    "right_valid_ratio",
                    "any_hand_valid_ratio",
                    "longest_internal_hand_missing_gap_s",
                    "longest_continuous_hand_visible_s",
                    "effective_video_duration_s",
                    "left_position_error_p95_m",
                    "right_position_error_p95_m",
                    "left_rotation_error_p95_deg",
                    "right_rotation_error_p95_deg",
                    "left_position_jitter_p99_mm",
                    "right_position_jitter_p99_mm",
                    "left_joint_rotation_jitter_p99_deg",
                    "right_joint_rotation_jitter_p99_deg",
                    "camera_translation_jitter_p99_mm",
                    "camera_rotation_jitter_p99_deg",
                    "bad_frame_count",
                    "bad_frame_ratio",
                    "numeric_frame_interval_jitter_mean_ms",
                    "numeric_frame_interval_jitter_max_ms",
                    "left_velocity_outlier_ratio",
                    "right_velocity_outlier_ratio",
                }
            },
        }
        for episode in visible_episodes
    ]
    payload = json.dumps(
        episode_payload,
        ensure_ascii=False,
        allow_nan=True,
        separators=(",", ":"),
    ).replace("</", "<\\/")
    throughput = float(summary.get("logical_throughput_mib_s", 0.0))
    elapsed = float(summary.get("elapsed_s", 0.0))
    decision_counts = summary.get("decisions", {}).get("counts", {})
    rejected_count = (
        int(decision_counts.get("reject", 0))
        + int(decision_counts.get("quarantine", 0))
        + int(decision_counts.get("rework", 0))
    )
    visibility = summary.get("visibility", {})
    effective_hours = float(visibility.get("effective_video_hours", 0.0))
    effective_ratio = float(visibility.get("effective_utilization_ratio", 0.0))
    bad_frames = summary.get("bad_frames", {})
    bad_frame_count = int(bad_frames.get("bad_frame_count", 0))
    bad_frame_ratio = float(bad_frames.get("bad_frame_ratio", 0.0))
    document = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>EgoQC-Lite · {html.escape(dataset.name)}</title>
<style>
:root{{--bg:#f5f7f4;--surface:#fff;--ink:#18201b;--muted:#69736c;--line:#dfe5df;
--green:#1f7a4d;--green-soft:#dcefe4;--silver:#7b8790;--silver-soft:#e9edef;
--bronze:#a8661f;--bronze-soft:#f4e6d1;--red:#b2413b;--red-soft:#f4dcda;--blue:#326ca8}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--ink);
font-family:Inter,ui-sans-serif,system-ui,-apple-system,sans-serif}} main{{max-width:1360px;margin:auto;padding:34px 28px 64px}}
h1{{font-size:32px;line-height:1.15;margin:8px 0}} h2{{font-size:17px;margin:0 0 18px;font-weight:650}}
.eyebrow,.muted{{color:var(--muted)}} .eyebrow{{font-size:12px;letter-spacing:.08em;text-transform:uppercase}}
.dataset-path{{word-break:break-all;margin:0}} .cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:28px 0 16px}}
.card,.panel{{background:var(--surface);border:1px solid var(--line);border-radius:12px}}
.card{{padding:17px}} .card span{{font-size:12px;color:var(--muted)}} .card b{{display:block;font-size:26px;margin-top:7px;font-weight:650}}
.tier-strip{{display:flex;height:12px;border-radius:999px;overflow:hidden;background:var(--line);margin-bottom:28px}}
.segment.gold{{background:var(--green)}} .segment.silver{{background:var(--silver)}} .segment.bronze{{background:var(--bronze)}}
.segment.quarantine{{background:var(--red)}} .grid{{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:14px}}
.panel{{padding:20px;min-width:0}} .bar-row{{display:grid;grid-template-columns:minmax(120px,1.7fr) 3fr 38px;gap:10px;align-items:center;margin:10px 0}}
code{{font-family:ui-monospace,SFMono-Regular,monospace;color:var(--ink);font-size:11px;overflow:hidden;text-overflow:ellipsis}}
.bar-track,.latency-track,.cache-track{{height:8px;background:#edf0ed;border-radius:99px;overflow:hidden}}
.bar-track span{{display:block;height:100%;background:var(--red);border-radius:99px}} .cache-line{{margin:17px 0}}
.cache-label{{display:flex;justify-content:space-between;font-size:12px;margin-bottom:7px}} .cache-track span{{display:block;height:100%;background:var(--blue)}}
.cache-meta{{display:grid;grid-template-columns:repeat(2,1fr);gap:8px;margin-top:18px}} .cache-meta div{{padding:12px;background:var(--bg);border-radius:8px}}
.cache-meta b{{display:block;font-size:18px;margin-bottom:3px}} .cache-meta span{{font-size:11px;color:var(--muted)}}
.table-panel{{padding:0;overflow:hidden;margin-bottom:14px}} .section-head{{padding:20px 20px 14px;display:flex;align-items:end;justify-content:space-between;gap:14px}}
table{{width:100%;border-collapse:collapse;font-size:12px}} th{{color:var(--muted);font-weight:550;background:#fafbfa}}
th,td{{padding:10px 14px;border-top:1px solid var(--line);text-align:left;vertical-align:top}} .path-cell{{max-width:360px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}}
.latency-track{{min-width:90px}} .latency-track span{{display:block;height:100%;background:var(--blue)}} .kind{{text-transform:uppercase;font-size:10px}}
.controls{{display:flex;gap:8px;flex-wrap:wrap;align-items:center}} button,input{{font:inherit;border:1px solid var(--line);background:var(--surface);color:var(--ink);border-radius:8px;padding:8px 10px}}
button{{cursor:pointer}} button.active{{background:var(--ink);color:#fff;border-color:var(--ink)}} input{{min-width:220px}} .tier{{display:inline-block;padding:3px 7px;border-radius:99px;font-size:10px}}
.tier.gold{{background:var(--green-soft);color:var(--green)}} .tier.silver{{background:var(--silver-soft);color:#4c5860}}
.tier.bronze{{background:var(--bronze-soft);color:#75430f}} .tier.quarantine{{background:var(--red-soft);color:var(--red)}}
.issue-line{{margin-bottom:5px}} .metric-line{{color:var(--muted);font-size:10px;margin-top:5px}} .empty{{color:var(--muted);text-align:center;padding:24px}}
.foot{{font-size:11px;color:var(--muted);margin-top:16px}} @media(max-width:850px){{main{{padding:22px 14px}}.cards{{grid-template-columns:1fr 1fr}}
.grid{{grid-template-columns:1fr}}.section-head{{align-items:stretch;flex-direction:column}}.table-panel{{overflow-x:auto}}table{{min-width:760px}}}}
</style></head><body><main>
<div class="eyebrow">EgoQC-Lite · {html.escape(summary["standard_version"])} · {html.escape(summary.get("code_version",""))}</div>
<h1>具身数据质量看板</h1>
<p class="dataset-path muted">{html.escape(str(dataset))}</p>
<section class="cards">
<div class="card"><span>Episodes</span><b>{len(episodes):,}</b></div>
<div class="card"><span>Gold 比例</span><b>{gold_rate:.1%}</b></div>
<div class="card"><span>问题记录</span><b>{issue_total:,}</b></div>
<div class="card"><span>拒绝 / 隔离</span><b>{rejected_count:,}</b></div>
<div class="card"><span>有效视频</span><b>{effective_hours:.2f}<small> h</small></b></div>
<div class="card"><span>有效时长率</span><b>{effective_ratio:.1%}</b></div>
<div class="card"><span>坏帧</span><b>{bad_frame_count:,}</b></div>
<div class="card"><span>坏帧比例</span><b>{bad_frame_ratio:.2%}</b></div>
<div class="card"><span>逻辑吞吐</span><b>{throughput:.1f}<small> MiB/s</small></b></div>
</section>
<div class="tier-strip" aria-label="质量等级分布">{tier_segments}</div>
<section class="grid">
<div class="panel"><h2>问题分布 · Top 12</h2>{issue_bars}</div>
<div class="panel"><h2>缓存与运行</h2>
<div class="cache-line"><div class="cache-label"><span>Parquet cache</span><b>{parquet_hit:.1%}</b></div>
<div class="cache-track"><span style="width:{parquet_hit * 100:.2f}%"></span></div></div>
<div class="cache-line"><div class="cache-label"><span>Video probe cache</span><b>{video_hit:.1%}</b></div>
<div class="cache-track"><span style="width:{video_hit * 100:.2f}%"></span></div></div>
<div class="cache-meta"><div><b>{elapsed:.2f}s</b><span>总耗时</span></div>
<div><b>{cache.get("parquet_hits",0) + cache.get("video_hits",0)}</b><span>缓存命中</span></div></div>
</div></section>
<section class="panel table-panel" id="shard-performance">
<div class="section-head"><div><h2>最慢 Shard</h2><span class="muted">按单 shard 耗时排序</span></div></div>
<table><thead><tr><th>类型</th><th>路径</th><th>耗时</th><th>相对耗时</th><th>缓存</th></tr></thead>
<tbody>{shard_rows}</tbody></table></section>
<section class="panel table-panel">
<div class="section-head"><div><h2>Episode 明细</h2><span class="muted">展示 {len(visible_episodes)} / {len(episodes)} 条，完整数据见 Parquet</span></div>
<div class="controls" id="tier-filter">
<button class="active" data-tier="all">全部</button>
<button data-tier="quarantine">隔离 {tier_counts["quarantine"]}</button>
<button data-tier="bronze">Bronze {tier_counts["bronze"]}</button>
<button data-tier="silver">Silver {tier_counts["silver"]}</button>
<button data-tier="gold">Gold {tier_counts["gold"]}</button>
<input id="episode-search" type="search" placeholder="搜索 episode 或 issue" aria-label="搜索 episode 或 issue">
</div></div>
<table><thead><tr><th>Episode</th><th>帧数</th><th>等级</th><th>关键指标</th><th>问题</th></tr></thead>
<tbody id="episode-body"></tbody></table></section>
<p class="foot">config {html.escape(summary.get("config_hash","")[:12])} · finished {html.escape(summary.get("finished_at","—"))}</p>
</main></body></html>"""
    script = f"""<script>
const episodes={payload};
let activeTier="all";
const body=document.getElementById("episode-body");
const search=document.getElementById("episode-search");
const esc=value=>String(value).replace(/[&<>"']/g,char=>({{"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}}[char]));
function metricText(metrics){{
  const entries=Object.entries(metrics).filter(([,value])=>Number.isFinite(value)).slice(0,4);
  return entries.map(([key,value])=>`${{esc(key.replaceAll("_"," "))}}: ${{Number(value).toFixed(3)}}`).join("<br>");
}}
function render(){{
  const query=search.value.trim().toLowerCase();
  const filtered=episodes.filter(ep=>{{
    if(activeTier!=="all"&&ep.tier!==activeTier)return false;
    const haystack=`${{ep.episode_index}} ${{ep.tier}} ${{ep.issues.map(issue=>issue.code+" "+issue.message).join(" ")}}`.toLowerCase();
    return !query||haystack.includes(query);
  }});
  body.innerHTML=filtered.map(ep=>`<tr>
    <td><strong>${{ep.episode_index}}</strong></td><td>${{ep.length}}</td>
    <td><span class="tier ${{esc(ep.tier)}}">${{esc(ep.tier)}}</span></td>
    <td><div class="metric-line">${{metricText(ep.metrics)||"—"}}</div></td>
    <td>${{ep.issues.length?ep.issues.map(issue=>`<div class="issue-line"><code>${{esc(issue.code)}}</code> ${{esc(issue.message)}}</div>`).join(""):"—"}}</td>
  </tr>`).join("")||`<tr><td colspan="5" class="empty">没有匹配的 episode</td></tr>`;
}}
document.querySelectorAll("#tier-filter button").forEach(button=>button.addEventListener("click",()=>{{
  document.querySelectorAll("#tier-filter button").forEach(item=>item.classList.remove("active"));
  button.classList.add("active");activeTier=button.dataset.tier;render();
}}));
search.addEventListener("input",render);render();
</script>"""
    document = document.replace("</body>", script + "</body>")
    path.write_text(document, encoding="utf-8")
