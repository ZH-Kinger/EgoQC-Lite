from __future__ import annotations

import html
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

from .registry import _connect


def write_registry_dashboard(registry_path: Path, output_path: Path) -> Dict[str, Any]:
    db = _connect(registry_path)
    try:
        datasets = [dict(row) for row in db.execute(
            "SELECT * FROM datasets ORDER BY source, logical_path"
        ).fetchall()]
        runs = [dict(row) for row in db.execute(
            """
            SELECT * FROM runs
            ORDER BY COALESCE(finished_at, started_at) DESC
            """
        ).fetchall()]
    finally:
        db.close()

    latest_by_dataset: Dict[str, Dict[str, Any]] = {}
    for run in runs:
        latest_by_dataset.setdefault(run["dataset_id"], run)
    records: List[Dict[str, Any]] = []
    for dataset in datasets:
        run = latest_by_dataset.get(dataset["dataset_id"])
        summary = {}
        if run and run.get("summary_json"):
            try:
                summary = json.loads(run["summary_json"])
            except json.JSONDecodeError:
                summary = {}
        records.append(
            {
                "dataset_id": dataset["dataset_id"],
                "source": dataset["source"],
                "logical_path": dataset["logical_path"],
                "total_bytes": dataset["total_bytes"],
                "missing_files": dataset["missing_file_count"],
                "status": run["status"] if run else "unplanned",
                "standard_version": run["standard_version"] if run else None,
                "finished_at": run["finished_at"] if run else None,
                "output_path": run["output_path"] if run else None,
                "error": run["error"] if run else None,
                "progress_fraction": run["progress_fraction"] if run else 0,
                "eta_seconds": run["eta_seconds"] if run else None,
                "progress_path": run["progress_path"] if run else None,
                "episode_count": summary.get("episode_count", 0),
                "tier_counts": summary.get("tier_counts", {}),
                "elapsed_s": summary.get("elapsed_s", 0.0),
                "throughput": summary.get("logical_throughput_mib_s", 0.0),
                "parquet_hit": summary.get("parquet_cache_hit_ratio", 0.0),
                "video_hit": summary.get("video_cache_hit_ratio", 0.0),
            }
        )

    status_counts = Counter(record["status"] for record in records)
    total_episodes = sum(int(record["episode_count"]) for record in records)
    tier_totals = Counter()
    for record in records:
        tier_totals.update(record["tier_counts"])
    gold_rate = tier_totals["gold"] / max(1, total_episodes)
    total_bytes = sum(int(record["total_bytes"]) for record in records)
    payload = json.dumps(
        records,
        ensure_ascii=False,
        separators=(",", ":"),
    ).replace("</", "<\\/")
    status_pills = "".join(
        f'<span class="status-pill {html.escape(status)}">{html.escape(status)} '
        f"<b>{count}</b></span>"
        for status, count in sorted(status_counts.items())
    )
    document = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>EgoQC-Lite · Registry</title><style>
:root{{--bg:#f4f6f4;--surface:#fff;--ink:#18211b;--muted:#6d766f;--line:#dde3de;
--green:#20744a;--green-soft:#dceee3;--red:#b3453f;--red-soft:#f3dcda;--amber:#9a641e;
--amber-soft:#f3e7d2;--blue:#326da8;--silver:#7c878f}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);
font-family:Inter,ui-sans-serif,system-ui,-apple-system,sans-serif}}main{{max-width:1420px;margin:auto;padding:34px 28px 64px}}
.eyebrow{{font-size:12px;letter-spacing:.08em;text-transform:uppercase;color:var(--muted)}}h1{{font-size:32px;margin:8px 0}}
.muted{{color:var(--muted)}}.cards{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:28px 0}}
.card,.panel{{background:var(--surface);border:1px solid var(--line);border-radius:12px}}.card{{padding:17px}}
.card span{{font-size:12px;color:var(--muted)}}.card b{{display:block;font-size:26px;margin-top:7px}}.panel{{overflow:hidden}}
.head{{padding:18px 20px;display:flex;align-items:center;justify-content:space-between;gap:14px;flex-wrap:wrap}}
.statuses{{display:flex;gap:7px;flex-wrap:wrap}}.status-pill,.status{{display:inline-block;padding:4px 8px;border-radius:99px;font-size:11px}}
.status-pill{{background:var(--bg)}}.status.succeeded{{background:var(--green-soft);color:var(--green)}}.status.failed,.status.stale{{background:var(--red-soft);color:var(--red)}}
.status.running{{background:var(--amber-soft);color:var(--amber)}}.status.unplanned{{background:#e9edef;color:#566169}}
input,select{{font:inherit;border:1px solid var(--line);border-radius:8px;background:var(--surface);padding:8px 10px;color:var(--ink)}}
input{{min-width:260px}}table{{width:100%;border-collapse:collapse;font-size:12px}}th,td{{padding:11px 14px;border-top:1px solid var(--line);text-align:left;vertical-align:middle}}
th{{background:#fafbfa;color:var(--muted);font-weight:550}}.path{{max-width:310px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}
.tierbar{{display:flex;width:130px;height:8px;background:var(--line);border-radius:99px;overflow:hidden}}.gold{{background:var(--green)}}.silver{{background:var(--silver)}}
.bronze{{background:var(--amber)}}.quarantine{{background:var(--red)}}.num{{font-variant-numeric:tabular-nums}}.empty{{padding:30px;text-align:center;color:var(--muted)}}
.foot{{font-size:11px;color:var(--muted);margin-top:14px}}@media(max-width:850px){{main{{padding:22px 14px}}.cards{{grid-template-columns:1fr 1fr}}
.panel{{overflow-x:auto}}table{{min-width:950px}}}}
</style></head><body><main>
<div class="eyebrow">EgoQC-Lite · Registry control plane</div><h1>全局数据质量总览</h1>
<p class="muted">{html.escape(str(registry_path.expanduser().resolve()))}</p>
<section class="cards">
<div class="card"><span>Datasets</span><b>{len(records):,}</b></div>
<div class="card"><span>Episodes</span><b>{total_episodes:,}</b></div>
<div class="card"><span>Gold 比例</span><b>{gold_rate:.1%}</b></div>
<div class="card"><span>登记数据量</span><b>{total_bytes / 1024**4:.2f}<small> TiB</small></b></div>
</section>
<section class="panel"><div class="head"><div class="statuses">{status_pills}</div>
<div><select id="status-filter" aria-label="状态筛选"><option value="all">全部状态</option>
<option>succeeded</option><option>failed</option><option>running</option><option>stale</option><option>unplanned</option></select>
<input id="search" type="search" placeholder="搜索数据集、来源或路径" aria-label="搜索"></div></div>
<table><thead><tr><th>数据集</th><th>来源 / 路径</th><th>状态 / 进度</th><th>Episodes</th>
<th>质量分层</th><th>吞吐</th><th>缓存</th><th>完成时间</th></tr></thead>
<tbody id="dataset-body"></tbody></table></section>
<p class="foot">最新 run 优先展示；完整历史保存在 registry.sqlite。</p>
</main><script>
const records={payload};const body=document.getElementById("dataset-body");
const search=document.getElementById("search");const statusFilter=document.getElementById("status-filter");
const esc=v=>String(v??"").replace(/[&<>"']/g,c=>({{"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}}[c]));
function tierBar(record){{const total=Math.max(1,record.episode_count);return ["gold","silver","bronze","quarantine"].map(tier=>{{
const count=record.tier_counts[tier]||0;return count?`<span class="${{tier}}" style="width:${{count/total*100}}%" title="${{tier}}: ${{count}}"></span>`:"";}}).join("");}}
function duration(seconds){{if(seconds==null)return "—";const s=Math.max(0,Number(seconds));if(s<60)return `${{s.toFixed(0)}}s`;if(s<3600)return `${{(s/60).toFixed(1)}}m`;return `${{(s/3600).toFixed(1)}}h`;}}
function render(){{const query=search.value.trim().toLowerCase();const selected=statusFilter.value;
const visible=records.filter(record=>(selected==="all"||record.status===selected)&&(!query||`${{record.dataset_id}} ${{record.source}} ${{record.logical_path}}`.toLowerCase().includes(query)));
body.innerHTML=visible.map(record=>`<tr><td><strong>${{esc(record.dataset_id)}}</strong><br><span class="muted">${{esc(record.standard_version||"—")}}</span></td>
<td><strong>${{esc(record.source)}}</strong><div class="path muted" title="${{esc(record.logical_path)}}">${{esc(record.logical_path)}}</div></td>
<td><span class="status ${{esc(record.status)}}">${{esc(record.status)}}</span><div class="muted num">${{record.status==="running"?`${{(Number(record.progress_fraction||0)*100).toFixed(1)}}% · ETA ${{duration(record.eta_seconds)}}`:""}}</div></td><td class="num">${{Number(record.episode_count).toLocaleString()}}</td>
<td><div class="tierbar">${{tierBar(record)}}</div></td><td class="num">${{Number(record.throughput).toFixed(1)}} MiB/s</td>
<td class="num">P ${{(record.parquet_hit*100).toFixed(0)}}% · V ${{(record.video_hit*100).toFixed(0)}}%</td>
<td>${{esc(record.finished_at||"—")}}</td></tr>`).join("")||`<tr><td colspan="8" class="empty">没有匹配的数据集</td></tr>`;}}
search.addEventListener("input",render);statusFilter.addEventListener("change",render);render();
</script></body></html>"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    temporary.write_text(document, encoding="utf-8")
    temporary.replace(output_path)
    return {
        "output": str(output_path.resolve()),
        "dataset_count": len(records),
        "episode_count": total_episodes,
        "gold_rate": gold_rate,
        "status_counts": dict(status_counts),
    }
