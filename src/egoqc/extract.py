from __future__ import annotations

import json
import math
import html
import os
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import av
import numpy as np
from PIL import Image, ImageDraw
import pyarrow as pa
import pyarrow.parquet as pq

from .mano import ManoOverlayRenderer
from .report import write_jsonl
from .validator import load_episode_index
from .review_workbench import write_review_workbench


def _read_plan(path: Path) -> Dict[int, List[int]]:
    plan: Dict[int, List[int]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            plan[int(row["episode_index"])] = [int(v) for v in row["frame_indices"]]
    return plan


def _fps(info: Dict[str, Any], video_key: str) -> float:
    if "fps" in info:
        return float(info["fps"])
    return float(info["features"][video_key]["info"]["video.fps"])


def _load_mano_records(
    dataset: Path,
    episode_rows: List[Dict[str, Any]],
    plan: Dict[int, List[int]],
) -> Dict[Tuple[int, int], Dict[str, Any]]:
    """Read only selected MANO rows, grouping all requests by Parquet shard."""

    routes: Dict[Path, set[Tuple[int, int]]] = defaultdict(set)
    for row in episode_rows:
        episode = int(row["episode_index"])
        if episode not in plan:
            continue
        data_path = (
            dataset
            / "data"
            / f"chunk-{int(row['data/chunk_index']):03d}"
            / f"file-{int(row['data/file_index']):03d}.parquet"
        )
        routes[data_path].update((episode, frame) for frame in plan[episode])

    desired = [
        "episode_index",
        "frame_index",
        "state_mask",
        "observation.state",
        "fov",
        "intrinsics",
        "extrinsics_w2c",
        "left_transl_world",
        "left_orient_world",
        "left_hand_pose",
        "right_transl_world",
        "right_orient_world",
        "right_hand_pose",
    ]
    records: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for data_path, targets in routes.items():
        parquet = pq.ParquetFile(data_path)
        available = set(parquet.schema_arrow.names)
        missing = sorted(set(desired) - available - {"intrinsics"})
        if missing:
            raise ValueError(f"MANO overlay 缺少字段 {missing}: {data_path}")
        table = parquet.read(columns=[name for name in desired if name in available])
        episode_values = np.asarray(table["episode_index"].to_pylist(), dtype=np.int64)
        frame_values = np.asarray(table["frame_index"].to_pylist(), dtype=np.int64)
        selected = [
            index
            for index, key in enumerate(zip(episode_values.tolist(), frame_values.tolist()))
            if key in targets
        ]
        if not selected:
            continue
        subset = table.take(pa.array(selected, type=pa.int64()))
        for row in subset.to_pylist():
            records[(int(row["episode_index"]), int(row["frame_index"]))] = row
    return records


def _write_evidence_gallery(
    output: Path,
    contact_frames: Dict[int, List[tuple[int, Path]]],
    manifest: List[Dict[str, Any]],
) -> None:
    by_episode: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for row in manifest:
        by_episode[int(row["episode_index"])].append(row)
    cards = []
    for episode_index in sorted(contact_frames):
        contact_name = f"episode-{episode_index:06d}-contact-sheet.jpg"
        source = by_episode[episode_index][0]["source_video"] if by_episode[episode_index] else "—"
        cards.append(
            f"""<article class="evidence" data-search="{episode_index} {html.escape(source)}">
            <a href="{contact_name}"><img loading="lazy" src="{contact_name}"
            alt="Episode {episode_index} contact sheet"></a>
            <div><strong>Episode {episode_index}</strong><span>{
                len(contact_frames[episode_index])
            } 帧</span></div><p title="{html.escape(source)}">{html.escape(source)}</p></article>"""
        )
    document = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>EgoQC Evidence</title>
<style>:root{{--bg:#f4f6f4;--surface:#fff;--ink:#18211b;--muted:#6b756e;--line:#dce3dd}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font-family:Inter,ui-sans-serif,system-ui,sans-serif}}
main{{max-width:1400px;margin:auto;padding:32px 24px 60px}}h1{{font-size:30px;margin:7px 0}}.eyebrow{{font-size:12px;color:var(--muted);letter-spacing:.08em;text-transform:uppercase}}
.toolbar{{display:flex;align-items:center;justify-content:space-between;gap:16px;margin:24px 0}}.stats{{color:var(--muted)}}
input{{font:inherit;border:1px solid var(--line);border-radius:8px;background:var(--surface);padding:9px 11px;min-width:260px}}
.gallery{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px}}.evidence{{background:var(--surface);border:1px solid var(--line);border-radius:10px;overflow:hidden}}
.evidence img{{display:block;width:100%;aspect-ratio:16/10;object-fit:cover;background:#18211b}}.evidence div{{display:flex;justify-content:space-between;padding:12px 13px 0}}
.evidence span,.evidence p{{font-size:11px;color:var(--muted)}}.evidence p{{padding:0 13px 12px;margin:6px 0 0;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}
.empty{{color:var(--muted);padding:30px}}@media(max-width:900px){{.gallery{{grid-template-columns:repeat(2,1fr)}}}}
@media(max-width:580px){{main{{padding:20px 14px}}.gallery{{grid-template-columns:1fr}}.toolbar{{align-items:stretch;flex-direction:column}}input{{min-width:0;width:100%}}}}
</style></head><body><main><div class="eyebrow">EgoQC-Lite · sampled visual evidence</div>
<h1>手部数据证据画廊</h1><div class="toolbar"><div class="stats">{
        len(contact_frames)
    } episodes · {len(manifest)} sampled frames · <a href="review.html">进入人工审核</a> · <a href="episodes-vlc.xspf">VLC 播放列表</a></div>
<input id="search" type="search" placeholder="搜索 episode 或视频路径" aria-label="搜索证据"></div>
<section class="gallery" id="gallery">{''.join(cards) or '<p class="empty">没有生成证据图</p>'}</section>
</main><script>const search=document.getElementById("search");const cards=[...document.querySelectorAll(".evidence")];
search.addEventListener("input",()=>{{const q=search.value.trim().toLowerCase();cards.forEach(card=>card.hidden=q&&!card.dataset.search.toLowerCase().includes(q));}});</script>
</body></html>"""
    path = output / "index.html"
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(document, encoding="utf-8")
    temporary.replace(path)


def _write_vlc_playlist(output: Path, routes: Dict[int, Dict[str, Any]]) -> Path:
    xspf = "http://xspf.org/ns/0/"
    vlc = "http://www.videolan.org/vlc/playlist/ns/0/"
    ET.register_namespace("", xspf)
    ET.register_namespace("vlc", vlc)
    playlist = ET.Element(f"{{{xspf}}}playlist", {"version": "1"})
    ET.SubElement(playlist, f"{{{xspf}}}title").text = "EgoQC sampled episodes"
    track_list = ET.SubElement(playlist, f"{{{xspf}}}trackList")
    for item_id, episode_index in enumerate(sorted(routes)):
        route = routes[episode_index]
        track = ET.SubElement(track_list, f"{{{xspf}}}track")
        ET.SubElement(track, f"{{{xspf}}}location").text = route["video_path"].as_uri()
        ET.SubElement(track, f"{{{xspf}}}title").text = f"Episode {episode_index}"
        ET.SubElement(track, f"{{{xspf}}}annotation").text = route["source_video"]
        duration_ms = max(0, round((route["stop_time"] - route["start_time"]) * 1000))
        ET.SubElement(track, f"{{{xspf}}}duration").text = str(duration_ms)
        extension = ET.SubElement(
            track,
            f"{{{xspf}}}extension",
            {"application": "http://www.videolan.org/vlc/playlist/0"},
        )
        ET.SubElement(extension, f"{{{vlc}}}id").text = str(item_id)
        ET.SubElement(extension, f"{{{vlc}}}option").text = f"start-time={route['start_time']:.6f}"
        ET.SubElement(extension, f"{{{vlc}}}option").text = f"stop-time={route['stop_time']:.6f}"
    extension = ET.SubElement(
        playlist,
        f"{{{xspf}}}extension",
        {"application": "http://www.videolan.org/vlc/playlist/0"},
    )
    for item_id in range(len(routes)):
        ET.SubElement(extension, f"{{{vlc}}}item", {"tid": str(item_id)})
    path = output / "episodes-vlc.xspf"
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    ET.ElementTree(playlist).write(temporary, encoding="utf-8", xml_declaration=True)
    temporary.replace(path)
    return path


def _write_review_page(
    dataset: Path,
    output: Path,
    contact_frames: Dict[int, List[tuple[int, Path]]],
    manifest: List[Dict[str, Any]],
    routes: Dict[int, Dict[str, Any]],
) -> Path:
    by_episode: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for row in manifest:
        by_episode[int(row["episode_index"])].append(row)
    records = []
    for episode_index in sorted(contact_frames):
        route = routes[episode_index]
        records.append(
            {
                "episode_index": episode_index,
                "contact_sheet": f"episode-{episode_index:06d}-contact-sheet.jpg",
                "sampled_frames": [frame for frame, _ in contact_frames[episode_index]],
                "source_video": route["source_video"],
                "start_time": route["start_time"],
                "stop_time": route["stop_time"],
                "mano_rendered": sum(row["mano_status"] == "rendered" for row in by_episode[episode_index]),
                "mano_failed": sum(row["mano_status"] == "failed" for row in by_episode[episode_index]),
            }
        )
    payload = json.dumps(records, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    dataset_json = json.dumps(str(dataset), ensure_ascii=False).replace("</", "<\\/")
    document = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>EgoQC 人工抽检</title>
<style>:root{{--bg:#edf1ee;--surface:#fff;--ink:#172019;--muted:#68736b;--line:#d7dfd9;--green:#1e7650;--amber:#a2691d;--red:#ba433d;--blue:#315f91}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font-family:Inter,system-ui,sans-serif}}main{{max-width:1320px;margin:auto;padding:28px}}
header{{display:flex;justify-content:space-between;gap:20px;align-items:end;flex-wrap:wrap}}h1{{margin:5px 0;font-size:30px}}.muted{{color:var(--muted);font-size:12px}}
.toolbar,.card{{background:var(--surface);border:1px solid var(--line);border-radius:12px}}.toolbar{{display:flex;gap:10px;align-items:center;flex-wrap:wrap;padding:12px;margin:20px 0}}
button,input{{font:inherit;border:1px solid var(--line);border-radius:8px;background:white;padding:9px 12px}}button{{cursor:pointer}}button.primary{{background:var(--ink);color:white}}
.progress{{margin-left:auto;font-variant-numeric:tabular-nums}}.card{{overflow:hidden}}.card img{{display:block;width:100%;max-height:68vh;object-fit:contain;background:#151916}}
.meta{{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;padding:15px 18px;border-top:1px solid var(--line)}}.meta b{{display:block;font-size:12px}}.meta span{{font-size:11px;color:var(--muted)}}
.decisions{{display:flex;gap:9px;padding:0 18px 14px;flex-wrap:wrap}}.decision.selected{{color:white}}[data-value=pass].selected{{background:var(--green)}}[data-value=minor].selected{{background:var(--amber)}}[data-value=fail].selected{{background:var(--red)}}[data-value=unsure].selected{{background:var(--blue)}}
#note{{margin:0 18px 18px;width:calc(100% - 36px)}}.hint{{padding:12px 18px;border-top:1px solid var(--line)}}@media(max-width:700px){{main{{padding:14px}}.meta{{grid-template-columns:1fr 1fr}}.progress{{width:100%;margin:0}}}}
</style></head><body><main><header><div><div class="muted">EgoQC-Lite · Human review</div><h1>人工抽检工作台</h1><div class="muted">{html.escape(str(dataset))}</div></div><a href="index.html">返回画廊</a></header>
<section class="toolbar"><button id="prev">← 上一个</button><button id="next">下一个 →</button><input id="reviewer" placeholder="审核人"><button id="import">导入 JSONL</button><input id="file" type="file" accept=".jsonl" hidden><button id="export" class="primary">导出审核结果</button><span class="progress" id="progress"></span></section>
<article class="card"><img id="image" alt="抽检 contact sheet"><div class="meta"><div><span>Episode</span><b id="episode"></b></div><div><span>采样帧</span><b id="frames"></b></div><div><span>源视频区间</span><b id="time"></b></div><div><span>MANO overlay</span><b id="mano"></b></div></div>
<div class="decisions"><button class="decision" data-value="pass">1 · 通过</button><button class="decision" data-value="minor">2 · 轻微问题</button><button class="decision" data-value="fail">3 · 失败</button><button class="decision" data-value="unsure">4 · 不确定</button></div><input id="note" placeholder="备注：错位、抖动、遮挡、左右手错误……"><div class="hint muted">键盘：←/→ 切换；1/2/3/4 判定。连续运动请打开 <a href="episodes-vlc.xspf">VLC 播放列表</a>。</div></article>
</main><script>
const records={payload};const dataset={dataset_json};const key="egoqc-review:"+dataset;let index=0;let reviews=JSON.parse(localStorage.getItem(key)||"{{}}");
const $=id=>document.getElementById(id);const save=()=>localStorage.setItem(key,JSON.stringify(reviews));
function current(){{return records[index]}}function render(){{const r=current();if(!r)return;$("image").src=r.contact_sheet;$("episode").textContent=r.episode_index;$("frames").textContent=r.sampled_frames.join(", ");$("time").textContent=r.start_time.toFixed(2)+"–"+r.stop_time.toFixed(2)+" s";$("mano").textContent=r.mano_rendered+" rendered / "+r.mano_failed+" failed";const review=reviews[r.episode_index]||{{}};$("note").value=review.note||"";document.querySelectorAll(".decision").forEach(b=>b.classList.toggle("selected",b.dataset.value===review.decision));const done=Object.values(reviews).filter(v=>v.decision).length;$("progress").textContent=(index+1)+" / "+records.length+" · 已审核 "+done;}}
function decide(value){{const r=current();reviews[r.episode_index]={{...(reviews[r.episode_index]||{{}}),decision:value,note:$("note").value,reviewed_at:new Date().toISOString()}};save();render();}}
document.querySelectorAll(".decision").forEach(b=>b.onclick=()=>decide(b.dataset.value));$("note").onchange=()=>{{const r=current();reviews[r.episode_index]={{...(reviews[r.episode_index]||{{}}),note:$("note").value}};save();}};
$("prev").onclick=()=>{{index=Math.max(0,index-1);render()}};$("next").onclick=()=>{{index=Math.min(records.length-1,index+1);render()}};document.onkeydown=e=>{{if(e.target.tagName==="INPUT")return;if(e.key==="ArrowLeft")$("prev").click();if(e.key==="ArrowRight")$("next").click();if("1234".includes(e.key))decide(["pass","minor","fail","unsure"][+e.key-1]);}};
$("export").onclick=()=>{{const reviewer=$("reviewer").value.trim();const lines=records.filter(r=>reviews[r.episode_index]?.decision).map(r=>JSON.stringify({{dataset_root:dataset,episode_index:r.episode_index,source_video:r.source_video,start_time:r.start_time,stop_time:r.stop_time,reviewer,...reviews[r.episode_index]}}));const blob=new Blob([lines.join("\\n")+(lines.length?"\\n":"")],{{type:"application/x-ndjson"}});const a=document.createElement("a");a.href=URL.createObjectURL(blob);a.download="human-reviews.jsonl";a.click();URL.revokeObjectURL(a.href);}};
$("import").onclick=()=>$("file").click();$("file").onchange=async e=>{{const text=await e.target.files[0].text();for(const line of text.split(/\\r?\\n/)){{if(!line.trim())continue;const row=JSON.parse(line);reviews[row.episode_index]={{decision:row.decision,note:row.note||"",reviewed_at:row.reviewed_at}};}}save();render();}};render();
</script></body></html>"""
    path = output / "review.html"
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(document, encoding="utf-8")
    temporary.replace(path)
    return path


def extract_samples(
    dataset: Path,
    plan_path: Path,
    output: Path,
    video_key: str = "observation.images.ego",
    jpeg_quality: int = 88,
    mano_renderer: Optional[ManoOverlayRenderer] = None,
    annotated_root: Optional[Path] = None,
) -> Dict[str, Any]:
    dataset = dataset.resolve()
    output.mkdir(parents=True, exist_ok=True)
    plan = _read_plan(plan_path)
    info = json.loads((dataset / "meta" / "info.json").read_text(encoding="utf-8"))
    fps = _fps(info, video_key)
    rows = load_episode_index(dataset).to_pylist()
    mano_records = _load_mano_records(dataset, rows, plan) if mano_renderer else {}
    grouped: Dict[Path, Dict[int, List[tuple[int, int]]]] = defaultdict(lambda: defaultdict(list))
    episode_routes: Dict[int, Dict[str, Any]] = {}

    for row in rows:
        ep = int(row["episode_index"])
        if ep not in plan:
            continue
        video_path = dataset / "videos" / video_key / f"chunk-{int(row[f'videos/{video_key}/chunk_index']):03d}" / f"file-{int(row[f'videos/{video_key}/file_index']):03d}.mp4"
        start_time = float(row[f"videos/{video_key}/from_timestamp"])
        stop_time = float(row[f"videos/{video_key}/to_timestamp"])
        start = int(round(start_time * fps))
        episode_routes[ep] = {
            "video_path": video_path.resolve(),
            "source_video": str(video_path.relative_to(dataset)),
            "start_time": start_time,
            "stop_time": stop_time,
        }
        for local_index in plan[ep]:
            grouped[video_path][start + local_index].append((ep, local_index))

    manifest = []
    contact_frames: Dict[int, List[tuple[int, Path]]] = defaultdict(list)
    overlay_failures = 0
    for video_path, targets in grouped.items():
        if not video_path.exists():
            continue
        pending = set(targets)
        with av.open(str(video_path)) as container:
            stream = container.streams.video[0]
            for decoded_index, frame in enumerate(container.decode(stream)):
                if decoded_index not in pending:
                    continue
                image = frame.to_image()
                for ep, local_index in targets[decoded_index]:
                    ep_dir = output / f"episode-{ep:06d}"
                    ep_dir.mkdir(exist_ok=True)
                    path = ep_dir / f"frame-{local_index:06d}.jpg"
                    image.save(path, quality=jpeg_quality, optimize=True)
                    display_path = path
                    evidence: Dict[str, Any] = {
                        "episode_index": ep,
                        "frame_index": local_index,
                        "absolute_video_frame": decoded_index,
                        "source_video": str(video_path.relative_to(dataset)),
                        "image": str(path),
                        "mano_status": "disabled",
                    }
                    if mano_renderer:
                        try:
                            record = mano_records[(ep, local_index)]
                            overlay, metrics = mano_renderer.render(image, record)
                            overlay_path = ep_dir / f"frame-{local_index:06d}-mano.jpg"
                            overlay.save(overlay_path, quality=jpeg_quality, optimize=True)
                            display_path = overlay_path
                            evidence.update(
                                {
                                    "mano_status": "rendered",
                                    "overlay_image": str(overlay_path),
                                    "mano_metrics": metrics,
                                }
                            )
                        except Exception as error:  # visual evidence must not stop QC
                            overlay_failures += 1
                            evidence.update(
                                {
                                    "mano_status": "failed",
                                    "mano_error": f"{type(error).__name__}: {error}",
                                }
                            )
                    contact_frames[ep].append((local_index, display_path))
                    manifest.append(evidence)
                pending.remove(decoded_index)
                if not pending:
                    break

    for ep, frames in contact_frames.items():
        frames.sort()
        images = [Image.open(path).convert("RGB") for _, path in frames]
        if not images:
            continue
        thumb_w = 320
        thumb_h = max(1, round(images[0].height * thumb_w / images[0].width))
        cols = min(4, len(images))
        rows_count = math.ceil(len(images) / cols)
        sheet = Image.new("RGB", (cols * thumb_w, rows_count * (thumb_h + 28)), "#171a18")
        draw = ImageDraw.Draw(sheet)
        for idx, (frame_info, image) in enumerate(zip(frames, images)):
            x = (idx % cols) * thumb_w
            y = (idx // cols) * (thumb_h + 28)
            sheet.paste(image.resize((thumb_w, thumb_h)), (x, y))
            draw.text((x + 8, y + thumb_h + 6), f"frame {frame_info[0]}", fill="white")
        sheet.save(output / f"episode-{ep:06d}-contact-sheet.jpg", quality=90)
        for image in images:
            image.close()

    write_jsonl(output / "evidence_manifest.jsonl", manifest)
    if mano_renderer:
        (output / "mano_provenance.json").write_text(
            json.dumps(mano_renderer.provenance(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    _write_evidence_gallery(output, contact_frames, manifest)
    reviewed_routes = {episode: episode_routes[episode] for episode in contact_frames}
    review_path = write_review_workbench(
        dataset,
        output,
        contact_frames,
        manifest,
        reviewed_routes,
        quality_root=plan_path.parent,
        annotated_root=annotated_root,
    )
    vlc_path = _write_vlc_playlist(output, reviewed_routes)
    return {
        "videos_opened": len(grouped),
        "frames_extracted": len(manifest),
        "episodes": len(contact_frames),
        "gallery": str((output / "index.html").resolve()),
        "review": str(review_path.resolve()),
        "vlc_playlist": str(vlc_path.resolve()),
        "mano_enabled": mano_renderer is not None,
        "mano_frames_rendered": sum(row["mano_status"] == "rendered" for row in manifest),
        "mano_failures": overlay_failures,
    }
