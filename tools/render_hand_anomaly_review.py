#!/usr/bin/env python3
"""Render short annotated clips and an offline human-review page for hand anomalies."""

from __future__ import annotations

import argparse
import bisect
import html
import json
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, List, Sequence

import av
import cv2

from egoqc.hand_screen import _open_video


def _nearest_sample(samples: Sequence[Dict[str, Any]], times: Sequence[float], timestamp: float) -> Dict[str, Any]:
    index = min(max(0, bisect.bisect_left(times, timestamp)), len(samples) - 1)
    if index and abs(times[index - 1] - timestamp) < abs(times[index] - timestamp):
        index -= 1
    return samples[index]


def _events(report: Dict[str, Any], maximum_per_type: int) -> List[Dict[str, Any]]:
    metrics = report["metrics"]
    events: List[Dict[str, Any]] = []
    for kind, source in (
        ("hand_absent", metrics.get("long_no_hand_segments", [])),
        ("persistent_extra_hands", metrics.get("suspected_extra_hand_segments", [])),
    ):
        ranked = sorted(source, key=lambda item: float(item["duration_s"]), reverse=True)
        for index, segment in enumerate(ranked[:maximum_per_type]):
            events.append({
                "event_id": f"{report['video_id']}--{kind}-{index:02d}",
                "video_id": report["video_id"],
                "kind": kind,
                "rank": index + 1,
                "start_s": float(segment["start_s"]),
                "end_s": float(segment["end_s"]),
                "duration_s": float(segment["duration_s"]),
            })
    return events


def _draw_overlay(
    image: Any,
    sample: Dict[str, Any],
    event: Dict[str, Any],
    timestamp: float,
    detector_width: int,
    detector_height: int,
) -> Any:
    height, width = image.shape[:2]
    scale_x, scale_y = width / detector_width, height / detector_height
    for x1, y1, x2, y2, confidence, hand_class in sample.get("detections", []):
        color = (109, 128, 23) if int(hand_class) == 0 else (36, 72, 179)
        thickness = 4 if confidence >= 0.7 else 2
        p1 = (round(x1 * scale_x), round(y1 * scale_y))
        p2 = (round(x2 * scale_x), round(y2 * scale_y))
        cv2.rectangle(image, p1, p2, color, thickness, cv2.LINE_AA)
        cv2.putText(
            image,
            f"{'L' if int(hand_class) == 0 else 'R'} {confidence:.2f}",
            (p1[0] + 3, max(20, p1[1] - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    active = event["start_s"] <= timestamp <= event["end_s"]
    if active:
        cv2.rectangle(image, (4, 4), (width - 5, height - 5), (28, 45, 210), 7)
    panel_height = 78
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (width, panel_height), (16, 16, 14), -1)
    cv2.addWeighted(overlay, 0.82, image, 0.18, 0, image)
    label = "NO HAND > 1s" if event["kind"] == "hand_absent" else "PERSISTENT >2 HANDS"
    cv2.putText(image, label, (18, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    state = "ANOMALY ACTIVE" if active else "CONTEXT"
    cv2.putText(
        image,
        f"t={timestamp:.2f}s  detected={sample['hand_count']}  {state}",
        (18, 62),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (82, 184, 255) if active else (205, 205, 198),
        2,
        cv2.LINE_AA,
    )
    return image


def render_event_clip(
    dataset: Path,
    report: Dict[str, Any],
    samples: Sequence[Dict[str, Any]],
    event: Dict[str, Any],
    output: Path,
    context_s: float,
) -> Dict[str, Any]:
    metadata = report["metadata"]
    clip_start = max(0.0, event["start_s"] - context_s)
    clip_end = min(float(metadata["duration_s"]), event["end_s"] + context_s)
    sample_times = [float(sample["time_s"]) for sample in samples]
    output.parent.mkdir(parents=True, exist_ok=True)
    rendered = 0
    with _open_video(dataset, metadata) as (source, source_uri):
        source_stream = source.streams.video[0]
        fps = float(source_stream.average_rate or metadata.get("fps") or 30.0)
        width = int(source_stream.codec_context.width)
        height = int(source_stream.codec_context.height)
        source.seek(max(0, int((clip_start - 1.0 / fps) * av.time_base)), any_frame=False, backward=True)
        with av.open(str(output), "w", options={"movflags": "+faststart"}) as sink:
            target = sink.add_stream("libx264", rate=Fraction(str(fps)).limit_denominator(100000))
            target.width = width
            target.height = height
            target.pix_fmt = "yuv420p"
            target.options = {"crf": "20", "preset": "veryfast"}
            for frame in source.decode(source_stream):
                timestamp = float(frame.time) if frame.time is not None else rendered / fps
                if timestamp + 0.5 / fps < clip_start:
                    continue
                if timestamp > clip_end + 0.5 / fps:
                    break
                sample = _nearest_sample(samples, sample_times, timestamp)
                detector_width = int(sample.get("image_width") or width)
                detector_height = int(sample.get("image_height") or height)
                image = frame.to_ndarray(format="bgr24")
                image = _draw_overlay(image, sample, event, timestamp, detector_width, detector_height)
                rendered_frame = av.VideoFrame.from_ndarray(image, format="bgr24")
                for packet in target.encode(rendered_frame):
                    sink.mux(packet)
                rendered += 1
            for packet in target.encode():
                sink.mux(packet)
    return {
        **event,
        "clip_start_s": clip_start,
        "clip_end_s": clip_end,
        "clip_duration_s": clip_end - clip_start,
        "rendered_frames": rendered,
        "source_uri": source_uri,
        "clip": str(output),
    }


def write_review(events: Sequence[Dict[str, Any]], output: Path) -> None:
    rows = []
    for event in events:
        title = "手连续离画" if event["kind"] == "hand_absent" else "疑似第二人手"
        rows.append(f"""
<article class="case" data-state="pending" data-kind="{event['kind']}">
  <div class="media"><video controls preload="metadata" src="clips/{html.escape(Path(event['clip']).name)}"></video></div>
  <div class="detail"><div class="case-head"><div><code>{html.escape(event['video_id'])}</code><h2>{title}</h2></div><span>{event['duration_s']:.2f}s</span></div>
  <dl><dt>异常区间</dt><dd>{event['start_s']:.2f}–{event['end_s']:.2f}s</dd><dt>上下文</dt><dd>{event['clip_start_s']:.2f}–{event['clip_end_s']:.2f}s</dd><dt>排序</dt><dd>#{event['rank']}</dd></dl>
  <div class="choices" data-event="{html.escape(event['event_id'])}"><button data-value="confirmed">真实问题</button><button data-value="false_positive">模型误报</button><button data-value="unsure">不确定</button></div>
  <label>备注<textarea placeholder="写明画面内容、时间点和判断依据"></textarea></label></div>
</article>""")
    payload = json.dumps(list(events), ensure_ascii=False).replace("</", "<\\/")
    output.write_text(f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>EgoQC · 手部异常复检</title><style>
:root{{--paper:#f2f2ed;--ink:#171714;--muted:#66665e;--line:#c8c8be;--accent:#b44827;--ok:#18745f}}*{{box-sizing:border-box}}body{{margin:0;background:var(--paper);color:var(--ink);font:14px/1.45 ui-sans-serif,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}header{{position:sticky;top:0;z-index:3;display:grid;grid-template-columns:1fr auto;align-items:end;gap:20px;padding:18px 28px;background:rgba(242,242,237,.95);border-bottom:1px solid var(--line);backdrop-filter:blur(8px)}}h1{{margin:0 0 5px;font-size:24px;letter-spacing:-.03em}}header p{{margin:0;color:var(--muted)}}.tools{{display:flex;gap:6px}}button{{border:1px solid var(--line);background:transparent;padding:8px 11px;cursor:pointer;color:inherit}}button:hover{{border-color:var(--ink)}}button:active{{transform:translateY(1px)}}button.active{{background:var(--ink);color:var(--paper);border-color:var(--ink)}}main{{max-width:1500px;margin:auto;padding:12px 28px 64px}}.case{{display:grid;grid-template-columns:minmax(520px,1.65fr) minmax(320px,.75fr);gap:24px;padding:22px 0;border-bottom:1px solid var(--line)}}video{{display:block;width:100%;background:#111;max-height:600px}}.detail{{padding:3px 3px 0 0}}.case-head{{display:flex;justify-content:space-between;gap:12px}}code{{font-size:11px;color:var(--muted);word-break:break-all}}h2{{margin:6px 0 0;font-size:22px}}.case-head span{{color:var(--accent);font:700 12px ui-monospace,monospace}}dl{{display:grid;grid-template-columns:95px 1fr;margin:22px 0;border-top:1px solid var(--line)}}dt,dd{{margin:0;padding:7px 0;border-bottom:1px solid var(--line)}}dt{{color:var(--muted)}}dd{{font-family:ui-monospace,monospace;font-size:12px}}.choices{{display:grid;grid-template-columns:repeat(3,1fr);gap:6px;margin:16px 0}}.choices button.selected[data-value=confirmed]{{background:var(--accent);color:white;border-color:var(--accent)}}.choices button.selected[data-value=false_positive]{{background:var(--ok);color:white;border-color:var(--ok)}}.choices button.selected[data-value=unsure]{{background:var(--ink);color:white;border-color:var(--ink)}}label{{display:grid;gap:6px;color:var(--muted)}}textarea{{width:100%;min-height:88px;resize:vertical;border:1px solid var(--line);background:transparent;padding:9px;font:inherit;color:var(--ink)}}.case[data-state=reviewed]{{opacity:.62}}[hidden]{{display:none!important}}@media(max-width:850px){{header{{position:static;grid-template-columns:1fr}}.tools{{overflow:auto}}main{{padding:8px 14px}}.case{{grid-template-columns:1fr}}}}
</style></head><body><header><div><h1>手部异常人工复检</h1><p>{len(events)} 个事件 · 红框表示系统异常区间 · 结果仅保存在浏览器</p></div><div class="tools"><button data-filter="all" class="active">全部</button><button data-filter="pending">待判断</button><button data-filter="reviewed">已判断</button><button id="export">导出 JSONL</button></div></header><main>{''.join(rows)}</main><script>
const events={payload};const key='egoqc-hand-anomaly-review-v1';let reviews={{}};try{{reviews=JSON.parse(localStorage.getItem(key)||'{{}}')}}catch(e){{reviews={{}}}}function save(){{localStorage.setItem(key,JSON.stringify(reviews))}}function render(){{document.querySelectorAll('.choices').forEach(group=>{{const id=group.dataset.event,value=reviews[id]?.decision;group.closest('.case').dataset.state=value?'reviewed':'pending';group.querySelectorAll('button').forEach(b=>b.classList.toggle('selected',b.dataset.value===value));const textarea=group.parentElement.querySelector('textarea');textarea.value=reviews[id]?.note||'';textarea.oninput=()=>{{reviews[id]={{...(reviews[id]||{{}}),note:textarea.value}};save()}};group.querySelectorAll('button').forEach(b=>b.onclick=()=>{{reviews[id]={{...(reviews[id]||{{}}),decision:b.dataset.value,updated_at:new Date().toISOString()}};save();render()}})}})}}render();document.querySelectorAll('[data-filter]').forEach(b=>b.onclick=()=>{{document.querySelectorAll('[data-filter]').forEach(x=>x.classList.remove('active'));b.classList.add('active');document.querySelectorAll('.case').forEach(c=>c.hidden=b.dataset.filter!=='all'&&c.dataset.state!==b.dataset.filter)}});document.getElementById('export').onclick=()=>{{const lines=events.map(e=>JSON.stringify({{...e,...(reviews[e.event_id]||{{}})}})).join('\n')+'\n';const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([lines],{{type:'application/x-ndjson'}}));a.download='hand-anomaly-reviews.jsonl';a.click();URL.revokeObjectURL(a.href)}};
</script></body></html>""", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("hand_screen_root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-per-type", type=int, default=2)
    parser.add_argument("--context-s", type=float, default=2.0)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rendered_events: List[Dict[str, Any]] = []
    for report_path in sorted(args.hand_screen_root.glob("*/hand-screen.json")):
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report["metrics"]["provisional_decision"] == "candidate_for_mano":
            continue
        samples = [
            json.loads(line)
            for line in report_path.with_name("hand-samples.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        for event in _events(report, args.max_per_type):
            clip_path = args.output / "clips" / f"{event['event_id']}.mp4"
            rendered_events.append(
                render_event_clip(args.dataset, report, samples, event, clip_path, args.context_s)
            )
            print(event["event_id"], flush=True)
    (args.output / "events.json").write_text(
        json.dumps(rendered_events, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_review(rendered_events, args.output / "review.html")
    print(json.dumps({"events": len(rendered_events), "review": str(args.output / "review.html")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
