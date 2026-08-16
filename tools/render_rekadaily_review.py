#!/usr/bin/env python3
"""Render a compact offline review page from RekaDaily tar members."""

from __future__ import annotations

import argparse
import html
import json
import tarfile
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Dict, List

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from egoqc.adapters import RekaDailyRawAdapter


LABELS = {
    "candidate_for_hand_annotation": "CANDIDATE",
    "review_before_hand_annotation": "REVIEW",
    "screen_out_before_hand_annotation": "SCREEN OUT",
}


def _font(size: int) -> ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    ):
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def _open_container(stack: ExitStack, dataset: Path, row: Dict[str, Any]) -> av.container.InputContainer:
    adapter = RekaDailyRawAdapter()
    source_row = {**row, "src_ext": row.get("src_ext") or row.get("container")}
    tar_video = adapter._tar_video(dataset, source_row)
    if tar_video:
        shard, member = tar_video
        archive = stack.enter_context(tarfile.open(shard, "r"))
        stream = archive.extractfile(member)
        if stream is None:
            raise ValueError(f"cannot open tar member {member}")
        return stack.enter_context(av.open(stream))
    loose = adapter._loose_video(dataset, source_row)
    if loose and loose.suffix.lower() in {".mp4", ".mov", ".avi"}:
        return stack.enter_context(av.open(str(loose)))
    raise FileNotFoundError(str(row["video_id"]))


def _sample_frame(container: av.container.InputContainer, target: float) -> Image.Image:
    stream = container.streams.video[0]
    container.seek(max(0, int(target * av.time_base)), any_frame=False, backward=True)
    selected = None
    for frame in container.decode(stream):
        selected = frame
        if frame.time is None or float(frame.time) >= target - 0.05:
            break
    if selected is None:
        return Image.new("RGB", (480, 270), "#d8d8d2")
    return Image.fromarray(selected.to_ndarray(format="rgb24"))


def render_sheet(dataset: Path, record: Dict[str, Any], output: Path) -> None:
    width, height = 480, 270
    gutter, header = 8, 48
    canvas = Image.new("RGB", (width * 3 + gutter * 4, header + height * 2 + gutter * 3), "#ecece7")
    draw = ImageDraw.Draw(canvas)
    title_font, meta_font = _font(20), _font(13)
    decision = LABELS.get(record["decision"], record["decision"].upper())
    draw.text((gutter, 9), record["video_id"], fill="#11110f", font=title_font)
    draw.text((canvas.width - gutter, 15), decision, fill="#b34824", font=meta_font, anchor="ra")
    duration = float(record.get("duration_s") or 0)
    targets = np.linspace(0, max(0, duration - 1 / max(float(record.get("fps") or 30), 1)), 6)
    with ExitStack() as stack:
        container = _open_container(stack, dataset, record)
        for index, target in enumerate(targets):
            frame = ImageOps.fit(_sample_frame(container, float(target)), (width, height), Image.Resampling.LANCZOS)
            x = gutter + (index % 3) * (width + gutter)
            y = header + gutter + (index // 3) * (height + gutter)
            canvas.paste(frame, (x, y))
            ImageDraw.Draw(canvas).rectangle((x, y + height - 26, x + 83, y + height), fill="#11110f")
            ImageDraw.Draw(canvas).text((x + 7, y + height - 21), f"{target:7.1f}s", fill="white", font=meta_font)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=88, optimize=True)


def write_html(summary: Dict[str, Any], output: Path) -> None:
    rows: List[str] = []
    for record in summary["records"]:
        issues = ", ".join(issue["code"] for issue in record["issues"]) or "No cheap-stage issue"
        rows.append(f"""
<article class="item" data-decision="{html.escape(record['decision'])}">
  <img src="contact_sheets/{html.escape(record['video_id'])}.jpg" loading="lazy" alt="Sample frames for {html.escape(record['video_id'])}">
  <div class="body"><div class="title"><code>{html.escape(record['video_id'])}</code><b>{LABELS.get(record['decision'], record['decision'])}</b></div>
  <p>{html.escape(record['project'])} · {record['width']}×{record['height']} · {float(record['fps']):.3f} fps · {float(record['duration_s']) / 60:.1f} min</p>
  <dl><dt>Frame match</dt><dd>{record['counted_frames']} / {record['metadata_frames']}</dd><dt>Jitter max</dt><dd>{float(record['jitter_max_ms'] or 0):.3f} ms</dd><dt>Issues</dt><dd>{html.escape(issues)}</dd></dl></div>
</article>""")
    counts = summary["decision_counts"]
    output.write_text(f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>RekaDaily · QC review</title><style>
:root{{--paper:#f3f3ee;--ink:#151512;--line:#c9c9c0;--accent:#b34824;--muted:#66665f}}*{{box-sizing:border-box}}body{{margin:0;background:var(--paper);color:var(--ink);font:14px/1.45 ui-sans-serif,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}header{{position:sticky;top:0;z-index:2;display:grid;grid-template-columns:1fr auto;gap:20px;align-items:end;padding:22px 28px 18px;background:rgba(243,243,238,.94);border-bottom:1px solid var(--line);backdrop-filter:blur(8px)}}h1{{font-size:25px;line-height:1;margin:0 0 7px;letter-spacing:-.03em}}header p{{margin:0;color:var(--muted)}}nav{{display:flex;border:1px solid var(--line)}}button{{border:0;border-right:1px solid var(--line);background:transparent;padding:8px 12px;cursor:pointer}}button:last-child{{border:0}}button.active{{background:var(--ink);color:var(--paper)}}main{{max-width:1500px;margin:auto;padding:22px 28px 60px}}.item{{display:grid;grid-template-columns:minmax(460px,2fr) minmax(300px,1fr);gap:22px;padding:18px 0;border-top:1px solid var(--line)}}.item:first-child{{border-top:0}}.item img{{width:100%;display:block;border:1px solid var(--line)}}.body{{padding:4px 4px 0 0}}.title{{display:flex;justify-content:space-between;gap:12px;align-items:start}}code{{font-size:12px;word-break:break-all}}b{{font-size:11px;color:var(--accent);white-space:nowrap}}p{{color:var(--muted)}}dl{{display:grid;grid-template-columns:110px 1fr;margin:24px 0 0;border-top:1px solid var(--line)}}dt,dd{{margin:0;padding:8px 0;border-bottom:1px solid var(--line)}}dt{{color:var(--muted)}}dd{{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px}}[hidden]{{display:none!important}}@media(max-width:800px){{header{{position:static;grid-template-columns:1fr}}nav{{overflow:auto}}main{{padding:12px 16px}}.item{{grid-template-columns:1fr}}}}
</style></head><body><header><div><h1>RekaDaily / QC review</h1><p>{summary['videos']} videos · {summary['duration_hours']:.2f} h · exact frames {summary['exact_frame_matches']}/{summary['videos']}</p></div><nav><button class="active" data-filter="all">All {summary['videos']}</button><button data-filter="candidate_for_hand_annotation">Candidate {counts.get('candidate_for_hand_annotation',0)}</button><button data-filter="review_before_hand_annotation">Review {counts.get('review_before_hand_annotation',0)}</button><button data-filter="screen_out_before_hand_annotation">Screen out {counts.get('screen_out_before_hand_annotation',0)}</button></nav></header><main>{''.join(rows)}</main><script>document.querySelectorAll('button').forEach(b=>b.onclick=()=>{{document.querySelectorAll('button').forEach(x=>x.classList.remove('active'));b.classList.add('active');document.querySelectorAll('.item').forEach(x=>x.hidden=b.dataset.filter!=='all'&&x.dataset.decision!==b.dataset.filter)}})</script></body></html>""", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("summary", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    for record in summary["records"]:
        render_sheet(args.dataset, record, args.output / "contact_sheets" / f"{record['video_id']}.jpg")
        print(record["video_id"], flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    write_html(summary, args.output / "review.html")


if __name__ == "__main__":
    main()
