from __future__ import annotations

import html
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Sequence, Union

from .provenance import config_hash


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_tuner(
    quality_root: Union[Path, Sequence[Path]],
    config: Dict[str, Any],
    output: Path,
) -> Dict[str, Any]:
    quality_roots = [quality_root] if isinstance(quality_root, Path) else list(quality_root)
    episodes: List[Dict[str, Any]] = []
    for root in quality_roots:
        episodes_path = root / "episodes.jsonl"
        if not episodes_path.exists():
            raise FileNotFoundError(episodes_path)
        for episode in _read_jsonl(episodes_path):
            episodes.append({"quality_root": str(root), **episode})
    payload = json.dumps(episodes, ensure_ascii=False, separators=(",", ":")).replace(
        "</", "<\\/"
    )
    config_payload = json.dumps(config, ensure_ascii=False, separators=(",", ":")).replace(
        "</", "<\\/"
    )
    document = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>EgoQC 参数调优</title>
<style>:root{{--bg:#f3f5f2;--card:#fff;--ink:#172019;--muted:#69736c;--line:#dbe1dc;--green:#24734c;--amber:#a36b20;--red:#b14740}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:14px ui-sans-serif,system-ui}}
main{{max-width:1380px;margin:auto;padding:28px}}h1{{margin:5px 0 8px;font-size:30px}}.muted{{color:var(--muted)}}
.grid{{display:grid;grid-template-columns:minmax(360px,0.9fr) minmax(480px,1.5fr);gap:15px;margin-top:22px}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:18px}}.controls{{max-height:72vh;overflow:auto}}
.row{{display:grid;grid-template-columns:1fr 120px;gap:12px;align-items:center;padding:9px 0;border-bottom:1px solid var(--line)}}
input[type=number]{{width:100%;padding:7px;border:1px solid var(--line);border-radius:7px}}button{{border:0;border-radius:8px;padding:10px 14px;background:var(--green);color:white;cursor:pointer}}
.cards{{display:grid;grid-template-columns:repeat(4,1fr);gap:9px;margin:14px 0}}.stat{{padding:13px;background:var(--bg);border-radius:9px}}.stat b{{display:block;font-size:22px}}
table{{width:100%;border-collapse:collapse;font-size:12px}}th,td{{padding:8px;border-top:1px solid var(--line);text-align:left}}code{{font-size:11px}}
.warn{{color:var(--amber)}}.error{{color:var(--red)}}@media(max-width:900px){{.grid{{grid-template-columns:1fr}}}}</style></head>
<body><main><div class="muted">EgoQC-Lite · offline calibration workbench</div><h1>交互式阈值调优</h1>
<p class="muted">只在浏览器内重算当前样本，不修改原始数据。确认后下载新配置，再用新 config hash 正式运行。</p>
<div class="grid"><section class="card controls"><h2>质量阈值</h2><div id="controls"></div>
<p><button id="download">下载版本化配置</button></p><p class="muted">当前 hash <code id="hash">{config_hash(config)}</code></p></section>
<section class="card"><h2>样本影响预览</h2><div class="cards"><div class="stat"><span>Episodes</span><b id="total">0</b></div>
<div class="stat"><span>无阈值触发</span><b id="clean">0</b></div><div class="stat"><span>Warning</span><b id="warning">0</b></div>
<div class="stat"><span>Error</span><b id="error">0</b></div></div><p class="muted" id="changed"></p>
<table><thead><tr><th>Episode</th><th>当前 tier</th><th>调参预览</th><th>触发指标</th></tr></thead><tbody id="rows"></tbody></table></section></div>
</main><script>const episodes={payload};let config={config_payload};const original=JSON.stringify(config);
const metricRules=[
 ["left_position_jitter_p99_mm","position_jitter_warning_m","position_jitter_error_m",1000],
 ["right_position_jitter_p99_mm","position_jitter_warning_m","position_jitter_error_m",1000],
 ["left_wrist_rotation_jitter_p99_deg","rotation_jitter_warning_deg","rotation_jitter_error_deg",1],
 ["right_wrist_rotation_jitter_p99_deg","rotation_jitter_warning_deg","rotation_jitter_error_deg",1],
 ["left_joint_rotation_jitter_p99_deg","joint_jitter_warning_deg","joint_jitter_error_deg",1],
 ["right_joint_rotation_jitter_p99_deg","joint_jitter_warning_deg","joint_jitter_error_deg",1],
 ["camera_translation_jitter_p99_mm","camera_jitter_warning_m","camera_jitter_error_m",1000],
 ["camera_rotation_jitter_p99_deg","camera_rotation_jitter_warning_deg","camera_rotation_jitter_error_deg",1]
];
const controls=document.getElementById("controls");Object.entries(config.thresholds).forEach(([key,value])=>{{if(typeof value!=="number")return;
 const row=document.createElement("label");row.className="row";row.innerHTML=`<span><code>${{key}}</code></span><input type="number" step="any" value="${{value}}" data-key="${{key}}">`;controls.appendChild(row);}});
function evaluate(ep){{let level=0,triggers=[];for(const [metric,warnKey,errorKey,scale] of metricRules){{const value=Number(ep.metrics?.[metric]);if(!Number.isFinite(value))continue;
 const warn=Number(config.thresholds[warnKey])*scale,error=Number(config.thresholds[errorKey])*scale;if(value>error){{level=2;triggers.push(`${{metric}}=${{value.toFixed(2)}}`);}}
 else if(value>warn){{level=Math.max(level,1);triggers.push(`${{metric}}=${{value.toFixed(2)}}`);}}}}return {{level,triggers}};}}
function canonical(value){{if(Array.isArray(value))return `[${{value.map(canonical).join(',')}}]`;if(value&&typeof value==='object')return `{{${{Object.keys(value).sort().map(key=>JSON.stringify(key)+':'+canonical(value[key])).join(',')}}}}`;return JSON.stringify(value);}}
async function updateHash(){{const bytes=new TextEncoder().encode(canonical(config));const digest=await crypto.subtle.digest("SHA-256",bytes);document.getElementById("hash").textContent=Array.from(new Uint8Array(digest)).map(x=>x.toString(16).padStart(2,"0")).join("");}}
function render(){{let clean=0,warning=0,error=0;const values=episodes.map(ep=>[ep,evaluate(ep)]);for(const [,r] of values){{r.level===0?clean++:r.level===1?warning++:error++;}}
 document.getElementById("total").textContent=episodes.length.toLocaleString();document.getElementById("clean").textContent=clean.toLocaleString();document.getElementById("warning").textContent=warning.toLocaleString();document.getElementById("error").textContent=error.toLocaleString();
 document.getElementById("changed").textContent=JSON.stringify(config)===original?"尚未修改阈值":"预览已变化；下载配置后必须以新版本重新 plan。";
 document.getElementById("rows").innerHTML=values.filter(([,r])=>r.level>0).slice(0,300).map(([ep,r])=>`<tr><td>${{ep.episode_index}}</td><td>${{ep.tier}}</td><td class="${{r.level===2?'error':'warn'}}">${{r.level===2?'error':'warning'}}</td><td><code>${{r.triggers.join('<br>')}}</code></td></tr>`).join("");updateHash();}}
controls.addEventListener("input",event=>{{const input=event.target;if(!input.dataset.key)return;config.thresholds[input.dataset.key]=Number(input.value);render();}});
document.getElementById("download").addEventListener("click",()=>{{const blob=new Blob([JSON.stringify(config,null,2)],{{type:"application/json"}});const a=document.createElement("a");a.href=URL.createObjectURL(blob);a.download=`egoqc-${{config.standard_version}}-tuned.json`;a.click();URL.revokeObjectURL(a.href);}});render();</script></body></html>"""
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    temporary.write_text(document, encoding="utf-8")
    temporary.replace(output)
    return {
        "output": str(output.resolve()),
        "quality_roots": [str(root.resolve()) for root in quality_roots],
        "episode_count": len(episodes),
        "config_hash": config_hash(config),
    }
