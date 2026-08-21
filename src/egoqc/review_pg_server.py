from __future__ import annotations

import json
import mimetypes
from functools import partial
from http.cookies import SimpleCookie
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional
from urllib.parse import parse_qs, unquote, urlparse

from .feishu_auth import (
    FeishuAuthConfig,
    FeishuAuthError,
    authorization_url,
    exchange_code,
    fetch_user_info,
)
from .review_db import ReviewConflict, ReviewStore


LOGIN_HTML = """<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>EgoQC 登录</title>
<style>body{margin:0;background:#ecece7;color:#181815;font:15px/1.5 ui-sans-serif,-apple-system,sans-serif;display:grid;place-items:center;min-height:100vh}.box{width:min(420px,calc(100vw - 40px));border:1px solid #c9c9bf;padding:34px;background:#f8f8f3}h1{margin:0 0 10px;font-size:26px}p{color:#66665e;margin:0 0 28px}a{display:block;text-align:center;padding:11px;background:#181815;color:#f8f8f3;text-decoration:none}</style>
</head><body><main class="box"><h1>EgoQC 复检台</h1><p>使用企业飞书身份登录。任务将按视频分配到个人账号。</p><a href="/auth/login">使用飞书登录</a></main></body></html>"""


REVIEW_HTML = r"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>EgoQC Review</title><style>
:root{--bg:#ecece7;--card:#f8f8f3;--ink:#181815;--muted:#6b6b62;--line:#c9c9bf;--red:#b7492a;--green:#18725e;--blue:#365f8d}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 ui-sans-serif,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
header{position:sticky;top:0;z-index:4;padding:17px 24px;background:rgba(236,236,231,.96);border-bottom:1px solid var(--line);backdrop-filter:blur(10px)}
	.top{display:flex;align-items:end;justify-content:space-between;gap:20px;max-width:1880px;margin:auto}h1{font-size:24px;line-height:1;margin:0 0 7px;letter-spacing:-.04em}.sub{color:var(--muted)}
.identity{display:flex;align-items:center;gap:7px}.identity input{width:170px}.identity a{color:var(--muted);padding:7px}.status{display:inline-block;width:7px;height:7px;background:var(--green);border-radius:50%;margin-right:6px}.status.off{background:var(--red)}
	.bar{max-width:1880px;margin:14px auto 0;display:flex;gap:7px;align-items:center;overflow:auto}.stat{padding:7px 10px;border:1px solid var(--line);white-space:nowrap}.stat strong{font:700 13px ui-monospace,monospace;margin-left:8px}
button,input,textarea,select{font:inherit;color:inherit;border:1px solid var(--line);background:transparent;padding:8px 10px}button{cursor:pointer;transition:transform 90ms ease,border-color 90ms ease,background-color 90ms ease}button:hover:not(:disabled){border-color:var(--ink)}button:active:not(:disabled){transform:translateY(1px)}button:focus-visible,input:focus-visible,textarea:focus-visible,select:focus-visible{outline:2px solid var(--blue);outline-offset:2px}button:disabled{opacity:.38;cursor:not-allowed}button.active{background:var(--ink);color:var(--card);border-color:var(--ink)}
	main{max-width:1880px;margin:auto;padding:14px 24px 70px}.empty{padding:90px 20px;text-align:center;color:var(--muted)}
	.review-nav{display:flex;align-items:center;justify-content:space-between;gap:12px;padding:12px 0 4px}.review-position{color:var(--muted);font:12px ui-monospace,monospace}.review-nav-buttons{display:flex;gap:7px}.review-nav[hidden]{display:none}
	.case{display:grid;grid-template-columns:minmax(620px,1.45fr) minmax(480px,.9fr);gap:30px;padding:18px 0;border-bottom:1px solid var(--line);align-items:start}.media-panel{min-width:0}.media-shell{position:relative;width:100%;margin:auto;aspect-ratio:16/9;background:#111;display:grid;place-items:center;overflow:hidden}.media-shell[data-display-size="720"]{max-width:1280px}.media-shell[data-display-size="540"]{max-width:960px}.media-shell[data-display-size="360"]{max-width:640px}.media-shell video{display:block;width:100%;height:100%;object-fit:contain;background:#111}.media-status{position:absolute;left:12px;top:12px;padding:5px 8px;background:rgba(17,17,17,.78);color:#fff;font-size:12px;pointer-events:none}.media-status[hidden]{display:none}.video-controls{display:flex;align-items:center;gap:7px;min-height:48px;padding:7px;background:#181815;color:#f8f8f3}.video-controls button,.video-controls select{min-height:34px;padding:6px 9px;border-color:#55554e;color:#f8f8f3;background:#282824}.video-controls button:hover:not(:disabled),.video-controls select:hover{border-color:#f8f8f3}.video-progress{flex:1;min-width:150px;height:24px;padding:0;accent-color:#f8f8f3;border:0;background:transparent;cursor:pointer}.video-time,.source-resolution{font:11px ui-monospace,monospace;white-space:nowrap;color:#d2d2ca}.head{display:flex;justify-content:space-between;gap:12px}.eyebrow{font:11px ui-monospace,monospace;color:var(--muted);word-break:break-all}h2{font-size:21px;margin:5px 0}.duration{font:700 12px ui-monospace,monospace;color:var(--red)}
.badge{display:inline-block;padding:3px 7px;border:1px solid var(--line);font-size:11px}.badge.claimed{border-color:var(--blue);color:var(--blue)}.badge.reviewed{border-color:var(--green);color:var(--green)}.taxonomy{display:flex;gap:6px;align-items:center;margin:7px 0;color:var(--muted);font-size:12px}.severity{padding:2px 6px;border:1px solid var(--line)}.severity.reject{border-color:var(--red);color:var(--red)}.severity.review{border-color:#9a6a12;color:#7d560e}
dl{display:grid;grid-template-columns:92px 1fr;margin:18px 0;border-top:1px solid var(--line)}dt,dd{padding:7px 0;margin:0;border-bottom:1px solid var(--line)}dt{color:var(--muted)}dd{font:12px ui-monospace,monospace}.actions,.media-tabs{display:flex;gap:7px;margin:13px 0}.actions{align-items:center;flex-wrap:wrap}.actions .media-tabs{margin:0}.action-divider{width:1px;height:28px;background:var(--line)}.claim{background:var(--ink);color:var(--card);border-color:var(--ink)}.choices{display:grid;grid-template-columns:repeat(3,1fr);gap:6px;margin:13px 0}.choices button[data-decision=confirmed].selected{background:var(--red);color:#fff}.choices button[data-decision=false_positive].selected{background:var(--green);color:#fff}.choices button[data-decision=unsure].selected{background:var(--ink);color:#fff}label{display:grid;gap:6px;color:var(--muted)}textarea{min-height:80px;resize:vertical}.message{min-height:21px;margin-top:8px;color:var(--muted)}.message.error{color:var(--red)}
.gold{margin:16px 0;padding:14px;border:1px solid var(--line);background:#f2f2ec}.gold h3{font-size:13px;margin:0 0 8px}.gold p{margin:5px 0;color:var(--muted)}.machine-verdict{display:flex;align-items:center;justify-content:space-between;gap:12px;padding:10px 0;border-bottom:1px solid var(--line)}.machine-verdict span{color:var(--muted)}.machine-verdict strong{font-size:16px}.problem-title{margin-top:13px!important}.problem-list{display:grid;gap:7px;margin-top:8px}.problem{padding:9px 10px;border-left:3px solid var(--red);background:var(--card)}.problem strong{font-size:13px}.problem p{margin:3px 0}.problem small{display:block;color:var(--blue);font:11px/1.45 ui-monospace,monospace}.seek-section{margin-top:12px;padding-top:11px;border-top:1px solid var(--line)}.seek-section>span{display:block;color:var(--muted);font-size:12px}.seek-points{display:flex;flex-wrap:wrap;gap:6px;margin-top:7px}.seek-points button{padding:5px 7px;background:var(--card);font:11px ui-monospace,monospace}.media-tabs button.active,.issue-row button.selected{background:var(--ink);color:var(--card);border-color:var(--ink)}.quick-actions{display:grid;grid-template-columns:minmax(150px,1fr) auto auto;gap:7px;margin-top:13px}.quick-confirm{background:var(--ink);color:var(--card);border-color:var(--ink);font-weight:650}.correction{margin-top:14px;padding-top:13px;border-top:1px solid var(--line)}.correction[hidden]{display:none}.correction-head{display:flex;align-items:start;justify-content:space-between;gap:12px;margin-bottom:8px}.correction-head p{margin:0;max-width:45ch}.issue-row{display:grid;grid-template-columns:minmax(120px,1fr) repeat(3,80px);gap:5px;align-items:center;margin:6px 0}.issue-row span{font-size:12px}.save-correction{width:100%;margin-top:10px}.note-label{margin-top:12px}
	@media(max-width:1180px){.case{grid-template-columns:1fr}.media-shell{max-height:72vh}.case>section{max-width:none}}
	@media(max-width:850px){header{position:static;padding:14px}.top{align-items:start;flex-direction:column}.bar{margin-top:10px}.case{grid-template-columns:1fr;padding-top:12px}.media-shell{aspect-ratio:16/9}main{padding:8px 14px 50px}.identity{width:100%}.identity input{flex:1}.review-nav{position:sticky;top:0;z-index:3;background:var(--bg);padding:8px 0}.video-controls{flex-wrap:wrap}.video-progress{order:8;flex-basis:100%}.source-resolution{margin-left:auto}.quick-actions{grid-template-columns:1fr}.issue-row{grid-template-columns:1fr repeat(3,1fr)}}
	</style></head><body><header><div class="top"><div><h1>异常复检台</h1><div class="sub"><span id="dot" class="status"></span><span id="sync">正在连接 PostgreSQL…</span></div></div><div class="identity"><label for="reviewer">审核员</label><input id="reviewer" placeholder="姓名或工号"><button id="assign" hidden>自动分配</button><button id="refresh">刷新</button><a id="logout" href="/auth/logout" hidden>退出</a></div></div><div class="bar"><button data-filter="queue" class="active">我的任务</button><button data-filter="all">全部</button><button data-filter="pending">待领取</button><button data-filter="claimed">审核中</button><button data-filter="reviewed">已完成</button><select id="kind-filter" aria-label="错误类型"><option value="all">全部错误类型</option></select><span class="stat">总计<strong id="total">–</strong></span><span class="stat">待领取<strong id="pending">–</strong></span><span class="stat">审核中<strong id="claimed">–</strong></span><span class="stat">已完成<strong id="reviewed">–</strong></span></div></header><main><nav id="review-nav" class="review-nav" hidden aria-label="复检队列导航"><span id="review-position" class="review-position"></span><div class="review-nav-buttons"><button id="previous" type="button">上一条</button><button id="next" type="button">下一条</button></div></nav><div id="cases"><div class="empty">正在读取事件队列…</div></div></main>
<script>
const structuredKey='egoqc-structured-drafts';
let savedStructured={};try{savedStructured=JSON.parse(localStorage.getItem(structuredKey)||'{}')}catch(_){savedStructured={}}
	const state={events:[],filter:'queue',kindFilter:'all',focusIndex:0,activeId:null,displaySize:localStorage.getItem('egoqc-display-size')||'fit',tokens:{},drafts:{},structured:savedStructured,views:{},me:null};
const reviewer=document.querySelector('#reviewer');reviewer.value=localStorage.getItem('egoqc-reviewer')||'';reviewer.onchange=()=>localStorage.setItem('egoqc-reviewer',reviewer.value.trim());
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
async function api(path,options={}){const response=await fetch(path,{...options,headers:{'Content-Type':'application/json',...(options.headers||{})}});if(response.status===401){location.href='/auth/login';throw new Error('登录已过期')}let body={};try{body=await response.json()}catch(_){body={error:response.statusText}}if(!response.ok)throw new Error(body.error||`HTTP ${response.status}`);return body}
function who(){if(state.me?.authenticated)return state.me.user_id;const value=reviewer.value.trim();if(!value)throw new Error('请先填写审核员姓名或工号');localStorage.setItem('egoqc-reviewer',value);return value}
function whoOrEmpty(){return state.me?.authenticated?state.me.user_id:reviewer.value.trim()}
function title(kind){return kind==='hand_absent'?'手连续离画':kind==='persistent_extra_hands'?'疑似第二人手':kind}
function isGold(e){return e.metrics?.review_mode==='episode_gold'}
function decisionLabel(value,e){if(isGold(e))return {confirmed:'机器结论已确认',false_positive:'已修正机器误报',unsure:'需仲裁'}[value]||value;return {confirmed:'真实问题',false_positive:'模型误报',unsure:'不确定'}[value]||value}
function detail(e){if(!state.structured[e.event_id]){const stored=e.decision_details&&Object.keys(e.decision_details).length?e.decision_details:{};state.structured[e.event_id]=JSON.parse(JSON.stringify(stored));}const d=state.structured[e.event_id];d.issue_verdicts=d.issue_verdicts||{};d.observed_labels=d.observed_labels||[];d.causes=d.causes||[];return d}
function saveStructured(){localStorage.setItem(structuredKey,JSON.stringify(state.structured))}
function issueLabel(e,code){return e.metrics?.issue_labels?.[code]||title(code)}
const issueDescriptions={instantaneous_velocity_outlier:'相邻帧手腕速度超过本段自适应上限。重点区分真实快速动作与追踪瞬移。',joint_rotation_jitter:'真实手形变化较平滑，但 MANO 手指关节角出现高频抖动。重点看指节抽动、穿模或局部翻转。',position_jitter:'手腕三维轨迹存在高频残差。背景和真实手稳定而 mesh 来回跳动时，通常是标注问题。',temporal_spike:'某一帧或少量连续帧明显偏离前后轨迹，随后立即返回，常见于追踪跳点。',wrist_rotation_jitter:'手腕朝向在短时间内反复变化。重点确认 MANO 手掌是否突然翻转，而真实手并未同步翻转。',camera_jitter:'相机位姿出现高频不稳定。重点看背景、双手和 MANO 是否一起产生不合理运动。',mask_flicker:'手部有效标记在相邻帧间反复开关。重点看真实手仍可见时 mesh 是否突然消失。',pose_freeze_candidate:'真实手仍在运动，但 MANO 手指姿态长时间不更新或整体平移，疑似追踪冻结。',beta_drift:'同一 episode 内 MANO 手型参数发生变化，可能表现为 mesh 尺寸或手指比例缓慢漂移。'};
function issueDescription(e,code){return e.metrics?.issue_descriptions?.[code]||issueDescriptions[code]||'机器规则发现异常，请结合原视频和 MANO 叠加确认。'}
const evidenceSpecs={position_jitter:[['metric:left_position_jitter_p99_mm','左手 p99','mm',1],['metric:right_position_jitter_p99_mm','右手 p99','mm',1]],wrist_rotation_jitter:[['metric:left_wrist_rotation_jitter_p99_deg','左腕 p99','°',1],['metric:right_wrist_rotation_jitter_p99_deg','右腕 p99','°',1]],joint_rotation_jitter:[['metric:left_joint_rotation_jitter_p99_deg','左手指 p99','°',1],['metric:right_joint_rotation_jitter_p99_deg','右手指 p99','°',1]],temporal_spike:[['metric:left_temporal_spike_count','左手跳点','帧',1],['metric:right_temporal_spike_count','右手跳点','帧',1]],instantaneous_velocity_outlier:[['metric:left_velocity_outlier_ratio','左手速度离群','%',100],['metric:right_velocity_outlier_ratio','右手速度离群','%',100]]};
function issueEvidence(e,code){const evidence=e.metrics?.rule_evidence||{};return (evidenceSpecs[code]||[]).filter(([key])=>evidence[key]!=null).map(([key,label,unit,scale])=>{const value=Number(evidence[key])*scale;return `${label} ${value.toFixed(unit==='帧'?0:2)}${unit}`}).join('；')}
function sampleFrames(e){const fps=Number(e.metrics?.fps||30),frames=[...new Set((e.metrics?.sample_frames||[]).map(Number).filter(Number.isFinite))].sort((a,b)=>a-b);if(!frames.length)return '';const shown=frames.slice(0,12),extra=frames.length-shown.length;return `<div class="seek-section"><span>机器抽样的候选位置，点击可直接跳转${extra>0?`，另有 ${extra} 帧未展示`:''}</span><div class="seek-points">${shown.map(frame=>`<button type="button" data-seek-frame="${frame}">帧 ${frame} / ${(frame/fps).toFixed(2)}s</button>`).join('')}</div></div>`}
function machineTierLabel(e){return {bronze:'建议返工',silver:'建议人工复核',gold:'建议通过'}[e.metrics?.baseline_tier]||'已有机器结论'}
function mediaControls(e){if(!isGold(e))return '';const view=state.views[e.event_id]||'raw';return `<span class="action-divider" aria-hidden="true"></span><div class="media-tabs" aria-label="视频视图"><button data-view="raw" class="${view==='raw'?'active':''}">原视频</button><button data-view="annotated" class="${view==='annotated'?'active':''}" ${e.metrics?.annotated_clip_path?'':'disabled'}>MANO 叠加</button></div>`}
function goldPanel(e){if(!isGold(e))return '';const issues=e.metrics?.issue_codes||[];const problems=issues.length?issues.map(code=>{const evidence=issueEvidence(e,code);return `<div class="problem"><strong>${esc(issueLabel(e,code))}</strong><p>${esc(issueDescription(e,code))}</p>${evidence?`<small>机器证据：${esc(evidence)}</small>`:''}</div>`}).join(''):'<div class="problem"><strong>没有底层规则说明</strong><p>当前只命中 episode 聚合规则，需要结合完整视频人工判断。</p></div>';return `<div class="gold" data-role="gold"><h3>机器预审</h3><div class="machine-verdict"><span>建议结论</span><strong>${esc(machineTierLabel(e))}</strong></div><h3 class="problem-title">这个片段可能有什么问题</h3><div class="problem-list">${problems}</div>${sampleFrames(e)}<div class="quick-actions"><button class="quick-confirm" data-quick="confirm">确认机器结论</button><button data-action="edit-correction">有误报，展开修改</button><button data-quick="unsure">无法判断</button></div><div class="correction" data-role="correction" hidden><div class="correction-head"><div><h3>只修改有问题的规则</h3><p>默认均为真错误。只把误报或无法判断的规则改掉即可。</p></div><button data-action="cancel-correction">收起</button></div><div class="issue-list">${issues.map(code=>`<div class="issue-row" data-issue="${esc(code)}"><span>${esc(issueLabel(e,code))}</span>${[['confirmed','真错误'],['false_positive','误报'],['unsure','不确定']].map(([v,l])=>`<button data-issue-verdict="${v}">${l}</button>`).join('')}</div>`).join('')}</div><button class="save-correction" data-action="save-correction">保存修正</button></div></div>`}
	function card(e){const note=state.drafts[e.event_id]??e.note??'';const heading=isGold(e)?'机器预审复核':(e.error_label||title(e.kind));const choices=isGold(e)?'':`<div class="choices">${['confirmed','false_positive','unsure'].map(d=>`<button data-decision="${d}">${decisionLabel(d,e)}</button>`).join('')}</div>`;const view=isGold(e)?state.views[e.event_id]||'raw':'primary';const displayOptions=[['fit','自适应'],['720','720p 显示'],['540','540p 显示'],['360','360p 显示']].map(([value,label])=>`<option value="${value}" ${state.displaySize===value?'selected':''}>${label}</option>`).join('');return `<article class="case" data-id="${esc(e.event_id)}"><div class="media-panel"><div class="media-shell" data-display-size="${esc(state.displaySize)}"><video preload="metadata" playsinline src="/api/review/media/${encodeURIComponent(e.event_id)}?view=${view}"></video><span class="media-status">正在读取视频…</span></div><div class="video-controls" aria-label="视频播放控制"><button type="button" data-video-action="toggle">播放</button><button type="button" data-video-action="back" title="后退 1 秒">−1s</button><button type="button" data-video-action="forward" title="前进 1 秒">+1s</button><span class="video-time">00:00 / --:--</span><input class="video-progress" type="range" min="0" max="1000" value="0" step="1" aria-label="视频进度"><button type="button" data-video-action="mute">静音</button><select data-video-rate aria-label="播放速度"><option value="0.5">0.5×</option><option value="1" selected>1×</option><option value="1.5">1.5×</option><option value="2">2×</option></select><select data-display-resolution aria-label="显示分辨率">${displayOptions}</select><span class="source-resolution">源 --×--</span><button type="button" data-video-action="fullscreen">全屏</button></div></div><section><div class="head"><div><div class="eyebrow">${esc(e.video_id)}</div><h2 data-role="title">${esc(heading)}</h2></div><span class="duration">${Number(e.duration_s).toFixed(2)}s</span></div><span data-role="state" class="badge ${esc(e.state)}"></span><div class="taxonomy" data-role="taxonomy"><span data-role="category"></span><span data-role="severity" class="severity"></span><code>${esc(e.kind)}</code></div><dl><dt>${isGold(e)?'审核片段':'异常区间'}</dt><dd>${Number(e.start_s).toFixed(2)}-${Number(e.end_s).toFixed(2)}s</dd><dt>当前结论</dt><dd data-role="decision"></dd><dt>版本</dt><dd data-role="version"></dd></dl><div class="actions"></div>${goldPanel(e)}${choices}<label class="note-label">补充说明（可选）<textarea placeholder="仅在需要时填写画面、时间点或判断依据">${esc(note)}</textarea></label><div class="message"></div></section></article>`}
function isVisible(e){const reviewerName=state.me?.authenticated?state.me.user_id:reviewer.value.trim();const stateMatch=state.filter==='all'||e.state===state.filter||(state.filter==='queue'&&(e.state==='pending'||(e.state==='claimed'&&e.claimed_by===reviewerName)));return stateMatch&&(state.kindFilter==='all'||e.kind===state.kindFilter)}
	function visibleRows(events=state.events){return events.filter(isVisible)}
	function focusedRows(events=state.events){const rows=visibleRows(events);if(!rows.length){state.focusIndex=0;state.activeId=null;return []}const activeIndex=state.activeId?rows.findIndex(e=>e.event_id===state.activeId):-1;if(activeIndex>=0)state.focusIndex=activeIndex;state.focusIndex=Math.max(0,Math.min(state.focusIndex,rows.length-1));state.activeId=rows[state.focusIndex].event_id;return [rows[state.focusIndex]]}
	function updateNavigation(){const rows=visibleRows(),nav=document.querySelector('#review-nav');nav.hidden=!rows.length;if(!rows.length)return;state.focusIndex=Math.max(0,Math.min(state.focusIndex,rows.length-1));document.querySelector('#review-position').textContent=`第 ${state.focusIndex+1} / ${rows.length} 条 · 浏览器仅加载当前视频`;document.querySelector('#previous').disabled=state.focusIndex===0;document.querySelector('#next').disabled=state.focusIndex>=rows.length-1}
function eventNode(id){return Array.from(document.querySelectorAll('.case')).find(node=>node.dataset.id===id)}
function goldDetails(e,humanAction,verdict){const issues=e.metrics?.issue_codes||[];return {workflow_version:'assisted-fast-v1',human_action:humanAction,issue_verdicts:Object.fromEntries(issues.map(code=>[code,verdict])),aggregate_issue_codes:[...(e.metrics?.aggregate_issue_codes||[])],machine_tier:e.metrics?.baseline_tier??null,confidence:'not_collected'}}
function submitReview(node,e,button,decision,details){const id=e.event_id,textarea=node.querySelector('textarea');return act(node,button,'提交中…',async()=>{const result=await api(`/api/review/events/${encodeURIComponent(id)}/decision`,{method:'POST',body:JSON.stringify({reviewer:who(),claim_token:state.tokens[id],decision,note:textarea.value,expected_version:e.version,details})});delete state.tokens[id];delete state.drafts[id];delete state.structured[id];saveStructured();acceptEvent(result);await refreshStats()})}
function updateGold(node,e,canEdit){if(!isGold(e))return;const d=detail(e),issues=e.metrics?.issue_codes||[],editing=Boolean(d.editing);node.querySelector('[data-role=correction]').hidden=!editing;node.querySelectorAll('[data-quick],[data-action=edit-correction]').forEach(b=>b.disabled=!canEdit);node.querySelector('[data-action=cancel-correction]').disabled=!canEdit;node.querySelectorAll('[data-issue]').forEach(row=>row.querySelectorAll('[data-issue-verdict]').forEach(b=>{const value=d.issue_verdicts[row.dataset.issue]||'confirmed';b.classList.toggle('selected',value===b.dataset.issueVerdict);b.disabled=!canEdit}));const changed=issues.some(code=>(d.issue_verdicts[code]||'confirmed')!=='confirmed');node.querySelector('[data-action=save-correction]').disabled=!canEdit||!changed}
function bindGold(node,e,canEdit){if(!isGold(e))return;const d=detail(e),issues=e.metrics?.issue_codes||[],fps=Number(e.metrics?.fps||30),persist=()=>{saveStructured();updateGold(node,e,canEdit)};node.querySelectorAll('[data-view]').forEach(b=>b.onclick=()=>{if(b.disabled)return;const video=node.querySelector('video'),position=video.currentTime||0;state.views[e.event_id]=b.dataset.view;video.src=`/api/review/media/${encodeURIComponent(e.event_id)}?view=${b.dataset.view}`;video.load();video.onloadedmetadata=()=>{video.currentTime=Math.min(position,video.duration||position)};node.querySelectorAll('[data-view]').forEach(x=>x.classList.toggle('active',x===b))});node.querySelectorAll('[data-seek-frame]').forEach(b=>b.onclick=()=>{const video=node.querySelector('video');video.currentTime=Math.max(0,Number(b.dataset.seekFrame)/fps-.4);video.play().catch(()=>{})});node.querySelector('[data-action=edit-correction]').onclick=()=>{d.editing=true;for(const code of issues)d.issue_verdicts[code]=d.issue_verdicts[code]||'confirmed';persist()};node.querySelector('[data-action=cancel-correction]').onclick=()=>{d.editing=false;persist()};node.querySelectorAll('[data-issue]').forEach(row=>row.querySelectorAll('[data-issue-verdict]').forEach(b=>b.onclick=()=>{d.issue_verdicts[row.dataset.issue]=b.dataset.issueVerdict;persist()}));node.querySelector('[data-quick=confirm]').onclick=event=>submitReview(node,e,event.currentTarget,'confirmed',goldDetails(e,'confirm_machine','confirmed'));node.querySelector('[data-quick=unsure]').onclick=event=>submitReview(node,e,event.currentTarget,'unsure',goldDetails(e,'route_to_adjudication','unsure'));node.querySelector('[data-action=save-correction]').onclick=event=>{const verdicts=Object.fromEntries(issues.map(code=>[code,d.issue_verdicts[code]||'confirmed']));if(!Object.values(verdicts).some(value=>value!=='confirmed')){node.querySelector('.message').textContent='请至少把一条规则改为误报或不确定';return}const details={workflow_version:'assisted-fast-v1',human_action:'correct_machine',issue_verdicts:verdicts,aggregate_issue_codes:[...(e.metrics?.aggregate_issue_codes||[])],machine_tier:e.metrics?.baseline_tier??null,confidence:'not_collected'};submitReview(node,e,event.currentTarget,'false_positive',details)};updateGold(node,e,canEdit)}
	function bindVideo(node){const panel=node.querySelector('.media-panel');if(panel.dataset.bound)return;panel.dataset.bound='true';const video=node.querySelector('video'),shell=node.querySelector('.media-shell'),status=node.querySelector('.media-status'),toggle=node.querySelector('[data-video-action=toggle]'),mute=node.querySelector('[data-video-action=mute]'),progress=node.querySelector('.video-progress'),time=node.querySelector('.video-time'),source=node.querySelector('.source-resolution');const clock=value=>{if(!Number.isFinite(value))return '--:--';const minutes=Math.floor(value/60),seconds=Math.floor(value%60);return `${String(minutes).padStart(2,'0')}:${String(seconds).padStart(2,'0')}`};const sync=()=>{toggle.textContent=video.paused?'播放':'暂停';mute.textContent=video.muted?'取消静音':'静音';time.textContent=`${clock(video.currentTime)} / ${clock(video.duration)}`;progress.value=video.duration?Math.round(video.currentTime/video.duration*1000):0};const show=message=>{status.textContent=message;status.hidden=false};const hide=()=>status.hidden=true;video.addEventListener('loadedmetadata',()=>{source.textContent=`源 ${video.videoWidth}×${video.videoHeight}`;sync()});video.addEventListener('loadeddata',hide);video.addEventListener('canplay',hide);video.addEventListener('waiting',()=>show('视频缓冲中…'));video.addEventListener('stalled',()=>show('网络较慢，正在重试…'));video.addEventListener('error',()=>show('视频加载失败，请刷新重试'));video.addEventListener('timeupdate',sync);video.addEventListener('play',sync);video.addEventListener('pause',sync);video.addEventListener('volumechange',sync);video.addEventListener('ended',sync);toggle.onclick=()=>video.paused?video.play().catch(()=>{}):video.pause();video.onclick=toggle.onclick;node.querySelector('[data-video-action=back]').onclick=()=>video.currentTime=Math.max(0,video.currentTime-1);node.querySelector('[data-video-action=forward]').onclick=()=>video.currentTime=Math.min(video.duration||Infinity,video.currentTime+1);mute.onclick=()=>video.muted=!video.muted;progress.oninput=()=>{if(video.duration)video.currentTime=Number(progress.value)/1000*video.duration};node.querySelector('[data-video-rate]').onchange=event=>video.playbackRate=Number(event.target.value);node.querySelector('[data-display-resolution]').onchange=event=>{state.displaySize=event.target.value;shell.dataset.displaySize=state.displaySize;localStorage.setItem('egoqc-display-size',state.displaySize)};node.querySelector('[data-video-action=fullscreen]').onclick=()=>panel.requestFullscreen?.();if(video.readyState>=1){source.textContent=`源 ${video.videoWidth}×${video.videoHeight}`;sync()}if(video.readyState>=2)hide()}
	function bindCard(node,e){bindVideo(node);const id=e.event_id,textarea=node.querySelector('textarea');textarea.oninput=()=>state.drafts[id]=textarea.value;const claim=node.querySelector('[data-action=claim]');if(claim)claim.onclick=()=>act(node,claim,'领取中…',async()=>{const result=await api(`/api/review/events/${encodeURIComponent(id)}/claim`,{method:'POST',body:JSON.stringify({reviewer:who(),lease_seconds:900})});state.tokens[id]=result.claim_token;acceptEvent(result);await refreshStats()});const release=node.querySelector('[data-action=release]');if(release)release.onclick=()=>act(node,release,'释放中…',async()=>{const result=await api(`/api/review/events/${encodeURIComponent(id)}/release`,{method:'POST',body:JSON.stringify({reviewer:who(),claim_token:state.tokens[id]})});delete state.tokens[id];delete state.drafts[id];delete state.structured[id];saveStructured();acceptEvent(result);await refreshStats()});node.querySelectorAll('[data-decision]').forEach(button=>button.onclick=()=>submitReview(node,e,button,button.dataset.decision,{}))}
function patchCard(node,e){const mine=e.state==='claimed'&&e.claimed_by===whoOrEmpty(),canDecide=mine&&state.tokens[e.event_id];const badge=node.querySelector('[data-role=state]');badge.className=`badge ${e.state}`;badge.textContent=e.state+(e.claimed_by?' / '+e.claimed_by:'');node.querySelector('[data-role=category]').textContent=e.category_label||e.category||'未分类';const severity=node.querySelector('[data-role=severity]');severity.className=`severity ${e.severity||'review'}`;severity.textContent=e.severity_label||e.severity||'需复核';node.querySelector('[data-role=decision]').textContent=e.decision?decisionLabel(e.decision,e):'未提交';node.querySelector('[data-role=version]').textContent=e.version;node.querySelector('.actions').innerHTML=`<button class="claim" data-action="claim" ${e.state==='reviewed'||(e.state==='claimed'&&!mine)?'disabled':''}>${mine?'续期':'领取'}</button>${mine?'<button data-action="release">释放</button>':''}${mediaControls(e)}`;node.querySelectorAll('[data-decision]').forEach(button=>{button.textContent=decisionLabel(button.dataset.decision,e);button.disabled=!canDecide;button.classList.toggle('selected',button.dataset.decision===e.decision)});const textarea=node.querySelector('textarea');textarea.disabled=!canDecide;if(document.activeElement!==textarea&&state.drafts[e.event_id]===undefined)textarea.value=e.note||'';bindCard(node,e);bindGold(node,e,canDecide)}
	function render(){const rows=focusedRows();document.querySelector('#cases').innerHTML=rows.length?rows.map(card).join(''):'<div class="empty">当前筛选下没有事件</div>';rows.forEach(e=>patchCard(eventNode(e.event_id),e));updateNavigation()}
function eventStamp(e){return [e.state,e.version,e.claimed_by,e.decision,e.note,e.kind,e.category,e.severity].join('|')}
	function reconcile(events){const previous=new Map(state.events.map(e=>[e.event_id,eventStamp(e)]));state.events=events;const next=focusedRows(),current=Array.from(document.querySelectorAll('.case')).map(node=>node.dataset.id);if(current.length!==next.length||current.some((id,index)=>id!==next[index].event_id)){render();return}next.forEach(e=>{if(previous.get(e.event_id)!==eventStamp(e))patchCard(eventNode(e.event_id),e)});updateNavigation()}
function acceptEvent(updated){const index=state.events.findIndex(e=>e.event_id===updated.event_id);if(index>=0)state.events[index]=updated;const node=eventNode(updated.event_id);if(node&&isVisible(updated))patchCard(node,updated);else render()}
async function act(node,button,label,fn){const msg=node.querySelector('.message'),original=button.textContent;button.disabled=true;button.textContent=label;msg.className='message';msg.textContent='';try{await fn()}catch(error){msg.className='message error';msg.textContent=error.message;button.disabled=false;button.textContent=original}}
function applyStats(stats){for(const [key,value] of Object.entries(stats.states||{})){const el=document.querySelector('#'+key);if(el)el.textContent=value}document.querySelector('#total').textContent=stats.total;for(const key of ['pending','claimed','reviewed'])if(!stats.states?.[key])document.querySelector('#'+key).textContent='0'}
function applyTypeOptions(events){const select=document.querySelector('#kind-filter'),counts=new Map(),labels=new Map();events.forEach(e=>{counts.set(e.kind,(counts.get(e.kind)||0)+1);labels.set(e.kind,e.error_label||title(e.kind))});const signature=JSON.stringify(Array.from(counts.entries()));if(select.dataset.signature===signature)return;select.dataset.signature=signature;const selected=state.kindFilter;select.innerHTML='<option value="all">全部错误类型 · '+events.length+'</option>'+Array.from(counts.entries()).map(([kind,count])=>`<option value="${esc(kind)}">${esc(labels.get(kind))} · ${count}</option>`).join('');if(selected==='all'||counts.has(selected))select.value=selected;else{state.kindFilter='all';select.value='all'}}
async function refreshStats(){applyStats(await api('/api/review/stats'))}
async function refresh(){try{const [events,stats]=await Promise.all([api('/api/review/events?limit=500'),api('/api/review/stats')]);applyStats(stats);applyTypeOptions(events.events);reconcile(events.events);document.querySelector('#sync').textContent=`已同步 · ${new Date().toLocaleTimeString()}`;document.querySelector('#dot').classList.remove('off')}catch(error){document.querySelector('#sync').textContent='连接失败 · '+error.message;document.querySelector('#dot').classList.add('off')}}
async function loadIdentity(){const me=await api('/api/me');state.me=me;if(me.authenticated){reviewer.value=me.display_name;reviewer.readOnly=true;document.querySelector('#logout').hidden=false;document.querySelector('#assign').hidden=me.role!=='admin'}else{reviewer.readOnly=false}}
	document.querySelectorAll('[data-filter]').forEach(button=>button.onclick=()=>{document.querySelectorAll('[data-filter]').forEach(x=>x.classList.remove('active'));button.classList.add('active');state.filter=button.dataset.filter;state.focusIndex=0;state.activeId=null;render()});document.querySelector('#kind-filter').onchange=event=>{state.kindFilter=event.target.value;state.focusIndex=0;state.activeId=null;render()};document.querySelector('#previous').onclick=()=>{state.focusIndex=Math.max(0,state.focusIndex-1);state.activeId=null;render();scrollTo({top:document.querySelector('main').offsetTop,behavior:'smooth'})};document.querySelector('#next').onclick=()=>{state.focusIndex=Math.min(visibleRows().length-1,state.focusIndex+1);state.activeId=null;render();scrollTo({top:document.querySelector('main').offsetTop,behavior:'smooth'})};document.querySelector('#assign').onclick=async()=>{const button=document.querySelector('#assign'),original=button.textContent;button.disabled=true;button.textContent='分配中…';try{await api('/api/review/assign',{method:'POST',body:'{}'});await refresh()}finally{button.disabled=false;button.textContent=original}};document.querySelector('#refresh').onclick=refresh;reviewer.addEventListener('change',()=>{state.focusIndex=0;state.activeId=null;render()});loadIdentity().then(refresh).catch(error=>document.querySelector('#sync').textContent=error.message);setInterval(refresh,3000);
</script></body></html>"""


class _Handler(SimpleHTTPRequestHandler):
    store: ReviewStore
    evidence_root: Path
    auth_config: Optional[FeishuAuthConfig] = None

    def _json(self, status: int, payload: Any) -> None:
        body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length > 1024 * 1024:
            raise ValueError("请求体过大")
        value = json.loads(self.rfile.read(length) or b"{}")
        if not isinstance(value, dict):
            raise ValueError("请求体必须是 JSON object")
        return value

    def _session_token(self) -> str:
        cookie = SimpleCookie()
        try:
            cookie.load(self.headers.get("Cookie", ""))
        except Exception:
            return ""
        value = cookie.get("egoqc_session")
        return value.value if value else ""

    def _current_user(self) -> Optional[dict[str, Any]]:
        if self.auth_config is None:
            return None
        return self.store.session_user(self._session_token())

    def _require_user(self) -> tuple[bool, Optional[dict[str, Any]]]:
        user = self._current_user()
        if self.auth_config is not None and user is None:
            self._json(401, {"error": "需要飞书登录"})
            return False, None
        return True, user

    def _redirect(self, location: str, cookie: Optional[str] = None) -> None:
        self.send_response(302)
        self.send_header("Location", location)
        self.send_header("Cache-Control", "no-store")
        if cookie:
            self.send_header("Set-Cookie", cookie)
        self.end_headers()

    def _login_page(self) -> None:
        body = LOGIN_HTML.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _auth_callback(self, query: dict[str, list[str]]) -> None:
        if self.auth_config is None:
            self.send_error(404)
            return
        state = query.get("state", [""])[0]
        code = query.get("code", [""])[0]
        flow = self.store.consume_oauth_flow(state) if state else None
        if flow is None or not code:
            self._json(400, {"error": "飞书登录状态无效或授权已取消"})
            return
        try:
            token = exchange_code(self.auth_config, code, str(flow["code_verifier"]))
            profile = fetch_user_info(self.auth_config, str(token["access_token"]))
            role = "admin" if self.store.review_user_count() == 0 else "reviewer"
            user = self.store.upsert_feishu_user(profile, role)
            session = self.store.create_session(
                str(user["user_id"]), self.auth_config.session_ttl_seconds
            )
        except (FeishuAuthError, ValueError) as exc:
            self._json(401, {"error": str(exc)})
            return
        secure = "; Secure" if self.auth_config.secure_cookie else ""
        cookie = (
            f"egoqc_session={session}; Path=/; HttpOnly; SameSite=Lax; "
            f"Max-Age={self.auth_config.session_ttl_seconds}{secure}"
        )
        self._redirect(str(flow.get("return_to") or "/"), cookie)

    def _event_media(self, event_id: str, view: str = "primary") -> Optional[Path]:
        event = self.store.get_event(unquote(event_id))
        if not event:
            return None
        metrics = event.get("metrics") if isinstance(event.get("metrics"), dict) else {}
        if view == "raw":
            candidate = metrics.get("raw_clip_path") or event.get("clip_path")
        elif view == "annotated":
            candidate = metrics.get("annotated_clip_path")
        elif view == "primary":
            candidate = event.get("clip_path")
        else:
            return None
        if not candidate:
            return None
        path = Path(str(candidate))
        if not path.is_absolute():
            path = self.evidence_root / path
        path = path.expanduser().resolve()
        root = self.evidence_root.resolve()
        if path != root and root not in path.parents:
            return None
        return path if path.is_file() else None

    def _serve_media(
        self,
        event_id: str,
        user: Optional[dict[str, Any]] = None,
        head_only: bool = False,
        view: str = "primary",
    ) -> None:
        decoded_id = unquote(event_id)
        if user and user.get("role") != "admin":
            if self.store.event_assignee(decoded_id) != str(user["user_id"]):
                self._json(403, {"error": "该视频未分配给当前审核员"})
                return
        path = self._event_media(decoded_id, view)
        if path is None:
            self.send_error(404, "media not found")
            return
        size = path.stat().st_size
        start, end, partial = 0, max(size - 1, 0), False
        header = self.headers.get("Range")
        if header and header.startswith("bytes="):
            try:
                left, right = header[6:].split("-", 1)
                start = int(left) if left else 0
                end = min(end, int(right)) if right else end
                if start < 0 or start > end or start >= size:
                    raise ValueError
                partial = True
            except ValueError:
                self.send_response(416); self.send_header("Content-Range", f"bytes */{size}"); self.end_headers(); return
        length = end - start + 1
        self.send_response(206 if partial else 200)
        self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(length))
        if partial:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        if head_only:
            return
        with path.open("rb") as handle:
            handle.seek(start)
            remaining = length
            while remaining:
                chunk = handle.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk); remaining -= len(chunk)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/auth/login":
            if self.auth_config is None:
                self._redirect("/")
                return
            flow = self.store.create_oauth_flow(parse_qs(parsed.query).get("next", ["/"])[0])
            self._redirect(authorization_url(self.auth_config, flow["state"], flow["code_verifier"]))
            return
        if path == "/auth/callback":
            self._auth_callback(parse_qs(parsed.query))
            return
        if path == "/auth/logout":
            self.store.delete_session(self._session_token())
            self._redirect("/", "egoqc_session=; Path=/; HttpOnly; SameSite=Lax; Max-Age=0")
            return
        if path in {"/", "/review.html"}:
            if self.auth_config is not None and self._current_user() is None:
                self._login_page()
                return
            body = REVIEW_HTML.encode("utf-8")
            self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8"); self.send_header("Cache-Control", "no-store"); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body); return
        if path == "/api/me":
            if self.auth_config is None:
                self._json(200, {"authenticated": False, "auth_enabled": False})
                return
            allowed, user = self._require_user()
            if allowed and user:
                self._json(200, {"authenticated": True, "auth_enabled": True, **user})
            return
        allowed, user = self._require_user()
        if not allowed:
            return
        if path == "/api/review/events":
            query = parse_qs(parsed.query)
            assignee = None if user and user.get("role") == "admin" else (str(user["user_id"]) if user else None)
            self._json(200, {"events": self.store.list_events(query.get("state", [None])[0], query.get("kind", [None])[0], int(query.get("limit", [100])[0]), assignee)}); return
        if path == "/api/review/stats":
            assignee = None if user and user.get("role") == "admin" else (str(user["user_id"]) if user else None)
            self._json(200, self.store.stats(assignee)); return
        if path == "/api/review/assignments":
            if not user or user.get("role") != "admin":
                self._json(403, {"error": "仅管理员可查看分配统计"})
            else:
                self._json(200, {"reviewers": self.store.assignment_stats()})
            return
        parts = path.strip("/").split("/")
        if len(parts) == 4 and parts[:3] == ["api", "review", "media"]:
            query = parse_qs(parsed.query)
            self._serve_media(parts[3], user, view=query.get("view", ["primary"])[0]); return
        self.send_error(404)

    def do_HEAD(self) -> None:  # noqa: N802
        allowed, user = self._require_user()
        if not allowed:
            return
        parsed = urlparse(self.path)
        parts = parsed.path.strip("/").split("/")
        if len(parts) == 4 and parts[:3] == ["api", "review", "media"]:
            query = parse_qs(parsed.query)
            self._serve_media(parts[3], user, True, query.get("view", ["primary"])[0]); return
        self.send_error(404)

    def do_POST(self) -> None:  # noqa: N802
        parts = urlparse(self.path).path.strip("/").split("/")
        try:
            allowed, user = self._require_user()
            if not allowed:
                return
            if parts == ["api", "review", "assign"]:
                if not user or user.get("role") != "admin":
                    self._json(403, {"error": "仅管理员可执行任务分配"})
                else:
                    self._json(200, self.store.assign_unassigned_videos(str(user["user_id"])))
                return
            if len(parts) != 5 or parts[:3] != ["api", "review", "events"]:
                self.send_error(404); return
            event_id, action, body = unquote(parts[3]), parts[4], self._body()
            reviewer_id = str(user["user_id"]) if user else str(body["reviewer"])
            if user and user.get("role") != "admin" and self.store.event_assignee(event_id) != reviewer_id:
                self._json(403, {"error": "该视频未分配给当前审核员"})
                return
            if action == "claim":
                result = self.store.claim(event_id, reviewer_id, int(body.get("lease_seconds", 900)))
            elif action == "decision":
                result = self.store.decide(
                    event_id,
                    reviewer_id,
                    str(body["claim_token"]),
                    str(body["decision"]),
                    str(body.get("note", "")),
                    body.get("expected_version"),
                    body.get("details") or {},
                )
            elif action == "release":
                result = self.store.release(event_id, reviewer_id, str(body["claim_token"]))
            else:
                self.send_error(404); return
            self._json(200, result)
        except ReviewConflict as exc:
            self._json(409, {"error": str(exc)})
        except KeyError as exc:
            self._json(404, {"error": f"不存在: {exc}"})
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            self._json(400, {"error": str(exc)})

    def log_message(self, format: str, *args: Any) -> None:
        print(f"review-http {self.address_string()} {format % args}", flush=True)


def serve_postgres_review(
    database_url: str,
    evidence_root: Path,
    host: str = "127.0.0.1",
    port: int = 8767,
) -> None:
    store = ReviewStore(database_url)
    store.init_schema()
    evidence_root = evidence_root.expanduser().resolve()
    auth_config = FeishuAuthConfig.from_env()

    class BoundHandler(_Handler):
        pass

    BoundHandler.store = store
    BoundHandler.evidence_root = evidence_root
    BoundHandler.auth_config = auth_config
    server = ThreadingHTTPServer((host, port), partial(BoundHandler, directory=str(evidence_root)))
    auth_mode = "feishu" if auth_config else "development"
    print(f"EgoQC PostgreSQL review: http://{host}:{port}/ auth={auth_mode}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
