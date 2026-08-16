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
.top{display:flex;align-items:end;justify-content:space-between;gap:20px;max-width:1480px;margin:auto}h1{font-size:24px;line-height:1;margin:0 0 7px;letter-spacing:-.04em}.sub{color:var(--muted)}
.identity{display:flex;align-items:center;gap:7px}.identity input{width:170px}.identity a{color:var(--muted);padding:7px}.status{display:inline-block;width:7px;height:7px;background:var(--green);border-radius:50%;margin-right:6px}.status.off{background:var(--red)}
.bar{max-width:1480px;margin:14px auto 0;display:flex;gap:7px;align-items:center;overflow:auto}.stat{padding:7px 10px;border:1px solid var(--line);white-space:nowrap}.stat strong{font:700 13px ui-monospace,monospace;margin-left:8px}
button,input,textarea,select{font:inherit;color:inherit;border:1px solid var(--line);background:transparent;padding:8px 10px}button{cursor:pointer}button:hover:not(:disabled){border-color:var(--ink)}button:disabled{opacity:.38;cursor:not-allowed}button.active{background:var(--ink);color:var(--card);border-color:var(--ink)}
main{max-width:1480px;margin:auto;padding:14px 24px 70px}.empty{padding:90px 20px;text-align:center;color:var(--muted)}
.case{display:grid;grid-template-columns:minmax(480px,1.5fr) minmax(330px,.72fr);gap:24px;padding:22px 0;border-bottom:1px solid var(--line)}video{display:block;width:100%;max-height:590px;background:#111}.head{display:flex;justify-content:space-between;gap:12px}.eyebrow{font:11px ui-monospace,monospace;color:var(--muted);word-break:break-all}h2{font-size:21px;margin:5px 0}.duration{font:700 12px ui-monospace,monospace;color:var(--red)}
.badge{display:inline-block;padding:3px 7px;border:1px solid var(--line);font-size:11px}.badge.claimed{border-color:var(--blue);color:var(--blue)}.badge.reviewed{border-color:var(--green);color:var(--green)}.taxonomy{display:flex;gap:6px;align-items:center;margin:7px 0;color:var(--muted);font-size:12px}.severity{padding:2px 6px;border:1px solid var(--line)}.severity.reject{border-color:var(--red);color:var(--red)}.severity.review{border-color:#9a6a12;color:#7d560e}
dl{display:grid;grid-template-columns:92px 1fr;margin:18px 0;border-top:1px solid var(--line)}dt,dd{padding:7px 0;margin:0;border-bottom:1px solid var(--line)}dt{color:var(--muted)}dd{font:12px ui-monospace,monospace}.actions{display:flex;gap:7px;margin:13px 0}.claim{background:var(--ink);color:var(--card);border-color:var(--ink)}.choices{display:grid;grid-template-columns:repeat(3,1fr);gap:6px;margin:13px 0}.choices button[data-decision=confirmed].selected{background:var(--red);color:#fff}.choices button[data-decision=false_positive].selected{background:var(--green);color:#fff}.choices button[data-decision=unsure].selected{background:var(--ink);color:#fff}label{display:grid;gap:6px;color:var(--muted)}textarea{min-height:80px;resize:vertical}.message{min-height:21px;margin-top:8px;color:var(--muted)}.message.error{color:var(--red)}
@media(max-width:850px){header{position:static}.top{align-items:start;flex-direction:column}.case{grid-template-columns:1fr}main{padding:8px 14px}.identity{width:100%}.identity input{flex:1}}
</style></head><body><header><div class="top"><div><h1>异常复检台</h1><div class="sub"><span id="dot" class="status"></span><span id="sync">正在连接 PostgreSQL…</span></div></div><div class="identity"><label for="reviewer">审核员</label><input id="reviewer" placeholder="姓名或工号"><button id="assign" hidden>自动分配</button><button id="refresh">刷新</button><a id="logout" href="/auth/logout" hidden>退出</a></div></div><div class="bar"><button data-filter="queue" class="active">我的任务</button><button data-filter="all">全部</button><button data-filter="pending">待领取</button><button data-filter="claimed">审核中</button><button data-filter="reviewed">已完成</button><select id="kind-filter" aria-label="错误类型"><option value="all">全部错误类型</option></select><span class="stat">总计<strong id="total">–</strong></span><span class="stat">待领取<strong id="pending">–</strong></span><span class="stat">审核中<strong id="claimed">–</strong></span><span class="stat">已完成<strong id="reviewed">–</strong></span></div></header><main id="cases"><div class="empty">正在读取事件队列…</div></main>
<script>
const state={events:[],filter:'queue',kindFilter:'all',tokens:{},drafts:{},me:null};
const reviewer=document.querySelector('#reviewer');reviewer.value=localStorage.getItem('egoqc-reviewer')||'';reviewer.onchange=()=>localStorage.setItem('egoqc-reviewer',reviewer.value.trim());
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
async function api(path,options={}){const response=await fetch(path,{...options,headers:{'Content-Type':'application/json',...(options.headers||{})}});if(response.status===401){location.href='/auth/login';throw new Error('登录已过期')}let body={};try{body=await response.json()}catch(_){body={error:response.statusText}}if(!response.ok)throw new Error(body.error||`HTTP ${response.status}`);return body}
function who(){if(state.me?.authenticated)return state.me.user_id;const value=reviewer.value.trim();if(!value)throw new Error('请先填写审核员姓名或工号');localStorage.setItem('egoqc-reviewer',value);return value}
function whoOrEmpty(){return state.me?.authenticated?state.me.user_id:reviewer.value.trim()}
function title(kind){return kind==='hand_absent'?'手连续离画':kind==='persistent_extra_hands'?'疑似第二人手':kind}
function decisionLabel(value){return {confirmed:'真实问题',false_positive:'模型误报',unsure:'不确定'}[value]||value}
function card(e){const note=state.drafts[e.event_id]??e.note??'';return `<article class="case" data-id="${esc(e.event_id)}"><video controls preload="metadata" src="/api/review/media/${encodeURIComponent(e.event_id)}"></video><section><div class="head"><div><div class="eyebrow">${esc(e.video_id)}</div><h2>${esc(e.error_label||title(e.kind))}</h2></div><span class="duration">${Number(e.duration_s).toFixed(2)}s</span></div><span data-role="state" class="badge ${esc(e.state)}"></span><div class="taxonomy"><span data-role="category"></span><span data-role="severity" class="severity"></span><code>${esc(e.kind)}</code></div><dl><dt>异常区间</dt><dd>${Number(e.start_s).toFixed(2)}–${Number(e.end_s).toFixed(2)}s</dd><dt>当前结论</dt><dd data-role="decision"></dd><dt>版本</dt><dd data-role="version"></dd></dl><div class="actions"></div><div class="choices">${['confirmed','false_positive','unsure'].map(d=>`<button data-decision="${d}">${decisionLabel(d)}</button>`).join('')}</div><label>备注<textarea placeholder="画面内容、时间点和判断依据">${esc(note)}</textarea></label><div class="message"></div></section></article>`}
function isVisible(e){const reviewerName=state.me?.authenticated?state.me.user_id:reviewer.value.trim();const stateMatch=state.filter==='all'||e.state===state.filter||(state.filter==='queue'&&(e.state==='pending'||(e.state==='claimed'&&e.claimed_by===reviewerName)));return stateMatch&&(state.kindFilter==='all'||e.kind===state.kindFilter)}
function visibleRows(events=state.events){return events.filter(isVisible)}
function eventNode(id){return Array.from(document.querySelectorAll('.case')).find(node=>node.dataset.id===id)}
function bindCard(node,e){const id=e.event_id,textarea=node.querySelector('textarea');textarea.oninput=()=>state.drafts[id]=textarea.value;const claim=node.querySelector('[data-action=claim]');if(claim)claim.onclick=()=>act(node,claim,'领取中…',async()=>{const result=await api(`/api/review/events/${encodeURIComponent(id)}/claim`,{method:'POST',body:JSON.stringify({reviewer:who(),lease_seconds:900})});state.tokens[id]=result.claim_token;acceptEvent(result);await refreshStats()});const release=node.querySelector('[data-action=release]');if(release)release.onclick=()=>act(node,release,'释放中…',async()=>{const result=await api(`/api/review/events/${encodeURIComponent(id)}/release`,{method:'POST',body:JSON.stringify({reviewer:who(),claim_token:state.tokens[id]})});delete state.tokens[id];delete state.drafts[id];acceptEvent(result);await refreshStats()});node.querySelectorAll('[data-decision]').forEach(button=>button.onclick=()=>act(node,button,'提交中…',async()=>{const result=await api(`/api/review/events/${encodeURIComponent(id)}/decision`,{method:'POST',body:JSON.stringify({reviewer:who(),claim_token:state.tokens[id],decision:button.dataset.decision,note:textarea.value,expected_version:e.version})});delete state.tokens[id];delete state.drafts[id];acceptEvent(result);await refreshStats()}))}
function patchCard(node,e){const mine=e.state==='claimed'&&e.claimed_by===whoOrEmpty(),canDecide=mine&&state.tokens[e.event_id];const badge=node.querySelector('[data-role=state]');badge.className=`badge ${e.state}`;badge.textContent=e.state+(e.claimed_by?' · '+e.claimed_by:'');node.querySelector('[data-role=category]').textContent=e.category_label||e.category||'未分类';const severity=node.querySelector('[data-role=severity]');severity.className=`severity ${e.severity||'review'}`;severity.textContent=e.severity_label||e.severity||'需复核';node.querySelector('[data-role=decision]').textContent=e.decision||'—';node.querySelector('[data-role=version]').textContent=e.version;node.querySelector('.actions').innerHTML=`<button class="claim" data-action="claim" ${e.state==='reviewed'||(e.state==='claimed'&&!mine)?'disabled':''}>${mine?'续期':'领取'}</button>${mine?'<button data-action="release">释放</button>':''}`;node.querySelectorAll('[data-decision]').forEach(button=>{button.textContent=decisionLabel(button.dataset.decision);button.disabled=!canDecide;button.classList.toggle('selected',button.dataset.decision===e.decision)});const textarea=node.querySelector('textarea');textarea.disabled=!canDecide;if(document.activeElement!==textarea&&state.drafts[e.event_id]===undefined)textarea.value=e.note||'';bindCard(node,e)}
function render(){const rows=visibleRows();document.querySelector('#cases').innerHTML=rows.length?rows.map(card).join(''):'<div class="empty">当前筛选下没有事件</div>';rows.forEach(e=>patchCard(eventNode(e.event_id),e))}
function eventStamp(e){return [e.state,e.version,e.claimed_by,e.decision,e.note,e.kind,e.category,e.severity].join('|')}
function reconcile(events){const previous=new Map(state.events.map(e=>[e.event_id,eventStamp(e)])),next=visibleRows(events),current=Array.from(document.querySelectorAll('.case')).map(node=>node.dataset.id);state.events=events;if(current.length!==next.length||current.some((id,index)=>id!==next[index].event_id)){render();return}next.forEach(e=>{if(previous.get(e.event_id)!==eventStamp(e))patchCard(eventNode(e.event_id),e)})}
function acceptEvent(updated){const index=state.events.findIndex(e=>e.event_id===updated.event_id);if(index>=0)state.events[index]=updated;const node=eventNode(updated.event_id);if(node&&isVisible(updated))patchCard(node,updated);else render()}
async function act(node,button,label,fn){const msg=node.querySelector('.message'),original=button.textContent;button.disabled=true;button.textContent=label;msg.className='message';msg.textContent='';try{await fn()}catch(error){msg.className='message error';msg.textContent=error.message;button.disabled=false;button.textContent=original}}
function applyStats(stats){for(const [key,value] of Object.entries(stats.states||{})){const el=document.querySelector('#'+key);if(el)el.textContent=value}document.querySelector('#total').textContent=stats.total;for(const key of ['pending','claimed','reviewed'])if(!stats.states?.[key])document.querySelector('#'+key).textContent='0'}
function applyTypeOptions(events){const select=document.querySelector('#kind-filter'),counts=new Map(),labels=new Map();events.forEach(e=>{counts.set(e.kind,(counts.get(e.kind)||0)+1);labels.set(e.kind,e.error_label||title(e.kind))});const signature=JSON.stringify(Array.from(counts.entries()));if(select.dataset.signature===signature)return;select.dataset.signature=signature;const selected=state.kindFilter;select.innerHTML='<option value="all">全部错误类型 · '+events.length+'</option>'+Array.from(counts.entries()).map(([kind,count])=>`<option value="${esc(kind)}">${esc(labels.get(kind))} · ${count}</option>`).join('');if(selected==='all'||counts.has(selected))select.value=selected;else{state.kindFilter='all';select.value='all'}}
async function refreshStats(){applyStats(await api('/api/review/stats'))}
async function refresh(){try{const [events,stats]=await Promise.all([api('/api/review/events?limit=500'),api('/api/review/stats')]);applyStats(stats);applyTypeOptions(events.events);reconcile(events.events);document.querySelector('#sync').textContent=`已同步 · ${new Date().toLocaleTimeString()}`;document.querySelector('#dot').classList.remove('off')}catch(error){document.querySelector('#sync').textContent='连接失败 · '+error.message;document.querySelector('#dot').classList.add('off')}}
async function loadIdentity(){const me=await api('/api/me');state.me=me;if(me.authenticated){reviewer.value=me.display_name;reviewer.readOnly=true;document.querySelector('#logout').hidden=false;document.querySelector('#assign').hidden=me.role!=='admin'}else{reviewer.readOnly=false}}
document.querySelectorAll('[data-filter]').forEach(button=>button.onclick=()=>{document.querySelectorAll('[data-filter]').forEach(x=>x.classList.remove('active'));button.classList.add('active');state.filter=button.dataset.filter;render()});document.querySelector('#kind-filter').onchange=event=>{state.kindFilter=event.target.value;render()};document.querySelector('#assign').onclick=async()=>{const button=document.querySelector('#assign'),original=button.textContent;button.disabled=true;button.textContent='分配中…';try{await api('/api/review/assign',{method:'POST',body:'{}'});await refresh()}finally{button.disabled=false;button.textContent=original}};document.querySelector('#refresh').onclick=refresh;reviewer.addEventListener('change',render);loadIdentity().then(refresh).catch(error=>document.querySelector('#sync').textContent=error.message);setInterval(refresh,3000);
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

    def _event_media(self, event_id: str) -> Optional[Path]:
        event = self.store.get_event(unquote(event_id))
        if not event or not event.get("clip_path"):
            return None
        path = Path(event["clip_path"])
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
    ) -> None:
        decoded_id = unquote(event_id)
        if user and user.get("role") != "admin":
            if self.store.event_assignee(decoded_id) != str(user["user_id"]):
                self._json(403, {"error": "该视频未分配给当前审核员"})
                return
        path = self._event_media(decoded_id)
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
            self._serve_media(parts[3], user); return
        self.send_error(404)

    def do_HEAD(self) -> None:  # noqa: N802
        allowed, user = self._require_user()
        if not allowed:
            return
        parts = urlparse(self.path).path.strip("/").split("/")
        if len(parts) == 4 and parts[:3] == ["api", "review", "media"]:
            self._serve_media(parts[3], user, True); return
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
                result = self.store.decide(event_id, reviewer_id, str(body["claim_token"]), str(body["decision"]), str(body.get("note", "")), body.get("expected_version"))
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
