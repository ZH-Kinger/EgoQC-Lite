from __future__ import annotations

import json
import hashlib
import secrets
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from .review_taxonomy import describe_error


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS review_datasets (
    dataset_id uuid PRIMARY KEY,
    name text NOT NULL UNIQUE,
    source_root text,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS review_runs (
    run_id uuid PRIMARY KEY,
    dataset_id uuid NOT NULL REFERENCES review_datasets(dataset_id),
    name text NOT NULL,
    detector jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE(dataset_id, name)
);
CREATE TABLE IF NOT EXISTS review_events (
    event_id text PRIMARY KEY,
    run_id uuid NOT NULL REFERENCES review_runs(run_id),
    video_id text NOT NULL,
    kind text NOT NULL,
    category text NOT NULL DEFAULT 'unclassified',
    severity text NOT NULL DEFAULT 'review',
    start_s double precision NOT NULL,
    end_s double precision NOT NULL,
    duration_s double precision NOT NULL,
    clip_path text,
    source_uri text,
    priority integer NOT NULL DEFAULT 0,
    metrics jsonb NOT NULL DEFAULT '{}'::jsonb,
    state text NOT NULL DEFAULT 'pending'
        CHECK (state IN ('pending', 'claimed', 'reviewed', 'escalated')),
    claimed_by text,
    claim_token uuid,
    claimed_at timestamptz,
    lease_expires_at timestamptz,
    decision text CHECK (decision IS NULL OR decision IN ('confirmed', 'false_positive', 'unsure')),
    decision_details jsonb NOT NULL DEFAULT '{}'::jsonb,
    note text NOT NULL DEFAULT '',
    reviewed_by text,
    reviewed_at timestamptz,
    version integer NOT NULL DEFAULT 1,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS review_events_queue_idx
    ON review_events (state, priority DESC, created_at, event_id);
CREATE INDEX IF NOT EXISTS review_events_video_idx ON review_events (video_id);
CREATE TABLE IF NOT EXISTS review_decisions (
    decision_id uuid PRIMARY KEY,
    event_id text NOT NULL REFERENCES review_events(event_id),
    reviewer text NOT NULL,
    decision text NOT NULL CHECK (decision IN ('confirmed', 'false_positive', 'unsure')),
    details jsonb NOT NULL DEFAULT '{}'::jsonb,
    note text NOT NULL DEFAULT '',
    event_version integer NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS review_decisions_event_idx
    ON review_decisions (event_id, created_at DESC);
CREATE TABLE IF NOT EXISTS review_users (
    user_id text PRIMARY KEY,
    provider text NOT NULL,
    provider_subject text NOT NULL,
    union_id text,
    tenant_key text,
    display_name text NOT NULL,
    avatar_url text,
    role text NOT NULL DEFAULT 'reviewer'
        CHECK (role IN ('reviewer', 'admin', 'auditor')),
    active boolean NOT NULL DEFAULT true,
    capacity_weight double precision NOT NULL DEFAULT 1.0 CHECK (capacity_weight > 0),
    created_at timestamptz NOT NULL DEFAULT now(),
    last_login_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE(provider, provider_subject)
);
CREATE TABLE IF NOT EXISTS review_sessions (
    token_hash text PRIMARY KEY,
    user_id text NOT NULL REFERENCES review_users(user_id) ON DELETE CASCADE,
    expires_at timestamptz NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    last_seen_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS review_sessions_expiry_idx ON review_sessions(expires_at);
CREATE TABLE IF NOT EXISTS review_oauth_states (
    state_hash text PRIMARY KEY,
    code_verifier text NOT NULL,
    return_to text NOT NULL DEFAULT '/',
    expires_at timestamptz NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS review_video_assignments (
    run_id uuid NOT NULL REFERENCES review_runs(run_id) ON DELETE CASCADE,
    video_id text NOT NULL,
    primary_user_id text NOT NULL REFERENCES review_users(user_id),
    assigned_at timestamptz NOT NULL DEFAULT now(),
    assigned_by text,
    PRIMARY KEY(run_id, video_id)
);
CREATE INDEX IF NOT EXISTS review_video_assignments_user_idx
    ON review_video_assignments(primary_user_id, run_id);
ALTER TABLE review_events ADD COLUMN IF NOT EXISTS category text NOT NULL DEFAULT 'unclassified';
ALTER TABLE review_events ADD COLUMN IF NOT EXISTS severity text NOT NULL DEFAULT 'review';
ALTER TABLE review_events ADD COLUMN IF NOT EXISTS decision_details jsonb NOT NULL DEFAULT '{}'::jsonb;
ALTER TABLE review_decisions ADD COLUMN IF NOT EXISTS details jsonb NOT NULL DEFAULT '{}'::jsonb;
UPDATE review_events SET category='hand_visibility', severity='reject'
    WHERE kind='hand_absent' AND category='unclassified';
UPDATE review_events SET category='multi_person', severity='reject'
    WHERE kind='persistent_extra_hands' AND category='unclassified';
"""


class ReviewConflict(RuntimeError):
    pass


def _psycopg() -> Any:
    try:
        import psycopg
        from psycopg.rows import dict_row
        from psycopg.types.json import Jsonb
    except ImportError as exc:  # pragma: no cover - depends on optional install
        raise RuntimeError("PostgreSQL 支持未安装；请运行 pip install -e '.[postgres]'") from exc
    return psycopg, dict_row, Jsonb


def _stable_uuid(namespace: str, value: str) -> uuid.UUID:
    return uuid.uuid5(uuid.NAMESPACE_URL, f"egoqc:{namespace}:{value}")


def _jsonable(row: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if row is None:
        return None
    result = dict(row)
    for key, value in result.items():
        if isinstance(value, (datetime, uuid.UUID)):
            result[key] = value.isoformat() if isinstance(value, datetime) else str(value)
    return result


def _event_json(row: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    item = _jsonable(row)
    if item is None or "kind" not in item:
        return item
    description = describe_error(str(item["kind"]))
    description["category"] = item.get("category") or description["category"]
    description["severity"] = item.get("severity") or description["severity"]
    item.update(description)
    return item


class ReviewStore:
    """Small synchronous PostgreSQL store for shared human review state."""

    def __init__(self, database_url: str):
        if not database_url:
            raise ValueError("database_url 不能为空")
        self.database_url = database_url

    def _connect(self) -> Any:
        psycopg, dict_row, _ = _psycopg()
        return psycopg.connect(self.database_url, row_factory=dict_row)

    def init_schema(self) -> None:
        with self._connect() as connection:
            connection.execute(SCHEMA_SQL)

    def import_events(
        self,
        events: Iterable[Dict[str, Any]],
        dataset_name: str,
        run_name: str,
        source_root: Optional[str] = None,
        detector: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        _, _, Jsonb = _psycopg()
        dataset_id = _stable_uuid("dataset", dataset_name)
        run_id = _stable_uuid("run", f"{dataset_name}:{run_name}")
        imported = 0
        with self._connect() as connection:
            connection.execute(
                """INSERT INTO review_datasets(dataset_id, name, source_root)
                   VALUES (%s, %s, %s)
                   ON CONFLICT(name) DO UPDATE SET source_root=COALESCE(EXCLUDED.source_root, review_datasets.source_root)""",
                (dataset_id, dataset_name, source_root),
            )
            connection.execute(
                """INSERT INTO review_runs(run_id, dataset_id, name, detector)
                   VALUES (%s, %s, %s, %s)
                   ON CONFLICT(dataset_id, name) DO UPDATE SET detector=EXCLUDED.detector""",
                (run_id, dataset_id, run_name, Jsonb(detector or {})),
            )
            for event in events:
                known = {
                    "event_id", "video_id", "kind", "start_s", "end_s", "duration_s",
                    "clip", "clip_path", "source_uri", "priority", "category", "severity",
                }
                metrics = {key: value for key, value in event.items() if key not in known}
                taxonomy = describe_error(str(event["kind"]))
                connection.execute(
                    """INSERT INTO review_events(
                           event_id, run_id, video_id, kind, category, severity, start_s, end_s, duration_s,
                           clip_path, source_uri, priority, metrics)
                       VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                       ON CONFLICT(event_id) DO UPDATE SET
                           run_id=EXCLUDED.run_id, video_id=EXCLUDED.video_id, kind=EXCLUDED.kind,
                           category=EXCLUDED.category, severity=EXCLUDED.severity,
                           start_s=EXCLUDED.start_s, end_s=EXCLUDED.end_s,
                           duration_s=EXCLUDED.duration_s, clip_path=EXCLUDED.clip_path,
                           source_uri=EXCLUDED.source_uri, priority=EXCLUDED.priority,
                           metrics=EXCLUDED.metrics, updated_at=now()""",
                    (
                        str(event["event_id"]), run_id, str(event["video_id"]), str(event["kind"]),
                        str(event.get("category") or taxonomy["category"]),
                        str(event.get("severity") or taxonomy["severity"]),
                        float(event["start_s"]), float(event["end_s"]), float(event["duration_s"]),
                        str(event.get("clip") or event.get("clip_path") or ""),
                        str(event.get("source_uri") or ""), int(event.get("priority", 0)), Jsonb(metrics),
                    ),
                )
                imported += 1
        return {"dataset_id": str(dataset_id), "run_id": str(run_id), "events": imported}

    def list_events(
        self,
        state: Optional[str] = None,
        kind: Optional[str] = None,
        limit: int = 100,
        assigned_to: Optional[str] = None,
    ) -> list[Dict[str, Any]]:
        clauses, values = [], []
        if state:
            clauses.append("e.state = %s")
            values.append(state)
        if kind:
            clauses.append("e.kind = %s")
            values.append(kind)
        if assigned_to:
            clauses.append("a.primary_user_id = %s")
            values.append(assigned_to)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        values.append(max(1, min(int(limit), 500)))
        query = f"""SELECT e.event_id, e.video_id, e.kind, e.category, e.severity,
                    e.start_s, e.end_s, e.duration_s, e.clip_path, e.source_uri,
                    e.priority, e.metrics, e.state, e.claimed_by, e.lease_expires_at,
                    e.decision, e.decision_details, e.note, e.reviewed_by, e.reviewed_at, e.version,
                    e.updated_at, a.primary_user_id AS assigned_to
                    FROM review_events e
                    LEFT JOIN review_video_assignments a
                      ON a.run_id=e.run_id AND a.video_id=e.video_id
                    {where}
                    ORDER BY e.priority DESC, e.created_at, e.event_id LIMIT %s"""
        with self._connect() as connection:
            self._reap_expired(connection)
            rows = connection.execute(query, values).fetchall()
        result = []
        for row in rows:
            result.append(_event_json(row) or {})
        return result

    def stats(self, assigned_to: Optional[str] = None) -> Dict[str, Any]:
        with self._connect() as connection:
            self._reap_expired(connection)
            join = """LEFT JOIN review_video_assignments a
                      ON a.run_id=e.run_id AND a.video_id=e.video_id"""
            where, values = ("WHERE a.primary_user_id=%s", [assigned_to]) if assigned_to else ("", [])
            rows = connection.execute(
                f"SELECT e.state, count(*) AS count FROM review_events e {join} {where} GROUP BY e.state",
                values,
            ).fetchall()
            kinds = connection.execute(
                f"SELECT e.kind, count(*) AS count FROM review_events e {join} {where} GROUP BY e.kind",
                values,
            ).fetchall()
            categories = connection.execute(
                f"SELECT e.category, count(*) AS count FROM review_events e {join} {where} GROUP BY e.category",
                values,
            ).fetchall()
        counts = {row["state"]: row["count"] for row in rows}
        return {
            "total": sum(counts.values()),
            "states": counts,
            "kinds": {row["kind"]: row["count"] for row in kinds},
            "categories": {row["category"]: row["count"] for row in categories},
            "server_time": datetime.now(timezone.utc).isoformat(),
        }

    @staticmethod
    def _reap_expired(connection: Any) -> None:
        connection.execute(
            """UPDATE review_events SET state='pending', claimed_by=NULL, claim_token=NULL,
                   claimed_at=NULL, lease_expires_at=NULL, version=version+1, updated_at=now()
               WHERE state='claimed' AND lease_expires_at < now()"""
        )

    def get_event(self, event_id: str) -> Optional[Dict[str, Any]]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM review_events WHERE event_id=%s", (event_id,)
            ).fetchone()
        return _event_json(row)

    def claim(self, event_id: str, reviewer: str, lease_seconds: int = 900) -> Dict[str, Any]:
        reviewer = reviewer.strip()
        if not reviewer or len(reviewer) > 120:
            raise ValueError("审核员不能为空且长度不能超过 120")
        token = uuid.uuid4()
        with self._connect() as connection:
            row = connection.execute(
                """UPDATE review_events SET state='claimed', claimed_by=%s, claim_token=%s,
                       claimed_at=now(), lease_expires_at=now() + (%s * interval '1 second'),
                       version=version+1, updated_at=now()
                   WHERE event_id=%s AND state <> 'reviewed'
                     AND (state='pending' OR lease_expires_at < now() OR claimed_by=%s)
                   RETURNING *""",
                (reviewer, token, max(30, min(int(lease_seconds), 7200)), event_id, reviewer),
            ).fetchone()
            if row is None:
                raise ReviewConflict("事件已被其他审核员领取或已经完成")
        result = _event_json(row) or {}
        result["claim_token"] = str(token)
        return result

    def decide(
        self,
        event_id: str,
        reviewer: str,
        claim_token: str,
        decision: str,
        note: str = "",
        expected_version: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        reviewer = reviewer.strip()
        if not reviewer or len(reviewer) > 120:
            raise ValueError("审核员不能为空且长度不能超过 120")
        if len(note) > 10000:
            raise ValueError("备注不能超过 10000 字符")
        if decision not in {"confirmed", "false_positive", "unsure"}:
            raise ValueError("无效 decision")
        if details is None:
            details = {}
        if not isinstance(details, dict):
            raise ValueError("details 必须是 JSON object")
        if len(json.dumps(details, ensure_ascii=False)) > 100000:
            raise ValueError("details 不能超过 100000 字符")
        _, _, Jsonb = _psycopg()
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM review_events WHERE event_id=%s FOR UPDATE", (event_id,)
            ).fetchone()
            if row is None:
                raise KeyError(event_id)
            if str(row.get("claim_token")) != claim_token or row.get("claimed_by") != reviewer:
                raise ReviewConflict("领取令牌无效，请刷新后重新领取")
            if row.get("lease_expires_at") and row["lease_expires_at"] < datetime.now(timezone.utc):
                raise ReviewConflict("领取已过期，请重新领取")
            if expected_version is not None and row["version"] != expected_version:
                raise ReviewConflict("事件已被更新，请刷新后重试")
            next_version = int(row["version"]) + 1
            connection.execute(
                """INSERT INTO review_decisions(
                       decision_id, event_id, reviewer, decision, details, note, event_version)
                   VALUES (%s,%s,%s,%s,%s,%s,%s)""",
                (
                    uuid.uuid4(), event_id, reviewer, decision, Jsonb(details), note,
                    next_version,
                ),
            )
            updated = connection.execute(
                """UPDATE review_events SET state='reviewed', decision=%s,
                       decision_details=%s, note=%s,
                       reviewed_by=%s, reviewed_at=now(), claimed_by=NULL, claim_token=NULL,
                       claimed_at=NULL, lease_expires_at=NULL, version=%s, updated_at=now()
                   WHERE event_id=%s RETURNING *""",
                (decision, Jsonb(details), note, reviewer, next_version, event_id),
            ).fetchone()
        return _event_json(updated) or {}

    def release(self, event_id: str, reviewer: str, claim_token: str) -> Dict[str, Any]:
        reviewer = reviewer.strip()
        if not reviewer:
            raise ValueError("审核员不能为空")
        with self._connect() as connection:
            row = connection.execute(
                """UPDATE review_events SET state='pending', claimed_by=NULL, claim_token=NULL,
                       claimed_at=NULL, lease_expires_at=NULL, version=version+1, updated_at=now()
                   WHERE event_id=%s AND claimed_by=%s AND claim_token=%s AND state='claimed'
                   RETURNING *""",
                (event_id, reviewer, claim_token),
            ).fetchone()
            if row is None:
                raise ReviewConflict("事件不属于当前审核员或领取已经失效")
        return _event_json(row) or {}

    @staticmethod
    def _token_hash(token: str) -> str:
        return hashlib.sha256(token.encode("utf-8")).hexdigest()

    def create_oauth_flow(self, return_to: str = "/", ttl_seconds: int = 600) -> Dict[str, str]:
        state = secrets.token_urlsafe(32)
        verifier = secrets.token_urlsafe(64)[:96]
        safe_return = return_to if return_to.startswith("/") and not return_to.startswith("//") else "/"
        with self._connect() as connection:
            connection.execute("DELETE FROM review_oauth_states WHERE expires_at < now()")
            connection.execute(
                """INSERT INTO review_oauth_states(state_hash, code_verifier, return_to, expires_at)
                   VALUES (%s,%s,%s,now() + (%s * interval '1 second'))""",
                (self._token_hash(state), verifier, safe_return, max(60, min(ttl_seconds, 900))),
            )
        return {"state": state, "code_verifier": verifier, "return_to": safe_return}

    def consume_oauth_flow(self, state: str) -> Optional[Dict[str, Any]]:
        with self._connect() as connection:
            row = connection.execute(
                """DELETE FROM review_oauth_states
                   WHERE state_hash=%s AND expires_at >= now()
                   RETURNING code_verifier, return_to""",
                (self._token_hash(state),),
            ).fetchone()
        return _jsonable(row)

    def upsert_feishu_user(self, profile: Dict[str, Any], default_role: str = "reviewer") -> Dict[str, Any]:
        open_id = str(profile.get("open_id") or "").strip()
        if not open_id:
            raise ValueError("飞书用户信息缺少 open_id")
        user_id = f"feishu:{open_id}"
        display_name = str(profile.get("name") or profile.get("en_name") or open_id)
        with self._connect() as connection:
            row = connection.execute(
                """INSERT INTO review_users(
                       user_id, provider, provider_subject, union_id, tenant_key,
                       display_name, avatar_url, role)
                   VALUES (%s,'feishu',%s,%s,%s,%s,%s,%s)
                   ON CONFLICT(user_id) DO UPDATE SET
                       union_id=EXCLUDED.union_id, tenant_key=EXCLUDED.tenant_key,
                       display_name=EXCLUDED.display_name, avatar_url=EXCLUDED.avatar_url,
                       last_login_at=now()
                   RETURNING *""",
                (
                    user_id, open_id, profile.get("union_id"), profile.get("tenant_key"),
                    display_name, profile.get("avatar_url"), default_role,
                ),
            ).fetchone()
        return _jsonable(row) or {}

    def review_user_count(self) -> int:
        with self._connect() as connection:
            row = connection.execute("SELECT count(*) AS count FROM review_users").fetchone()
        return int(row["count"])

    def create_session(self, user_id: str, ttl_seconds: int = 43200) -> str:
        token = secrets.token_urlsafe(48)
        ttl = max(300, min(int(ttl_seconds), 7 * 24 * 3600))
        with self._connect() as connection:
            connection.execute("DELETE FROM review_sessions WHERE expires_at < now()")
            connection.execute(
                """INSERT INTO review_sessions(token_hash, user_id, expires_at)
                   VALUES (%s,%s,now() + (%s * interval '1 second'))""",
                (self._token_hash(token), user_id, ttl),
            )
        return token

    def session_user(self, token: str) -> Optional[Dict[str, Any]]:
        if not token:
            return None
        with self._connect() as connection:
            row = connection.execute(
                """UPDATE review_sessions s SET last_seen_at=now()
                   FROM review_users u
                   WHERE s.token_hash=%s AND s.user_id=u.user_id
                     AND s.expires_at >= now() AND u.active
                   RETURNING u.user_id, u.provider, u.provider_subject, u.union_id,
                             u.tenant_key, u.display_name, u.avatar_url, u.role""",
                (self._token_hash(token),),
            ).fetchone()
        return _jsonable(row)

    def delete_session(self, token: str) -> None:
        if not token:
            return
        with self._connect() as connection:
            connection.execute(
                "DELETE FROM review_sessions WHERE token_hash=%s", (self._token_hash(token),)
            )

    def assign_unassigned_videos(self, assigned_by: str = "system") -> Dict[str, Any]:
        """Greedy workload balancing; all events from one video stay with one reviewer."""
        with self._connect() as connection:
            connection.execute("SELECT pg_advisory_xact_lock(hashtext('egoqc-review-assignment'))")
            reviewers = connection.execute(
                """SELECT user_id, capacity_weight FROM review_users
                   WHERE active AND role IN ('reviewer','admin') ORDER BY user_id"""
            ).fetchall()
            if not reviewers:
                return {"assigned_videos": 0, "reviewers": 0}
            load_rows = connection.execute(
                """SELECT a.primary_user_id,
                          COALESCE(sum(e.duration_s),0) + count(e.event_id) * 2 AS workload
                   FROM review_video_assignments a
                   JOIN review_events e ON e.run_id=a.run_id AND e.video_id=a.video_id
                   WHERE e.state <> 'reviewed' GROUP BY a.primary_user_id"""
            ).fetchall()
            loads = {row["primary_user_id"]: float(row["workload"]) for row in load_rows}
            weights = {row["user_id"]: float(row["capacity_weight"]) for row in reviewers}
            pending = connection.execute(
                """SELECT e.run_id, e.video_id,
                          sum(e.duration_s) + count(*) * 2 AS workload
                   FROM review_events e
                   LEFT JOIN review_video_assignments a
                     ON a.run_id=e.run_id AND a.video_id=e.video_id
                   WHERE e.state <> 'reviewed' AND a.video_id IS NULL
                   GROUP BY e.run_id, e.video_id
                   ORDER BY workload DESC, e.video_id"""
            ).fetchall()
            assigned = 0
            for item in pending:
                reviewer = min(weights, key=lambda key: (loads.get(key, 0.0) / weights[key], key))
                inserted = connection.execute(
                    """INSERT INTO review_video_assignments(
                           run_id, video_id, primary_user_id, assigned_by)
                       VALUES (%s,%s,%s,%s) ON CONFLICT DO NOTHING
                       RETURNING video_id""",
                    (item["run_id"], item["video_id"], reviewer, assigned_by),
                ).fetchone()
                if inserted:
                    loads[reviewer] = loads.get(reviewer, 0.0) + float(item["workload"])
                    assigned += 1
        return {"assigned_videos": assigned, "reviewers": len(reviewers), "workload": loads}

    def assignment_stats(self) -> list[Dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT u.user_id, u.display_name, u.role, u.active,
                          count(DISTINCT a.video_id) AS videos,
                          count(e.event_id) FILTER (WHERE e.state <> 'reviewed') AS open_events,
                          count(e.event_id) FILTER (WHERE e.state = 'reviewed') AS reviewed_events
                   FROM review_users u
                   LEFT JOIN review_video_assignments a ON a.primary_user_id=u.user_id
                   LEFT JOIN review_events e ON e.run_id=a.run_id AND e.video_id=a.video_id
                   GROUP BY u.user_id ORDER BY u.display_name"""
            ).fetchall()
        return [_jsonable(row) or {} for row in rows]

    def event_assignee(self, event_id: str) -> Optional[str]:
        with self._connect() as connection:
            row = connection.execute(
                """SELECT a.primary_user_id FROM review_events e
                   LEFT JOIN review_video_assignments a
                     ON a.run_id=e.run_id AND a.video_id=e.video_id
                   WHERE e.event_id=%s""",
                (event_id,),
            ).fetchone()
        return str(row["primary_user_id"]) if row and row.get("primary_user_id") else None


def load_event_file(path: Path) -> list[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("events 文件必须是 JSON 数组")
    required = {"event_id", "video_id", "kind", "start_s", "end_s", "duration_s"}
    for index, event in enumerate(data):
        if not isinstance(event, dict) or not required.issubset(event):
            raise ValueError(f"events[{index}] 缺少必需字段")
    return data
