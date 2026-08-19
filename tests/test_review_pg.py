import json
import threading
import urllib.error
import urllib.request
from functools import partial
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from egoqc.review_db import ReviewConflict, load_event_file
from egoqc.feishu_auth import FeishuAuthConfig
from egoqc.review_pg_server import REVIEW_HTML, _Handler
from egoqc.review_taxonomy import describe_error


class FakeStore:
    def __init__(self, clip: Path):
        self.event = {
            "event_id": "video-1--hand_absent-00",
            "video_id": "video-1",
            "kind": "hand_absent",
            "start_s": 1.0,
            "end_s": 2.5,
            "duration_s": 1.5,
            "clip_path": str(clip),
            "state": "pending",
            "claimed_by": None,
            "decision": None,
            "note": "",
            "version": 1,
        }

    def list_events(self, state=None, kind=None, limit=100, assigned_to=None):
        rows = [self.event.copy()]
        return [row for row in rows if (not state or row["state"] == state) and (not kind or row["kind"] == kind)][:limit]

    def stats(self, assigned_to=None):
        return {"total": 1, "states": {self.event["state"]: 1}, "kinds": {"hand_absent": 1}}

    def get_event(self, event_id):
        return self.event.copy() if event_id == self.event["event_id"] else None

    def claim(self, event_id, reviewer, lease_seconds=900):
        if self.event["state"] != "pending":
            raise ReviewConflict("busy")
        self.event.update(state="claimed", claimed_by=reviewer, version=2)
        return {**self.event, "claim_token": "token-1"}

    def decide(
        self, event_id, reviewer, claim_token, decision, note="", expected_version=None,
        details=None,
    ):
        if claim_token != "token-1" or reviewer != self.event["claimed_by"]:
            raise ReviewConflict("bad token")
        self.event.update(
            state="reviewed", decision=decision, decision_details=details or {},
            note=note, version=3,
        )
        return self.event.copy()

    def release(self, event_id, reviewer, claim_token):
        self.event.update(state="pending", claimed_by=None, version=3)
        return self.event.copy()

    def session_user(self, token):
        if token not in {"alice-token", "bob-token"}:
            return None
        name = token.removesuffix("-token")
        return {
            "user_id": f"feishu:{name}",
            "display_name": name.title(),
            "role": "reviewer",
        }

    def event_assignee(self, event_id):
        return "feishu:alice" if event_id == self.event["event_id"] else None


@pytest.fixture()
def review_server(tmp_path):
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"0123456789")
    store = FakeStore(clip)

    class Handler(_Handler):
        pass

    Handler.store = store
    Handler.evidence_root = tmp_path
    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(Handler, directory=str(tmp_path)))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}", store
    server.shutdown()
    server.server_close()


@pytest.fixture()
def authenticated_review_server(tmp_path):
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"0123456789")
    store = FakeStore(clip)

    class Handler(_Handler):
        pass

    Handler.store = store
    Handler.evidence_root = tmp_path
    Handler.auth_config = FeishuAuthConfig(
        "cli_test", "secret", "http://127.0.0.1:8767/auth/callback"
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(Handler, directory=str(tmp_path)))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    server.server_close()


def _post(url, payload):
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request) as response:
        return json.load(response)


def test_dynamic_review_api_claim_decide_and_range_media(review_server):
    base, _ = review_server
    with urllib.request.urlopen(base + "/api/review/events") as response:
        assert json.load(response)["events"][0]["state"] == "pending"
    claimed = _post(
        base + "/api/review/events/video-1--hand_absent-00/claim",
        {"reviewer": "alice"},
    )
    assert claimed["claim_token"] == "token-1"
    decided = _post(
        base + "/api/review/events/video-1--hand_absent-00/decision",
        {
            "reviewer": "alice",
            "claim_token": "token-1",
            "decision": "confirmed",
            "note": "visible issue",
            "expected_version": 2,
            "details": {"confidence": "high"},
        },
    )
    assert decided["state"] == "reviewed"
    assert decided["decision_details"]["confidence"] == "high"
    request = urllib.request.Request(
        base + "/api/review/media/video-1--hand_absent-00",
        headers={"Range": "bytes=2-5"},
    )
    with urllib.request.urlopen(request) as response:
        assert response.status == 206
        assert response.read() == b"2345"


def test_review_conflict_returns_409(review_server):
    base, store = review_server
    store.event.update(state="claimed", claimed_by="bob")
    with pytest.raises(urllib.error.HTTPError) as error:
        _post(base + "/api/review/events/video-1--hand_absent-00/claim", {"reviewer": "alice"})
    assert error.value.code == 409


def test_feishu_session_protects_api_and_assigned_media(authenticated_review_server):
    base = authenticated_review_server
    with pytest.raises(urllib.error.HTTPError) as error:
        urllib.request.urlopen(base + "/api/review/events")
    assert error.value.code == 401

    mine = urllib.request.Request(
        base + "/api/review/media/video-1--hand_absent-00",
        headers={"Cookie": "egoqc_session=alice-token"},
    )
    with urllib.request.urlopen(mine) as response:
        assert response.read() == b"0123456789"

    other = urllib.request.Request(
        base + "/api/review/media/video-1--hand_absent-00",
        headers={"Cookie": "egoqc_session=bob-token"},
    )
    with pytest.raises(urllib.error.HTTPError) as error:
        urllib.request.urlopen(other)
    assert error.value.code == 403


def test_load_event_file_validates_required_fields(tmp_path):
    path = tmp_path / "events.json"
    path.write_text(json.dumps([{"event_id": "missing"}]))
    with pytest.raises(ValueError, match="缺少必需字段"):
        load_event_file(path)


def test_error_taxonomy_and_type_filter_are_exposed():
    description = describe_error("persistent_extra_hands")
    assert description["category"] == "multi_person"
    assert description["severity"] == "reject"
    assert description["error_label"] == "疑似第二人手"
    assert 'id="kind-filter"' in REVIEW_HTML
    assert "button.textContent=decisionLabel" in REVIEW_HTML
    assert "data-filter=\"queue\" class=\"active\"" in REVIEW_HTML
    assert "filter:'queue'" in REVIEW_HTML
    assert "loadIdentity().then(refresh)" in REVIEW_HTML


def test_gold_review_defaults_to_machine_assisted_confirmation():
    assert "确认机器结论" in REVIEW_HTML
    assert "有误报，展开修改" in REVIEW_HTML
    assert "保存修正" in REVIEW_HTML
    assert "workflow_version:'assisted-fast-v1'" in REVIEW_HTML
    assert "goldDetails(e,'confirm_machine','confirmed')" in REVIEW_HTML
    assert "这个片段可能有什么问题" in REVIEW_HTML
    assert "data-seek-frame" in REVIEW_HTML
    assert "机器证据" in REVIEW_HTML
    assert "看到了什么（可多选）" not in REVIEW_HTML
    assert "首个坏点（秒，可空）" not in REVIEW_HTML
