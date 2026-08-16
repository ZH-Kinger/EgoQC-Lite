#!/usr/bin/env python3
"""Run a disposable PostgreSQL integration check for review IAM and assignment."""

from __future__ import annotations

import os
import secrets
from urllib.parse import quote

import psycopg
from psycopg import sql

from egoqc.review_db import ReviewStore


def main() -> None:
    database_url = os.environ.get("DATABASE_URL", "postgresql:///egoqc?user=root")
    schema = "egoqc_iam_verify_" + secrets.token_hex(6)
    separator = "&" if "?" in database_url else "?"
    isolated_url = database_url + separator + "options=" + quote(f"-csearch_path={schema}")

    with psycopg.connect(database_url, autocommit=True) as connection:
        connection.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))

    try:
        store = ReviewStore(isolated_url)
        store.init_schema()
        imported = store.import_events(
            [
                {"event_id": "verify-a-1", "video_id": "video-a", "kind": "hand_absent", "start_s": 0, "end_s": 7, "duration_s": 7},
                {"event_id": "verify-a-2", "video_id": "video-a", "kind": "jitter", "start_s": 8, "end_s": 12, "duration_s": 4},
                {"event_id": "verify-b-1", "video_id": "video-b", "kind": "persistent_extra_hands", "start_s": 0, "end_s": 5, "duration_s": 5},
            ],
            dataset_name="iam-verification",
            run_name="integration",
        )
        alice = store.upsert_feishu_user({"open_id": "verify-alice", "name": "Alice"}, "admin")
        bob = store.upsert_feishu_user({"open_id": "verify-bob", "name": "Bob"}, "reviewer")

        flow = store.create_oauth_flow("/review.html")
        assert store.consume_oauth_flow(flow["state"])["return_to"] == "/review.html"
        assert store.consume_oauth_flow(flow["state"]) is None

        session = store.create_session(bob["user_id"], 600)
        assert store.session_user(session)["user_id"] == bob["user_id"]
        store.delete_session(session)
        assert store.session_user(session) is None

        result = store.assign_unassigned_videos(alice["user_id"])
        assert result["assigned_videos"] == 2
        assignees = {
            event_id: store.event_assignee(event_id)
            for event_id in ("verify-a-1", "verify-a-2", "verify-b-1")
        }
        assert assignees["verify-a-1"] == assignees["verify-a-2"]
        assert set(assignees.values()) == {alice["user_id"], bob["user_id"]}
        assert sum(row["videos"] for row in store.assignment_stats()) == 2
        assert imported["events"] == 3
        print("POSTGRES_IAM_OK", result, assignees)
    finally:
        with psycopg.connect(database_url, autocommit=True) as connection:
            connection.execute(
                sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema))
            )


if __name__ == "__main__":
    main()
