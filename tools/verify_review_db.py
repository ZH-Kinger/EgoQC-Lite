#!/usr/bin/env python3
"""Destructive only to its own __integration_probe__ rows; verifies real PostgreSQL flow."""

from __future__ import annotations

import os
import uuid

from egoqc.review_db import ReviewStore


def main() -> None:
    database_url = os.environ["DATABASE_URL"]
    suffix = uuid.uuid4().hex
    dataset_name = f"__integration_probe__-{suffix}"
    run_name = "claim-decide"
    event_id = f"__integration_probe__-{suffix}"
    store = ReviewStore(database_url)
    store.init_schema()
    imported = store.import_events(
        [{
            "event_id": event_id,
            "video_id": "probe",
            "kind": "hand_absent",
            "start_s": 1.0,
            "end_s": 2.1,
            "duration_s": 1.1,
        }],
        dataset_name,
        run_name,
    )
    try:
        claimed = store.claim(event_id, "integration-test", 60)
        reviewed = store.decide(
            event_id,
            "integration-test",
            claimed["claim_token"],
            "confirmed",
            "temporary probe",
            claimed["version"],
        )
        assert reviewed["state"] == "reviewed"
        assert reviewed["decision"] == "confirmed"
        print({"ok": True, "event_id": event_id, "state": reviewed["state"]})
    finally:
        with store._connect() as connection:
            connection.execute("DELETE FROM review_decisions WHERE event_id=%s", (event_id,))
            connection.execute("DELETE FROM review_events WHERE event_id=%s", (event_id,))
            connection.execute("DELETE FROM review_runs WHERE run_id=%s", (imported["run_id"],))
            connection.execute("DELETE FROM review_datasets WHERE dataset_id=%s", (imported["dataset_id"],))


if __name__ == "__main__":
    main()
