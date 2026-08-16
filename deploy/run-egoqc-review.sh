#!/usr/bin/env bash
set -euo pipefail

EGOQC_FEISHU_ENV=${EGOQC_FEISHU_ENV:-/srv/egoqc/secrets/feishu.env}
if [[ -r "$EGOQC_FEISHU_ENV" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$EGOQC_FEISHU_ENV"
  set +a
fi

exec /srv/egoqc/app/.venv-review/bin/egoqc \
  serve-postgres-review \
  --evidence-root /srv/egoqc/results/review-evidence \
  --host 127.0.0.1 \
  --port 8767
