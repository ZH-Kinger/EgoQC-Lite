# VITRA-compatible undistortion

EgoQC-Lite follows Microsoft VITRA commit
`b35517202b39d32a753fdd42014b2cc3c41fab58` and records the upstream repository,
script and geometry in every task. Raw videos are never modified.

## Geometry

- Ego4D: load VITRA `intrinsics_ori.K/D/xi` and `intrinsics_new.K`; when `xi>0`, use
  `cv2.omnidir.initUndistortRectifyMap` with `RECTIFY_PERSPECTIVE`, followed by cubic remap.
- EgoExo4D: load Project Aria device calibration, select `camera-rgb`, build the VITRA linear
  camera calibration `(1408,1408,f=412.5)`, then call `distort_by_calibration`.
- VITRA training crop for EgoExo4D is a separate downstream transform: resize 1408 to 448 and
  center-crop 256. Intrinsics must be transformed using the same resize/crop chain.

## Plan

```bash
egoqc plan-vitra-undistortion \
  --dataset-kind ego4d \
  --video-root /path/to/ego4d/v2/full_scale \
  --intrinsics-root /path/to/intrinsics/ego4d \
  --selection-list /path/to/vitra-selected-video-ids.txt \
  --save-root /path/to/derived/ego4d-undistorted \
  --output /path/to/quality/ego4d-undistortion-plan
```

EgoExo4D additionally requires `--aria-name-map`. The summary separately reports source population,
selected videos and selection coverage, so “100% of selected videos completed” is never confused with
“100% of raw videos processed”. Missing source, calibration or Aria mapping is blocked before GPU work.

The manifest is an orchestration contract for the official VITRA scripts. It fixes source/calibration
fingerprints, camera model, output geometry, encoding and upstream commit.

## Execute the official implementation

Install the optional dependencies and pin the upstream checkout:

```bash
pip install -e '.[vitra]'
git clone https://github.com/microsoft/VITRA /path/to/VITRA
git -C /path/to/VITRA checkout b35517202b39d32a753fdd42014b2cc3c41fab58

egoqc run-vitra-undistortion \
  --manifest /path/to/plan/ready.jsonl \
  --vitra-root /path/to/VITRA \
  --output /path/to/quality/undistortion-execution \
  --shard-count 8 --shard-index 0
```

Run shard indexes `0..7` independently. Stable task-id hashing makes retries deterministic. Each official
VITRA invocation writes into a same-filesystem temporary directory and publishes the final MP4 with an
atomic rename; per-video pass/fail logs are flushed immediately. Existing outputs are skipped unless
`--overwrite` is set. Raw videos are only opened for reading.

## Verify

```bash
egoqc verify-vitra-undistortion \
  --manifest /path/to/undistortion-tasks.jsonl \
  --output /path/to/quality/undistortion-verification
```

Verification fully decodes source and output, requires equal frame counts and FPS, checks EgoExo4D
output is 1408 square, and detects source changes after planning. This is an integrity gate only.
Geometric acceptance still requires calibration provenance, rectified intrinsics and MANO/keypoint
reprojection or human overlay evidence.
