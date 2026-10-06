#!/usr/bin/env bash
# Stage 0-2 for the RIGHT-hand (rh_v0) split: crop -> manifests+QA grammar ->
# evidence -> real Qwen3-VL vision features.
#
# Nothing here touches the left-hand artifacts: everything lands under
# artifacts/havid_side_right/ and artifacts/havid_clips/rh_v0_cropped/.
#
#   bash scripts/havid_side.sh            # all stages
#   STAGE=features CUDA_VISIBLE_DEVICES=1 bash ...                 # one stage
set -euo pipefail
ROOT="${1:-$PWD}"
cd "$ROOT"
CFG="${CFG:-configs/havid_side_right_vlm.yaml}"
EV_CFG="${EV_CFG:-configs/evidence_havid_side_right.yaml}"
STAGE="${STAGE:-all}"
WORKERS="${WORKERS:-24}"
DEVICE="${DEVICE:-cuda}"
RUN="python -m"
LOG_DIR="outputs/havid_side_right/logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/prepare.log"
log() { echo "[$(date -Iseconds)] $*" | tee -a "$LOG"; }

# 1) crop rh_v0 with the lh_v0 box + write manifests and the train grammar.
#    Splits come from rh_v0/{train_list_video.txt,val_list_video.txt};
#    `_extended_` and `w` clips are excluded.
stage_clips() {
  log "===== CLIPS+MANIFESTS (crop = lh_v0 box, drop _extended_ and w) ====="
  $RUN data.havid_side --workers "$WORKERS"
  log "clips+manifests done"
}

# 2) evidence: keyframes + badges + SoM marks + necessity QA + degraded video.
#    CPU only (som_device: cpu upstream; MediaPipe/motion are CPU).
stage_evidence() {
  log "===== EVIDENCE (CPU) ====="
  $RUN data.evidence --config "$EV_CFG" build --resume --splits train,test
  $RUN data.evidence --config "$EV_CFG" audit | tee -a "$LOG"
  log "evidence done"
}

# 3) Vision features. Uses the FIXED extractor (image_grid_thw is passed through,
#    so the Qwen3-VL tower actually runs); the stock model/features.py
#    silently falls back to 96-d colour histograms.
stage_features() {
  log "===== FEATURES (Qwen3-VL vision tower, D=1152) ====="
  $RUN model.features_vlm --config "$CFG" \
    --splits train,test --device "$DEVICE" --output_dirname features_vlm
  log "features done"
}

verify() {
  log "===== VERIFY ====="
  python - <<'PY' | tee -a "$LOG"
import json, glob, os
import numpy as np
root = "artifacts/havid_side_right"
for split in ("train", "test"):
    rows = [json.loads(l) for l in open(f"{root}/manifests/{split}.jsonl")]
    vids = sum(1 for r in rows if os.path.exists(r["video"]))
    ev = sum(1 for r in rows if os.path.exists(f"{root}/evidence/{r['clip_id']}/meta.json"))
    ft = sum(1 for r in rows if glob.glob(f"{root}/features_vlm/{r['clip_id']}.np*"))
    print(f"  {split:5s} clips={len(rows):5d}  videos={vids:5d}  evidence={ev:5d}  features={ft:5d}")
f = glob.glob(f"{root}/features_vlm/*.npy")
if f:
    a = np.load(f[0], mmap_mode="r")
    print(f"  feature shape {tuple(a.shape)}  dim={a.shape[-1]}"
          f"{'  !! looks like the histogram fallback' if a.shape[-1] <= 96 else '  (real vision features)'}")
PY
}

case "$STAGE" in
  clips)    stage_clips ;;
  evidence) stage_evidence ;;
  features) stage_features ;;
  verify)   verify ;;
  all)      stage_clips; stage_evidence; stage_features; verify ;;
  *) echo "STAGE must be clips|evidence|features|verify|all"; exit 1 ;;
esac
log "rh prepare OK ($STAGE)"
