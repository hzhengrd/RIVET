#!/usr/bin/env bash
# Frontal or overhead training material for both hands: crop -> manifests and
# grammar -> evidence -> Qwen3-VL vision features.
#
#   bash scripts/havid_prepare.sh
#   STAGE=evidence bash ...                                           # one stage
#   CUDA_VISIBLE_DEVICES=0 STAGE=features bash ...
#
# Nothing here touches the side view: everything lands under
# artifacts/havid_{frontal,overhead}_{left,right}
# and artifacts/havid_clips/{frontal,overhead}_{left,right}_cropped.
#
# The evidence builder has no --workers flag, but records() reads one manifest
# per name passed to --splits, so data.shard + N processes give
# parallelism with no change to the shipped builder. --resume skips any clip
# whose meta.json already exists, so shards are safe to re-run.
set -uo pipefail
cd "${ROOT:-$PWD}"

VIEW="${VIEW:-frontal}"
STAGE="${STAGE:-all}"
HANDS="${HANDS:-left right}"
SHARDS="${SHARDS:-6}"
WORKERS="${WORKERS:-24}"
DEVICE="${DEVICE:-cuda}"
RUN="python -m"
case "$VIEW" in
  v1|frontal) PLACE=frontal; CODE=v1 ;;
  v2|overhead) PLACE=overhead; CODE=v2 ;;
  *) echo "VIEW must be frontal or overhead"; exit 1 ;;
esac
LOG_DIR="outputs/havid_${PLACE}_prep/logs"; mkdir -p "$LOG_DIR"
log() { echo "[$(date -Iseconds)] $*" | tee -a "$LOG_DIR/prepare.log"; }

stage_clips() {
  log "===== CLIPS + MANIFESTS (crop 150,110,1000,670) ====="
  $RUN data.havid --view "$CODE" --workers "$WORKERS" 2>&1 | tee -a "$LOG_DIR/prep.log"
}

stage_evidence() {
  for h in $HANDS; do
    log "===== EVIDENCE $h ($SHARDS shards x 2 splits) ====="
    local root="artifacts/havid_${PLACE}_${h}"
    $RUN data.shard --root "$root/manifests" \
        --splits train test --shards "$SHARDS"
    mkdir -p "outputs/evidence_havid_${PLACE}_${h}/logs"
    local pids=()
    for sp in train test; do
      for i in $(seq 0 $((SHARDS - 1))); do
        local tag; tag=$(printf "%02d" "$i")
        $RUN data.evidence --config "configs/evidence_havid_${PLACE}_${h}.yaml" \
            build --resume --splits "${sp}_sh${tag}" \
            > "outputs/evidence_havid_${PLACE}_${h}/logs/${sp}_sh${tag}.log" 2>&1 &
        pids+=($!)
      done
    done
    log "  ${#pids[@]} shards running for $h"
    wait "${pids[@]}"
    log "  $h evidence dirs: $(ls -d "$root/evidence"/*/ 2>/dev/null | wc -l)"
  done
}

stage_features() {
  for h in $HANDS; do
    log "===== FEATURES $h (Qwen3-VL vision tower, D=1152) ====="
    $RUN model.features_vlm --config "configs/havid_${PLACE}_${h}.yaml" \
        --splits train,test --device "$DEVICE" --output_dirname features_vlm 2>&1 | tail -25
  done
}

stage_verify() {
  log "===== VERIFY ====="
  python scripts/check_havid.py "$PLACE" 2>&1 | tee -a "$LOG_DIR/prepare.log"
}

case "$STAGE" in
  clips) stage_clips ;;
  evidence) stage_evidence ;;
  features) stage_features ;;
  verify) stage_verify ;;
  all) stage_clips; stage_evidence; stage_verify ;;
  *) echo "STAGE must be clips|evidence|features|verify|all"; exit 1 ;;
esac
log "PREPARE $PLACE $STAGE DONE"
