#!/usr/bin/env bash
# Assembly101 single-stream run: head-capped subsample + long-tail augmentation,
# then the two-stage recipe validated on HA-ViD.
#
#   STAGE=data                          bash scripts/assembly101.sh
#   CUDA_VISIBLE_DEVICES=2 STAGE=train  bash ...        # parent, 3 epochs
#   CUDA_VISIBLE_DEVICES=2 STAGE=test   bash ...        # full test on one checkpoint
#
# Deviations from the HA-ViD cells, both forced by the dataset and reported as
# such in the paper:
#   * no hand-conditioning clause -- every row is target_hand "both"
#   * degenerate status slot (vocabulary {active}), so 5-slot exact is 4-slot
#
# Schedule. 21,335 rows x 2.494 steps/row ~= 53,200 steps per epoch, so one
# epoch here already exceeds the 58,108-step four-epoch HA-ViD schedule. Three
# epochs are scheduled. Per-epoch monitoring uses a class-stratified 2,519-clip
# subset of validation (<=3 per class); the full 21,776-clip test split is run
# once, on the selected checkpoint, by STAGE=test.
set -uo pipefail
cd "${ROOT:-$PWD}"

STAGE="${STAGE:-all}"
EPOCHS="${EPOCHS:-3}"
SEED="${SEED:-17}"
SHARDS="${SHARDS:-12}"
CAP="${CAP:-20}"
DEVICE="${DEVICE:-cuda}"
CFG="${CFG:-configs/assembly101_aug.yaml}"
MATRIX="${MATRIX:-configs/method.yaml}"
EV_CFG="${EV_CFG:-configs/evidence_assembly101_aug.yaml}"
VARIANT="${VARIANT:-a101_parent}"
SPLIT=train_aug
MON=validation_mon
ROOT_DIR=artifacts/assembly101
RUN="python -m"
LOG_DIR="outputs/assembly101/logs"; mkdir -p "$LOG_DIR"
log() { echo "[$(date -Iseconds)] $*" | tee -a "$LOG_DIR/a101.log"; }

stage_data() {
  log "===== SUBSAMPLE (head cap $CAP) ====="
  $RUN model.subsample \
      --manifest "$ROOT_DIR/manifests/train.jsonl" \
      --out "$ROOT_DIR/manifests/train_cap${CAP}.jsonl" --cap "$CAP"

  log "===== TAIL AUGMENTATION ====="
  $RUN model.augment \
      --manifest "$ROOT_DIR/manifests/train_cap${CAP}.jsonl" \
      --out_manifest "$ROOT_DIR/manifests/train_aug_only.jsonl" \
      --merged_manifest "$ROOT_DIR/manifests/train_aug.jsonl" \
      --video_root artifacts/augmented_clips/tail_a101 \
      --target "$CAP" --max_per_clip 4 --workers 24

  log "===== EVIDENCE for augmented clips ($SHARDS shards) ====="
  $RUN data.shard --root "$ROOT_DIR/manifests" \
      --splits train_aug_only --shards "$SHARDS"
  mkdir -p outputs/evidence_assembly101/logs
  local pids=()
  for i in $(seq 0 $((SHARDS - 1))); do
    local tag; tag=$(printf "%02d" "$i")
    $RUN data.evidence --config "$EV_CFG" build --resume \
        --splits "train_aug_only_sh${tag}" \
        > "outputs/evidence_assembly101/logs/sh${tag}.log" 2>&1 &
    pids+=($!)
  done
  wait "${pids[@]}"
  log "  augmented evidence: $(ls -d $ROOT_DIR/evidence/*_aug*/ 2>/dev/null | wc -l)"

  log "===== FEATURES for augmented clips ====="
  $RUN model.features_vlm --config "$CFG" \
      --splits train_aug_only --device "$DEVICE" --output_dirname features_vlm 2>&1 | tail -4

  stage_verify
}

stage_verify() {
  log "===== VERIFY ====="
  python scripts/check_assembly101.py 2>&1 | tee -a "$LOG_DIR/a101.log"
}

stage_train() {
  for E in $(seq 1 "$EPOCHS"); do
    log "===== TRAIN $VARIANT -> epoch $E/$EPOCHS ====="
    $RUN model.train --config "$CFG" --variant "$VARIANT" --matrix "$MATRIX" \
      --train_split "$SPLIT" --seed "$SEED" --resume --stop_after_epoch "$E" || exit $?
    log "INFER epoch $E on $MON (class-stratified validation subset)"
    $RUN model.infer --config "$CFG" --variant "$VARIANT" --matrix "$MATRIX" \
      --train_split "$SPLIT" --seed "$SEED" --split "$MON" --output_suffix "epoch_$E"
    log "EPOCH $E DONE"
  done
}

stage_stage2() {
  # The adapter class comes from the runner the matrix declares, not from the
  # config, so it is resolved here rather than hardcoded.
  local SV="${STAGE2_VARIANT:-a101_L0_v2}"
  local MOD
  MOD=$(python scripts/stage2.py "$CFG" "$MATRIX" "$SV") || exit 1
  log "===== STAGE 2 variant=$SV runner=$MOD ====="
  for E in $(seq 1 "${STAGE2_EPOCHS:-4}"); do
    log "TRAIN $SV -> epoch $E"
    $RUN "$MOD" train --config "$CFG" --variant "$SV" --matrix "$MATRIX" \
      --train_split "$SPLIT" --seed "$SEED" --resume --stop_after_epoch "$E" || exit $?
    log "INFER $SV epoch $E on $MON"
    $RUN "$MOD" infer --config "$CFG" --variant "$SV" --matrix "$MATRIX" \
      --train_split "$SPLIT" --seed "$SEED" --split "$MON" --output_suffix "epoch_$E"
    log "STAGE2 EPOCH $E DONE"
  done
}

stage_test() {
  log "===== FULL TEST (21,776 clips) variant=$VARIANT ====="
  $RUN model.infer --config "$CFG" --variant "$VARIANT" --matrix "$MATRIX" \
    --train_split "$SPLIT" --seed "$SEED" --split test --output_suffix "${TEST_TAG:-final}"
}

case "$STAGE" in
  data) stage_data ;;
  verify) stage_verify ;;
  train) stage_train ;;
  stage2) stage_stage2 ;;
  test) stage_test ;;
  all) stage_data; stage_train ;;
  *) echo "STAGE must be data|verify|train|stage2|test|all"; exit 1 ;;
esac
log "A101 $STAGE DONE"
