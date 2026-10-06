#!/usr/bin/env bash
# VIEW-1 pooled dual-hand model, same recipe as the view-0 parent_aug cell.
#
#   STAGE=data VIEW=frontal                   bash scripts/havid_view.sh
#   CUDA_VISIBLE_DEVICES=1 STAGE=train        bash ...
#   CUDA_VISIBLE_DEVICES=1 STAGE=all          bash ...
#
# The point of this run is a viewpoint control, so the configuration is held
# identical to the view-0 cell it mirrors -- same variant, same matrix, same
# no-SoM spec, same 4 epochs, same per-epoch evaluation on test_lh / test_rh:
#
#   view 0  parent_aug (artifacts/havid,    train_aug)  ep4: lh 0.6729 rh 0.6980
#   view 1  this run   (artifacts/havid_frontal, train_aug)  ep4: ?
#
# Only the artifact root differs. If the view-0 finding -- pooled right hand
# beating the right-hand specialist, and the left-right gap reversing -- shows up
# here too, it is a property of the method rather than of one camera angle.
#
# STAGE=data covers: pooled split -> tail augmentation -> evidence for the new
# clips -> hand boxes -> vision features. Each step resumes, so re-running is
# safe and cheap.
set -uo pipefail
cd "${ROOT:-$PWD}"

VIEW="${VIEW:-frontal}"
STAGE="${STAGE:-all}"
EPOCHS="${EPOCHS:-4}"
SEED="${SEED:-17}"
SHARDS="${SHARDS:-10}"
DEVICE="${DEVICE:-cuda}"
case "$VIEW" in
  v1|frontal) PLACE=frontal ;;
  v2|overhead) PLACE=overhead ;;
  *) echo "VIEW must be frontal or overhead"; exit 1 ;;
esac
CFG="${CFG:-configs/havid_${PLACE}_aug.yaml}"
MATRIX="${MATRIX:-configs/method.yaml}"
EV_CFG="${EV_CFG:-configs/evidence_havid_${PLACE}_aug.yaml}"
VARIANT=rivet_r_sft_no_som
SPLIT=train_aug
ROOT_DIR=artifacts/havid_${PLACE}
RUN="python -m"
LOG_DIR="outputs/havid_${PLACE}/logs"; mkdir -p "$LOG_DIR"
log() { echo "[$(date -Iseconds)] $*" | tee -a "$LOG_DIR/train.log"; }

stage_data() {
  log "===== POOLED SPLIT ====="
  $RUN model.pool_views --view "$VIEW"

  log "===== TAIL AUGMENTATION ====="
  $RUN model.augment \
      --manifest "$ROOT_DIR/manifests/train.jsonl" \
      --out_manifest "$ROOT_DIR/manifests/train_aug_only.jsonl" \
      --merged_manifest "$ROOT_DIR/manifests/train_aug.jsonl" \
      --video_root artifacts/augmented_clips/tail_${PLACE} --workers 24

  log "===== EVIDENCE for augmented clips ($SHARDS shards) ====="
  $RUN data.shard --root "$ROOT_DIR/manifests" \
      --splits train_aug_only --shards "$SHARDS"
  mkdir -p outputs/evidence_havid_${PLACE}_aug/logs
  local pids=()
  for i in $(seq 0 $((SHARDS - 1))); do
    local tag; tag=$(printf "%02d" "$i")
    $RUN data.evidence --config "$EV_CFG" build --resume \
        --splits "train_aug_only_sh${tag}" \
        > "outputs/evidence_havid_${PLACE}_aug/logs/sh${tag}.log" 2>&1 &
    pids+=($!)
  done
  wait "${pids[@]}"
  log "  augmented evidence dirs: $(ls -d $ROOT_DIR/evidence/*_aug*/ 2>/dev/null | wc -l)"

  log "===== HAND BOXES ====="
  $RUN model.hand_boxes --config "$CFG" \
      --ev_config "configs/evidence_havid_${PLACE}_left.yaml" \
      --splits "train,train_aug_only,test_lh,test_rh" --workers 12 \
      2>&1 | grep -viE "feedback manager|gl_context|TensorFlow|landmark_proj" | tail -4

  log "===== FEATURES for augmented clips ====="
  $RUN model.features_vlm --config "$CFG" \
      --splits train_aug_only --device "$DEVICE" --output_dirname features_vlm 2>&1 | tail -4

  stage_verify
}

stage_verify() {
  log "===== VERIFY ====="
  python scripts/check_havid_view.py "$PLACE" 2>&1 | tee -a "$LOG_DIR/train.log"
}

stage_train() {
  for E in $(seq 1 "$EPOCHS"); do
    log "===== TRAIN $VARIANT ($VIEW $SPLIT) -> epoch $E/$EPOCHS ====="
    $RUN model.train --config "$CFG" --variant "$VARIANT" --matrix "$MATRIX" \
      --train_split "$SPLIT" --seed "$SEED" --resume --stop_after_epoch "$E" || exit $?
    for H in lh rh; do
      log "INFER epoch $E on test_$H"
      $RUN model.infer --config "$CFG" --variant "$VARIANT" --matrix "$MATRIX" \
        --train_split "$SPLIT" --seed "$SEED" --split "test_$H" --output_suffix "epoch_$E"
    done
    log "EPOCH $E DONE"
  done
}

case "$STAGE" in
  data) stage_data ;;
  verify) stage_verify ;;
  train) stage_train ;;
  all) stage_data; stage_train ;;
  *) echo "STAGE must be data|verify|train|all"; exit 1 ;;
esac
log "DUAL-$VIEW $STAGE DONE"
