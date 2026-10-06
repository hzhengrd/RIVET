#!/usr/bin/env bash
# Pooled dual-hand improvement runs: long-tail augmentation x hand anchoring.
#
#   CUDA_VISIBLE_DEVICES=3 CELL=parent_aug bash scripts/havid.sh
#   CUDA_VISIBLE_DEVICES=3 CELL=parent     bash ...     # no-augmentation control
#   CUDA_VISIBLE_DEVICES=0 CELL=hand_aug     bash ...   # hand-anchored adapter
#   CUDA_VISIBLE_DEVICES=2 CELL=spatial_aug  bash ...   # spatial adapter
#   EPOCHS=4
#
# Every cell evaluates on test_lh (584) and test_rh (596) after each epoch, which
# is how the first pooled run was scored, so the numbers are directly comparable:
#     pooled epoch 4:  lh 0.5856   rh 0.5285
#
# CELL -> (variant, train split, runner)
#   parent      rivet_r_sft_no_som  train      stock trainer
#   parent_aug  rivet_r_sft_no_som  train_aug  stock trainer
#   spatial / spatial_aug  d_L0_v2  train(_aug) run_spatial
#   hand / hand_aug        d_L0_v3  train(_aug) run_hand_anchored
set -uo pipefail
cd "${ROOT:-$PWD}"

CELL="${CELL:?set CELL=parent|parent_aug|spatial|spatial_aug|spatial_aug_ep4|hand|hand_aug|hand_aug_ep4}"
EPOCHS="${EPOCHS:-4}"
START_EPOCH="${START_EPOCH:-1}"
SEED="${SEED:-17}"
CFG="${CFG:-configs/havid_aug.yaml}"
MATRIX="${MATRIX:-configs/method.yaml}"
RUN="python -m"

case "$CELL" in
  parent)     VARIANT=rivet_r_sft_no_som; SPLIT=train;     TRAIN_MOD="model.train"; INFER_MOD="model.infer" ;;
  parent_aug) VARIANT=rivet_r_sft_no_som; SPLIT=train_aug; TRAIN_MOD="model.train"; INFER_MOD="model.infer" ;;
  spatial)         VARIANT=d_L0_v2; SPLIT=train;     TRAIN_MOD="model.run_spatial train"; INFER_MOD="model.run_spatial infer" ;;
  spatial_aug)     VARIANT=d_L0_v2; SPLIT=train_aug; TRAIN_MOD="model.run_spatial train"; INFER_MOD="model.run_spatial infer" ;;
  spatial_aug_ep4) VARIANT=d_L0_v2_ep4; SPLIT=train_aug; TRAIN_MOD="model.run_spatial train"; INFER_MOD="model.run_spatial infer" ;;
  hand)            VARIANT=d_L0_v3; SPLIT=train;     TRAIN_MOD="model.run_hand_anchored train"; INFER_MOD="model.run_hand_anchored infer" ;;
  hand_aug)        VARIANT=d_L0_v3; SPLIT=train_aug; TRAIN_MOD="model.run_hand_anchored train"; INFER_MOD="model.run_hand_anchored infer" ;;
  hand_aug_ep4)    VARIANT=d_L0_v3_ep4; SPLIT=train_aug; TRAIN_MOD="model.run_hand_anchored train"; INFER_MOD="model.run_hand_anchored infer" ;;
  *) echo "bad CELL=$CELL"; exit 1 ;;
esac

LOG_DIR="outputs/havid/logs"; mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/${CELL}.log"
log() { echo "[$(date -Iseconds)] $*" | tee -a "$LOG"; }

preflight() {
  log "PREFLIGHT cell=$CELL variant=$VARIANT split=$SPLIT"
  python - "$CFG" "$MATRIX" "$VARIANT" "$SPLIT" "$CELL" <<'PY'
import glob, json, os, sys
import numpy as np
from model.config import load_config, artifact_path, feature_root
from model.experiment import variant_spec
from model.train import _resolve_parent
cfgp, matrix, variant, split, cell = sys.argv[1:6]
cfg = load_config(cfgp)
spec = variant_spec(cfg, variant, matrix)
print(f"  use_som={spec['use_som']}  rapt={spec.get('rapt')}  init={spec.get('init_variant')}")
if spec["use_som"]:
    sys.exit("FATAL: every cell here must be no-SoM")
if spec.get("rapt"):
    p = _resolve_parent(cfg, spec, split, 17)
    if p is None or not (p / "complete.json").exists():
        sys.exit(f"FATAL: parent for split={split} not trained yet ({p}).\n"
                 f"       run CELL=parent{'_aug' if split.endswith('aug') else ''} first")
    print(f"  parent OK: {p}")
root = "artifacts/havid"
need_boxes = cell.startswith("hand")
for name in (split, "test_lh", "test_rh"):
    rows = [json.loads(l) for l in open(artifact_path(cfg, "manifests", f"{name}.jsonl"))]
    ev = sum(1 for r in rows if os.path.exists(f"{root}/evidence/{r['clip_id']}/meta.json"))
    ft = sum(1 for r in rows if glob.glob(f"{feature_root(cfg)}/{r['clip_id']}.np*"))
    bx = sum(1 for r in rows if os.path.exists(f"{root}/hand_boxes/{r['clip_id']}.json"))
    print(f"  {name}: {len(rows)} rows  evidence={ev} features={ft} boxes={bx}")
    if ev != len(rows) or ft != len(rows):
        sys.exit(f"FATAL: {name} missing evidence or features")
    if need_boxes and bx < 0.9 * len(rows):
        sys.exit(f"FATAL: {name} hand_boxes coverage {bx}/{len(rows)} too low for a hand-anchored cell")
a = np.load(glob.glob(f"{feature_root(cfg)}/*.npy")[0], mmap_mode="r")
if a.shape[-1] <= 96:
    sys.exit(f"FATAL: feature dim {a.shape[-1]} is the histogram fallback")
print(f"  features OK: D={a.shape[-1]}")
PY
}

preflight || exit 1
for E in $(seq "$START_EPOCH" "$EPOCHS"); do
  log "===== TRAIN $VARIANT ($SPLIT) -> epoch $E/$EPOCHS ====="
  $RUN $TRAIN_MOD --config "$CFG" --variant "$VARIANT" --matrix "$MATRIX" \
    --train_split "$SPLIT" --seed "$SEED" --resume --stop_after_epoch "$E" || exit $?
  for H in lh rh; do
    log "INFER epoch $E on test_$H"
    $RUN $INFER_MOD --config "$CFG" --variant "$VARIANT" --matrix "$MATRIX" \
      --train_split "$SPLIT" --seed "$SEED" --split "test_$H" --output_suffix "epoch_$E"
  done
  log "EPOCH $E DONE"
done
log "DUAL-IMPROVE $CELL DONE"
