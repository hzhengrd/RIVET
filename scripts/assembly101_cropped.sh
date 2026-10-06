#!/usr/bin/env bash
# Full pipeline for the workspace-cropped Assembly101 variant.
#
#   bash scripts/assembly101_cropped.sh
#   DEV=2 SHARDS=16 bash ...
#
# Waits for data.crop to finish, then builds
# every downstream artefact under a SEPARATE artifact root
# (artifacts/assembly101_cropped) so the uncropped build stays intact as the control.
# Evidence and features are keyed by clip_id, which the crop does not change, so
# a shared root would silently mix the two.
#
# Splits, head cap and augmentation plan are inherited unchanged from the
# uncropped build: crop_clips rewrites only the video paths.
set -uo pipefail
cd "${ROOT:-$PWD}"
# Activate the environment that has the requirements installed, if one is named.
if [ -n "${CONDA_ENV:-}" ]; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate "$CONDA_ENV"
fi

SRC=artifacts/assembly101
DST=artifacts/assembly101_cropped
CFG=configs/assembly101_cropped_aug.yaml
MATRIX=configs/method.yaml
DEV="${DEV:-2}"
SHARDS="${SHARDS:-16}"
N_CLIPS=58385
log() { echo "[a101c $(date +%H:%M:%S)] $*"; }

log "=== 0) wait for the spatial crop to finish ==="
while [ "$(find artifacts/assembly101_videos/clips_c -name '*.mp4' 2>/dev/null | wc -l)" -lt "$N_CLIPS" ]; do
  log "cropping $(find artifacts/assembly101_videos/clips_c -name '*.mp4' 2>/dev/null | wc -l)/$N_CLIPS"
  sleep 300
done
log "crop complete: $(find artifacts/assembly101_videos/clips_c -name '*.mp4' | wc -l) clips"

log "=== 1) manifests under the cropped root ==="
mkdir -p "$DST/manifests" "$DST/audits"
for s in train_cap200 validation_nat test; do
  cp "$SRC/manifests/${s}_c.jsonl" "$DST/manifests/${s}.jsonl"
  echo "  $s: $(wc -l < "$DST/manifests/${s}.jsonl") rows"
done
cp "$SRC/manifests/train_grammar.json" "$DST/manifests/"

log "=== 2) tail augmentation from the cropped clips (target 60) ==="
if [ ! -s "$DST/manifests/train_aug200_only.jsonl" ]; then
  python -m model.augment \
    --manifest "$DST/manifests/train_cap200.jsonl" \
    --out_manifest "$DST/manifests/train_aug200_only.jsonl" \
    --merged_manifest "$DST/manifests/train_aug200.jsonl" \
    --video_root artifacts/augmented_clips/tail_a101c \
    --probe_cache "$DST/audits/probe_cache.json" \
    --target 60 --max_per_clip 4 --workers 32 2>&1 | tail -6
fi
N_AUG=$(wc -l < "$DST/manifests/train_aug200_only.jsonl")
log "augmented=$N_AUG   total train rows=$(wc -l < "$DST/manifests/train_aug200.jsonl")"

log "=== 3) evidence, base clips ($SHARDS shards) ==="
python -m data.shard --root "$DST/manifests" \
    --splits train_cap200 validation_nat test --shards "$SHARDS"
mkdir -p outputs/evidence_assembly101_cropped/logs
for sp in train_cap200 validation_nat test; do
  for i in $(seq 0 $((SHARDS-1))); do
    t=$(printf "%02d" "$i")
    python -m data.evidence --config configs/evidence_assembly101_cropped.yaml \
        build --resume --splits "${sp}_sh${t}" \
        > "outputs/evidence_assembly101_cropped/logs/${sp}_sh${t}.log" 2>&1 &
  done
  wait
  log "evidence $sp done: $(find "$DST/evidence" -maxdepth 1 -mindepth 1 -type d 2>/dev/null | wc -l) total"
done

log "=== 4) evidence, augmented clips ==="
python -m data.shard --root "$DST/manifests" --splits train_aug200_only --shards "$SHARDS"
mkdir -p outputs/evidence_assembly101_cropped/logs
for i in $(seq 0 $((SHARDS-1))); do
  t=$(printf "%02d" "$i")
  python -m data.evidence --config configs/evidence_assembly101_cropped_aug.yaml \
      build --resume --splits "train_aug200_only_sh${t}" \
      > "outputs/evidence_assembly101_cropped/logs/sh${t}.log" 2>&1 &
done
wait
log "evidence complete: $(find "$DST/evidence" -maxdepth 1 -mindepth 1 -type d 2>/dev/null | wc -l)"

log "=== 5) vision features (GPU $DEV) ==="
CUDA_VISIBLE_DEVICES=$DEV python -m model.features_vlm --config "$CFG" \
    --splits train_cap200,train_aug200_only,validation_nat,test \
    --device cuda --output_dirname features_vlm 2>&1 | tail -5

log "=== 6) coverage check ==="
python - <<'PY'
import json, os
R = "artifacts/assembly101_cropped"
ev = set(os.listdir(f"{R}/evidence")); ft = {f.split(".")[0] for f in os.listdir(f"{R}/features_vlm")}
for sp in ("train_cap200", "train_aug200_only", "train_aug200", "validation_nat", "test"):
    p = f"{R}/manifests/{sp}.jsonl"
    if not os.path.exists(p):
        print(f"  {sp}: MISSING"); continue
    rows = [json.loads(l) for l in open(p)]
    v = sum(1 for r in rows if os.path.exists(r["video"]))
    e = sum(1 for r in rows if r["clip_id"] in ev)
    f = sum(1 for r in rows if r["clip_id"] in ft)
    print(f"  {'OK ' if v == e == f == len(rows) else 'GAP'} {sp:<20} {len(rows):>6} rows  "
          f"videos={v} evidence={e} features={f}")
PY

log "=== 7) train 3 epochs (GPU $DEV) ==="
for E in 1 2 3; do
  log "TRAIN epoch $E/3"
  CUDA_VISIBLE_DEVICES=$DEV python -m model.train \
    --config "$CFG" --variant a101c_parent --matrix "$MATRIX" \
    --train_split train_aug200 --seed 17 --resume --stop_after_epoch "$E" || exit $?
  log "INFER epoch $E on validation_nat"
  CUDA_VISIBLE_DEVICES=$DEV python -m model.infer \
    --config "$CFG" --variant a101c_parent --matrix "$MATRIX" \
    --train_split train_aug200 --seed 17 --split validation_nat --output_suffix "epoch_$E"
  log "EPOCH $E DONE"
done
log "ALL DONE"
