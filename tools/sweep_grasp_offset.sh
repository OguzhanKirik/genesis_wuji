#!/usr/bin/env bash
# Graspable-area sweep along the rod's axis: evaluate one checkpoint with the
# commanded grasp point pinned at each offset, and tabulate.
#
#   tools/sweep_grasp_offset.sh <level> <checkpoint> [episodes] [offsets mm...]
#   tools/sweep_grasp_offset.sh 15 logs/rsl_rl/adam_u30_wuji_reach_grasp_L15/<run>/model_<k>.pt 512
#
# Runs from the host; each offset is one eval_reach_grasp.py run inside the
# container. Logs go to logs/sweeps/grasp_offset_<timestamp>/.
# Environment knobs:
#   PINCH=<deg>   also pin the pinch angle (level 16+); the angle goes into the directory name
#   UPRIGHT=0|1   pin the rod lying (0) or standing on its end (1) (level 18+); goes into the name
#   TILT=0|1      pin the rod leaning on a support block (1) or not (0) (level 20+); goes into the name
#   COVER=0|1     pin the cover block over one end of lying rods (1) or none (0) (level 23+); goes into the name
#   OUT=<dir>     reuse a sweep directory: offsets whose log already has a result are skipped (resume)
# Each eval is capped at 15 minutes (Isaac Sim occasionally hangs at scene creation); a hung or
# failed offset shows as '?' in the table and is retried on a resumed run.
set -uo pipefail
LEVEL=${1:?level}; CKPT=${2:?checkpoint}; EPISODES=${3:-512}; shift 3 || shift $#
OFFSETS=("$@"); [ ${#OFFSETS[@]} -eq 0 ] && OFFSETS=(-16 -14 -12 -10 -8 -6 -4 -2 0 2 4 6 8 10 12 14 16)
PINCH=${PINCH:-}; UPRIGHT=${UPRIGHT:-}; TILT=${TILT:-}; COVER=${COVER:-}; BED=${BED:-}
TAG=grasp_offset$( [ -n "$PINCH" ] && echo "_pinch${PINCH}" )$( [ -n "$UPRIGHT" ] && echo "_upright${UPRIGHT}" )$( [ -n "$TILT" ] && echo "_tilt${TILT}" )$( [ -n "$COVER" ] && echo "_cover${COVER}" )$( [ -n "$BED" ] && echo "_bed${BED}" )
OUT=${OUT:-logs/sweeps/${TAG}_$(date +%Y-%m-%d_%H-%M-%S)}; mkdir -p "$OUT"
EXTRA=(); [ -n "$PINCH" ] && EXTRA+=(--pinch-angle-deg "$PINCH"); [ -n "$UPRIGHT" ] && EXTRA+=(--upright "$UPRIGHT"); [ -n "$TILT" ] && EXTRA+=(--tilt "$TILT"); [ -n "$COVER" ] && EXTRA+=(--cover "$COVER"); [ -n "$BED" ] && EXTRA+=(--bed "$BED")
row() {  # <offset> -> one table line from its log
  local s=$1 f="$OUT/eval_$1.log"
  local succ inv drag lat ge
  succ=$(grep '^    lift-and-hold success' "$f" 2>/dev/null | awk '{print $3}')
  inv=$(grep 'invalid collisions  ' "$f" 2>/dev/null | awk '{print $3}')
  drag=$(grep '^    bolt drag during lift' "$f" 2>/dev/null | sed 's/.*p95 *\([0-9.]*\) mm.*/\1/')
  lat=$(grep 'latched (early grasp end)' "$f" 2>/dev/null | awk '{print $5}')
  ge=$(grep 'latched (early grasp end)' "$f" 2>/dev/null | sed 's/.*step *\([0-9.]*\).*/\1/')
  perr=$(grep 'pinch vs commanded point' "$f" 2>/dev/null | sed 's/.*mean *\([0-9.]*\) mm.*/\1/')
  printf "%-10s %-9s %-11s %-11s %-9s %-13s %-14s\n" "$s" "${succ:-?} %" "${inv:-?} %" "${drag:-?} mm" "${lat:-?} %" "${ge:-?}" "${perr:-?} mm"
}
for s in "${OFFSETS[@]}"; do
  if grep -q '^    lift-and-hold success' "$OUT/eval_$s.log" 2>/dev/null; then continue; fi
  timeout 900 docker compose exec -e ADAM_U_RG_LEVEL="$LEVEL" adam_rl python /opt/tools/eval_reach_grasp.py \
      --checkpoint_path "$CKPT" --num_envs 256 --episodes "$EPISODES" --grasp-offset-mm "$s" "${EXTRA[@]}" \
      > "$OUT/eval_$s.log" 2>&1 || echo "  (offset $s: eval exited $? -- hung or failed; rerun with OUT=$OUT to retry)"
  row "$s"
done
{ printf "%-10s %-9s %-11s %-11s %-9s %-13s %-14s\n" "offset mm" "success" "collisions" "drag p95" "latched" "grasp end" "pinch err mean"
  for s in "${OFFSETS[@]}"; do row "$s"; done; } | tee "$OUT/table.txt"
echo "logs in $OUT"
