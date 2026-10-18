#!/usr/bin/env bash
# Watch a reach-grasp checkpoint in the Isaac window, from the container prompt.
#   bash /opt/tools/viz.sh LEVEL MODEL_NUMBER [eval flags...]      e.g.  bash /opt/tools/viz.sh 38 17100 --pile 1
#   bash /opt/tools/viz.sh LEVEL MODEL_NUMBER --stats [flags...]   headless, 2048 episodes with the quota rule
# The checkpoint is looked up as logs/rsl_rl/adam_u30_wuji_reach_grasp_L<LEVEL>/*/model_<MODEL_NUMBER>.pt.
set -eu
lvl=$1; num=$2; shift 2
ck=$(ls logs/rsl_rl/adam_u30_wuji_reach_grasp_L${lvl}/*/model_${num}.pt 2>/dev/null | head -1)
[ -n "$ck" ] || { echo "no model_${num}.pt under logs/rsl_rl/adam_u30_wuji_reach_grasp_L${lvl}/"; exit 1; }
if [ "${1:-}" = "--stats" ]; then
  shift
  exec env ADAM_U_RG_LEVEL=$lvl python /opt/tools/eval_reach_grasp.py --checkpoint_path "$ck" --num_envs 256 --episodes 2048 "$@"
fi
exec env ADAM_U_RG_LEVEL=$lvl python /opt/tools/eval_reach_grasp.py --checkpoint_path "$ck" --num_envs 16 --episodes 64 \
  --real-time --speed 0.5 --start-delay 15 --viz kit "$@"
