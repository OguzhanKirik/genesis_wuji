#!/bin/bash
# Refresh the files this package borrows from adam-moveit, so it builds and runs without that workspace.
#   deploy/genesis_policy/scripts/vendor_from_adam_moveit.sh [/path/to/adam-moveit] [git-ref]
# The URDF is generated from genesis_description's xacro (design:=genesis_wuji, ros2_control:=false), which
# needs xacro and a built adam-moveit workspace (its install tree): run this inside the adam-moveit container. The JSON and YAML files are copied as they are. assets/PROVENANCE.txt records the source commit.
set -euo pipefail
SRC=${1:-$HOME/adam-moveit}
REF=${2:-HEAD}
HERE=$(cd "$(dirname "$0")/.." && pwd)
A=$HERE/assets
mkdir -p "$A/workcell" "$A/config"
show() { git -C "$SRC" show "$REF:$1"; }
for f in workcell.json workcell_bin_right.json workcell_single_right.json; do
  show ws/src/genesis_isaac/assets/workcell/$f > "$A/workcell/$f"
done
show ws/src/genesis_robot_config/config/ros2_control/ros2_controllers_genesis_wuji.yaml > "$A/config/ros2_controllers_genesis_wuji.yaml"
show ws/src/genesis_robot_config/config/moveit/joint_limits_genesis_wuji.yaml > "$A/config/joint_limits_genesis_wuji.yaml"
# The URDF: xacro needs genesis_description AND genesis_robot_config (the designs table), so it runs against a
# built adam-moveit workspace (WS, default /root/ws as the container mounts it; ws/ of a checkout on the host).
WS=${WS:-/root/ws}
if [ ! -f "$WS/install/setup.bash" ] && [ -f "$SRC/ws/install/setup.bash" ]; then WS=$SRC/ws; fi
if [ ! -f "$WS/install/setup.bash" ]; then
  echo "no built workspace at $WS (WS=... to point at one); URDF not generated" >&2; exit 2
fi
# shellcheck disable=SC1091
set +u; source "$WS/install/setup.bash"; set -u
XACRO=$(ros2 pkg prefix genesis_description)/share/genesis_description/urdf/genesis.urdf.xacro
xacro "$XACRO" design:=genesis_wuji ros2_control:=false > "$A/genesis_wuji.urdf"
echo "$(git -C "$SRC" rev-parse --short "$REF") $(git -C "$SRC" log -1 --format=%ad --date=short "$REF") (adam-moveit $REF)" > "$A/PROVENANCE.txt"
echo "vendored from adam-moveit $REF into $A"
