#!/bin/bash
# ./run.sh robot            the robot stack (adam-moveit real.launch.py, or sim.launch.py when ADAM_MODE=sim)
# ./run.sh app              a shell in the application container
# ./run.sh perception       the rod perception node on the head camera (params from config/perception.yaml)
# ./run.sh chain            perception + selector + policy node, left running (logs in logs/app)
# ./run.sh cycle [N]        N grasp cycles (default 1) against a running chain
# Each is idempotent: containers are created on first use and reused afterwards.
set -euo pipefail
cd "$(dirname "$0")"
[ -f .env ] || { echo "no .env: cp .env.example .env" >&2; exit 2; }
set -a; source .env; set +a
mkdir -p logs/robot logs/app policy config
APP_ENTRY=/usr/local/bin/genesis_rl_entrypoint.sh
POLICY_DIR=/handoff/policy/${POLICY_NAME:-L53b}
PERC_PARAMS=$([ -f config/perception.yaml ] && echo "params:=/config/perception.yaml" || echo "")
POLICY_PARAMS=$([ -f config/policy.yaml ] && echo "params_file:=/config/policy.yaml" || echo "")
case "${1:-}" in
  robot)
    docker compose up -d robot
    if [ "${ADAM_MODE:-real}" = real ]; then
      echo "REAL ROBOT: the launch below moves the arms once the dead-man and the policy enable are given." >&2
      docker exec -d adam_moveit bash -lc "ros2 launch genesis_cell real.launch.py design:=genesis_wuji confirm_real_hardware:=true > /root/logs/stack.log 2>&1"
    else
      docker exec -d adam_moveit bash -lc "ros2 launch genesis_cell sim.launch.py design:=genesis_wuji headless:=true rviz:=false > /root/logs/stack.log 2>&1"
    fi
    echo "robot stack starting; tail logs/robot/stack.log" ;;
  app)
    docker compose up -d app
    docker exec -it genesis_rl_app $APP_ENTRY bash ;;
  perception)
    docker compose up -d app
    docker exec -d genesis_rl_app $APP_ENTRY bash -c "ros2 launch genesis_perception perception.launch.py $PERC_PARAMS > /logs/perception.log 2>&1"
    echo "perception node started; tail logs/app/perception.log" ;;
  chain)
    docker compose up -d app
    docker exec -d genesis_rl_app $APP_ENTRY bash -c "ros2 launch genesis_perception perception.launch.py $PERC_PARAMS > /logs/perception.log 2>&1"
    docker exec -d genesis_rl_app $APP_ENTRY bash -c "ros2 run genesis_policy selector_node.py --ros-args -p level_path:=$POLICY_DIR/level.yaml > /logs/selector.log 2>&1"
    docker exec -d genesis_rl_app $APP_ENTRY bash -c "ros2 launch genesis_policy policy.launch.py policy_dir:=$POLICY_DIR $POLICY_PARAMS confirm_real_hardware:=${CONFIRM_REAL_HARDWARE:-false} > /logs/policy.log 2>&1"
    echo "perception, selector and policy node started; logs in logs/app/" ;;
  cycle)
    docker exec -it genesis_rl_app $APP_ENTRY bash -c "ros2 run genesis_policy grasp_cycle.py --ros-args -p level_path:=$POLICY_DIR/level.yaml -p cycles:=${2:-1} 2>&1 | tee /logs/cycle_\$(date +%F_%H%M%S).log" ;;
  *)
    sed -n '2,7p' "$0"; exit 1 ;;
esac
