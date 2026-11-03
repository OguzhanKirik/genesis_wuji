#!/bin/bash
# genesis_rl deploy container: ROS from the base image, adam-moveit's workspace when mounted (bench: /root/ws),
# then this application's overlay. ROS_DOMAIN_ID and RMW_IMPLEMENTATION come from the environment (compose).
set -e
set +u
[ -f /root/.adam_moveit_env ] && source /root/.adam_moveit_env
[ -f /root/ws/install/setup.bash ] && source /root/ws/install/setup.bash
source "${DEPLOY_WS:-/opt/genesis_rl_ws}/install/setup.bash"
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_cyclonedds_cpp}"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-42}"
echo "[genesis_rl] RMW=$RMW_IMPLEMENTATION ROS_DOMAIN_ID=$ROS_DOMAIN_ID overlay=${DEPLOY_WS:-/opt/genesis_rl_ws}" >&2
exec "$@"
