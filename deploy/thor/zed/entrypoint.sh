#!/bin/bash
set -e
source /opt/ros/jazzy/setup.bash
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_cyclonedds_cpp}"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-1}"
echo "[zed] RMW=$RMW_IMPLEMENTATION ROS_DOMAIN_ID=$ROS_DOMAIN_ID" >&2
exec "$@"
