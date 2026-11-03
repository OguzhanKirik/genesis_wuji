"""The policy node against a running stack (sim or real).

    ros2 launch genesis_policy policy.launch.py policy_dir:=/handoff/policy/stub dry_run:=true
    ros2 launch genesis_policy policy.launch.py policy_dir:=/handoff/policy/<name>

Then enable the dead-man and keep the heartbeat alive:

    ros2 service call /policy_node/enable std_srvs/srv/SetBool "{data: true}"
    ros2 topic pub -r 10 /policy_node/heartbeat std_msgs/msg/Empty

In real mode (ADAM_MODE=real) the node refuses to start without
confirm_real_hardware:=true, exactly as real.launch.py does. Parameters not on
this line come from config/defaults.yaml, or from params_file:= on top.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

TRUTHY = ('true', '1', 'yes')


def _setup(context, *args, **kwargs):
    def arg(name):
        return LaunchConfiguration(name).perform(context)

    share = get_package_share_directory('genesis_policy')
    params = [os.path.join(share, 'config', 'defaults.yaml')]
    if arg('params_file'):
        params.append(arg('params_file'))
    overrides = {
        'policy_dir': arg('policy_dir'),
        'design': arg('design'),
        'dry_run': arg('dry_run').lower() in TRUTHY,
        'rate_hz': float(arg('rate_hz')),
        'confirm_real_hardware': arg('confirm_real_hardware').lower() in TRUTHY,
    }
    if arg('cell_file'):
        overrides['cell_file'] = arg('cell_file')
    params.append(overrides)
    return [
        Node(
            package='genesis_policy',
            executable='policy_node.py',
            name='policy_node',
            output='screen',
            ros_arguments=['--log-level', 'policy_node:=' + arg('log_level')],
            parameters=params,
        ),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('policy_dir', default_value='/handoff/policy/stub',
                              description='Directory with actor.onnx and level.yaml (the adam-rl handoff).'),
        DeclareLaunchArgument('design', default_value='genesis_wuji',
                              description='Body the stack runs; the controllers and URDF are read for it.'),
        DeclareLaunchArgument('dry_run', default_value='false',
                              description='true: run the policy, log and publish status, send no trajectory.'),
        DeclareLaunchArgument('rate_hz', default_value='0.0',
                              description='Policy rate; 0 takes level.yaml control.rate_hz.'),
        DeclareLaunchArgument('cell_file', default_value='',
                              description='Workcell JSON with the table and bin; empty = genesis_isaac workcell.json.'),
        DeclareLaunchArgument('params_file', default_value='',
                              description='Extra parameter YAML overlaid on config/defaults.yaml (fence, topics, ...).'),
        DeclareLaunchArgument('log_level', default_value='info'),
        DeclareLaunchArgument(
            'confirm_real_hardware', default_value='false',
            description='Must be true when ADAM_MODE=real. The mode in .env is not consent on its own.'),
        OpaqueFunction(function=_setup),
    ])
