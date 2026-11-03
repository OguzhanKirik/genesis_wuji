import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    defaults = os.path.join(get_package_share_directory('genesis_perception'), 'config', 'defaults.yaml')
    return LaunchDescription([
        DeclareLaunchArgument('params', default_value=defaults),
        DeclareLaunchArgument('segmenter', default_value='depth'),
        Node(package='genesis_perception', executable='perception_node', name='rod_perception', output='screen',
             parameters=[LaunchConfiguration('params'), {'segmenter': LaunchConfiguration('segmenter')}]),
    ])
