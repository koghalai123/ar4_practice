from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='ar4_practice',
            executable='joint_commander.py',
            output='screen',
            emulate_tty=True,
            parameters=[],
            arguments=[],
            shell=True,  # Allow interactive terminal input
        )
    ])