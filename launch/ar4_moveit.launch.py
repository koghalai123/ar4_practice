import os
import tempfile
import yaml

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, FindExecutable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterFile
from launch_ros.substitutions import FindPackageShare

def load_yaml(package_name, file_name):
    package_path = get_package_share_directory(package_name)
    absolute_file_path = os.path.join(package_path, file_name)
    with open(absolute_file_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def generate_launch_description():
    declared_arguments = []
    declared_arguments.append(
        DeclareLaunchArgument(
            "use_sim_time",
            default_value="True",
            description="Use simulation (Gazebo) clock if true",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "ar_model",
            default_value="mk3",
            choices=["mk1", "mk2", "mk3"],
            description="Model of AR4",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "tf_prefix",
            default_value="",
            description="Prefix for AR4 tf_tree",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "include_gripper",
            default_value="True",
            description="Run the servo gripper",
            choices=["True", "False"],
        )
    )

    use_sim_time = LaunchConfiguration("use_sim_time")
    ar_model_config = LaunchConfiguration("ar_model")
    tf_prefix = LaunchConfiguration("tf_prefix")
    include_gripper = LaunchConfiguration("include_gripper")

    # Paths to your packages
    annin_ar4_description_dir = FindPackageShare("annin_ar4_description")
    annin_ar4_moveit_config_dir = FindPackageShare("annin_ar4_moveit_config")
    annin_ar4_gazebo_dir = FindPackageShare("annin_ar4_gazebo")

    # Define robot description content from your URDF xacro
    robot_description_content = Command([
        PathJoinSubstitution([FindExecutable(name="xacro")]),
        " ",
        PathJoinSubstitution([annin_ar4_description_dir, "urdf", "ar_gazebo.urdf.xacro"]),
        " ",
        "ar_model:=", ar_model_config,
        " ",
        "tf_prefix:=", tf_prefix,
        " ",
        "include_gripper:=", include_gripper,
    ])
    robot_description = {"robot_description": robot_description_content}

    # Define semantic robot description from your SRDF xacro
    robot_description_semantic_content = Command([
        PathJoinSubstitution([FindExecutable(name="xacro")]),
        " ",
        PathJoinSubstitution([annin_ar4_moveit_config_dir, "srdf", "ar.srdf.xacro"]),
        " ",
        "name:=", ar_model_config,
        " ",
        "tf_prefix:=", tf_prefix,
        " ",
        "include_gripper:=", include_gripper,
    ])
    robot_description_semantic = {"robot_description_semantic": robot_description_semantic_content}

    # Gazebo launch
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([annin_ar4_gazebo_dir, "/launch", "/gazebo.launch.py"]),
        launch_arguments={'ar_model': ar_model_config, 'tf_prefix': tf_prefix}.items()
    )

    # Robot State Publisher
    robot_state_publisher_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="both",
        parameters=[robot_description],
    )

    # Define planning pipeline configuration
    planning_pipeline_config = {
        "default_planning_pipeline": "ompl",
        "planning_pipelines": ["ompl", "pilz"],
        "ompl": load_yaml("annin_ar4_moveit_config", os.path.join("config", "ompl_planning.yaml")),
        "pilz": load_yaml("annin_ar4_moveit_config", os.path.join("config", "pilz_planning.yaml")),
    }

    # Start the actual move_group node/action server
    move_group_node = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[
            robot_description,
            robot_description_semantic,
            load_yaml("annin_ar4_moveit_config", os.path.join("config", "kinematics.yaml")),
            load_yaml("annin_ar4_moveit_config", os.path.join("config", "joint_limits.yaml")),
            planning_pipeline_config, # <-- This is the added line
            load_yaml("annin_ar4_moveit_config", os.path.join("config", "controllers.yaml")),
            {"moveit_controller_manager": "moveit_simple_controller_manager/MoveItSimpleControllerManager"},
            {"use_sim_time": use_sim_time},
        ],
    )

    # RViz Node
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        output="log",
        arguments=["-d", PathJoinSubstitution([annin_ar4_moveit_config_dir, "rviz", "moveit.rviz"])],
        parameters=[
            robot_description,
            robot_description_semantic,
            load_yaml("annin_ar4_moveit_config", os.path.join("config", "kinematics.yaml")),
            load_yaml("annin_ar4_moveit_config", os.path.join("config", "ompl_planning.yaml")),
            {"use_sim_time": use_sim_time},
        ],
    )

    # Your Python script node
    ar4_python_node = Node(
        package='ar4_practice',
        executable='ar4_robot_moveit_py.py',
        name='ar4_robot_commander',
        output='screen',
        parameters=[
            robot_description,
            robot_description_semantic,
            load_yaml("annin_ar4_moveit_config", os.path.join("config", "kinematics.yaml")),
            load_yaml("annin_ar4_moveit_config", os.path.join("config", "ompl_planning.yaml")),
            {"use_sim_time": use_sim_time},
        ],
    )

    return LaunchDescription(
        declared_arguments + 
        [
            gazebo,
            robot_state_publisher_node,
            move_group_node,
            rviz_node,
            ar4_python_node,
        ]
    )