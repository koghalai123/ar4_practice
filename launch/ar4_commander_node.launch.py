import os
import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import Command, FindExecutable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    declared_arguments = []
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

    ar_model_config = LaunchConfiguration("ar_model")
    tf_prefix = LaunchConfiguration("tf_prefix")
    include_gripper = LaunchConfiguration("include_gripper")

    annin_ar4_description_dir = FindPackageShare("annin_ar4_description")
    annin_ar4_moveit_config_dir = FindPackageShare("annin_ar4_moveit_config")

    # Generate the robot description content from your XACRO file
    robot_description_content = Command([
        PathJoinSubstitution([FindExecutable(name="xacro")]),
        " ",
        PathJoinSubstitution([annin_ar4_description_dir, "urdf", "ar.urdf.xacro"]),
        " ",
        "ar_model:=", ar_model_config,
        " ",
        "tf_prefix:=", tf_prefix,
        " ",
        "include_gripper:=", include_gripper,
    ])
    robot_description = {"robot_description": robot_description_content}

    # Generate the semantic robot description from your SRDF file
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

    # Load planning configuration from YAML files
    planning_pipeline_config_path = os.path.join(
        get_package_share_directory("annin_ar4_moveit_config"), "config", "ompl_planning.yaml"
    )
    kinematics_config_path = os.path.join(
        get_package_share_directory("annin_ar4_moveit_config"), "config", "kinematics.yaml"
    )

    # Define your Python node, passing the loaded parameters
    ar4_python_node = Node(
        package='ar4_practice',
        executable='ar4_robot_moveit_py.py',
        name='ar4_robot_commander',
        output='screen',
        parameters=[
            robot_description,
            robot_description_semantic,
            {"planning_pipelines": load_yaml("annin_ar4_moveit_config", os.path.join("config", "planning_pipelines.yaml"))},
            {"ompl": load_yaml("annin_ar4_moveit_config", os.path.join("config", "ompl_planning.yaml"))},
            {"kinematics": load_yaml("annin_ar4_moveit_config", os.path.join("config", "kinematics.yaml"))},
            {"robot_description": robot_description_content},
            {"robot_description_semantic": robot_description_semantic_content},
        ],
    )
    return LaunchDescription(declared_arguments + [ar4_python_node])