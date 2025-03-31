mkdir -p ~/ar4_ws/src
cd ~/ar4_ws/src
git clone https://github.com/ycheng517/ar4_ros_driver
git clone https://github.com/ycheng517/ar4_hand_eye_calibration
git clone https://github.com/koghalai123/ROS2Practice
vcs import . --input ar4_hand_eye_calibration/hand_eye_calibration.repos
sudo apt install ros-jazzy-librealsense2* ros-jazzy-realsense2-*
sudo apt update && rosdep install --from-paths . --ignore-src -y
cd ~/ar4_ws
colcon build
source install/setup.bash

To give commands for gazebo simulation: 
ros2 launch annin_ar4_gazebo gazebo.launch.py
ros2 launch annin_ar4_moveit_config moveit.launch.py use_sim_time:=true include_gripper:=True
ros2 run ar4_practice joint_commander.py
