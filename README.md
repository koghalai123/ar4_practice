# Installation instructions

This needs to run on linux. You can 'dual boot' your computer so that you can boot into either your standard OS or Linux. This link gives a reasonable guide on how to do it: https://www.tomshardware.com/software/linux/how-to-dual-boot-linux-and-windows-on-any-pc. I have personally tried Ubuntu 24.04 and Linux Mint Cinnamon 22.2(which was built from Ubuntu 24). Linux Mint is based on Ubuntu, so most programs that work on Ubuntu will also work on the corresponding version of Linux Mint.

Make sure ROS 2 Jazzy(https://docs.ros.org/en/jazzy/Installation/Alternatives/Ubuntu-Development-Setup.html), Moveit2(for Jazzy), and Gazebo Sim(Harmonic) are installed. To install ROS 2, you need to go to the provided link and install it. For the other two, the following commands work.

sudo apt install ros-jazzy-moveit

sudo apt-get install ros-${ROS_DISTRO}-ros-gz

Next, set up a workspace and get necessary code installed:

mkdir -p ~/ar4_ws/src

cd ~/ar4_ws/src

git clone https://github.com/ycheng517/ar4_ros_driver

git clone https://github.com/ycheng517/ar4_hand_eye_calibration

git clone https://github.com/koghalai123/ar4_practice

source /opt/ros/jazzy/setup.bash

vcs import . --input ar4_hand_eye_calibration/hand_eye_calibration.repos

sudo apt install ros-jazzy-librealsense2* ros-jazzy-realsense2-*

sudo apt update && rosdep install --from-paths . --ignore-src -y

cd ~/ar4_ws

rosdep install --from-paths . --ignore-src -r -y

sudo apt update

sudo apt install ros-$ROS_DISTRO-ament-cmake ros-$ROS_DISTRO-ament-cmake-python

sudo apt install ros-jazzy-controller-manager

sudo apt install ros-jazzy-ros-gz-sim

sudo apt install ros-jazzy-ros-gz-bridge

sudo apt install ros-jazzy-gz-ros2-control

sudo apt install ros-jazzy-ros2-control ros-jazzy-ros2-controllers

source /opt/ros/jazzy/setup.bash

colcon build

echo "source /opt/ros/jazzy/setup.bash" >> ~/.bashrc

echo "source ar4_ws/install/setup.bash" >> ~/.bashrc

source install/setup.bash

sudo apt install python3-pip

sudo apt install ros-jazzy-tf-transformations

pip install pandas --break-system-packages

# To give actual commands to the physical robot(each in its own terminal window): 
ros2 launch annin_ar4_driver driver.launch.py calibrate:=True include_gripper:=True

ros2 launch annin_ar4_moveit_config moveit.launch.py use_sim_time:=true include_gripper:=True

ros2 run ar4_practice joint_commander.py



# To give commands for gazebo simulation(each in its own terminal window): 
ros2 launch annin_ar4_gazebo gazebo.launch.py

ros2 launch annin_ar4_moveit_config moveit.launch.py use_sim_time:=true include_gripper:=True

ros2 run ar4_practice joint_commander.py

# To plot the workplace envelope in gazebo(each in its own terminal window): 
ros2 launch annin_ar4_gazebo gazebo.launch.py

ros2 launch annin_ar4_moveit_config moveit.launch.py use_sim_time:=true include_gripper:=True

Run "surfacePublisher.py" either through ROS2 or simply through python and add the surface publisher topic to the Rviz2 display to see the envelope. You need the faces.csv and vertices.csv files created through other scripts as well though.


# To display the work envelope and find the test cube, some additional packages are needed:

pip install shapely trimesh open3d alphashape descartes numpy-stl "pyglet<2" --break-system-packages

# To start Realsense node: 

ros2 run realsense2_camera realsense2_camera_node
