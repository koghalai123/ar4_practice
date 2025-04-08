# Installation instructions
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

colcon build

source install/setup.bash

# To run in Docker(you wont have to install Linux and can work on Windows): 
Install VS Code(not Visual Studio, you need Visual Studio Code). Install Windows Susbsytem for Linux(WSL). Copy the Dockerfile from this repo into a folder on your computer. Use VS Code to open that folder, then open a terminal window within VS Code. It defaults to opening a powershell window, but it has options nearby to open a WSL terminal window, which is what you need. You may need to install Docker and WSL related extensions within VS Code(You can find them from the buttons towards the left of the VS Code GUI). Within the terminal window, use the instructions here to install Docker for Ubuntu: https://docs.docker.com/engine/install/ubuntu/. Then run the following code in the terminal window to create a Docker image and run the container(if there are any issues with this step, let me know):

sudo usermod -aG docker $USER

sudo docker build -t ar4_sim .

Now change permissions and restart your terminal:

sudo apt update

sudo apt install x11-xserver-utils mesa-utils

xhost +local:docker

docker run -it --rm \
  -p 8080:8080 \
  -p 11345:11345 \
  -p 6080:6080 \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -e DISPLAY=$DISPLAY \
  -v "$HOME/.Xauthority:/root/.Xauthority:rw" \
  -e QT_X11_NO_MITSHM=1 \
  --privileged \
  ar4_sim

Your terminal windows should have closed. Give it a few minutes to restart(I have had to give it about 5 minutes for VS Code to work, even though ). Next, open up a new WSL terminal instance. A Docker container is available, but you may have issues opening and working with it, so you can open a VS Code instance fully within WSL, rather than a VS Code instance on Windows with a WSL terminal

code .

sudo apt install -y libgl1-mesa-dev libglu1-mesa-dev freeglut3-dev qtbase5-dev qtchooser qt5-qmake qtbase5-dev-tools

sudo apt install -y x11-apps

xeyes  # Should show moving eyes on Windows

Install the Docker and Dev Containers extensions for VS Code on the new VS Code instance and you should see a container available for usage.

To open multiple terminal windows within VSCode, there is a "+" button in the terminal area which will allow you to open multiple instances, which is necessary to run multiple programs at a time for controlling or simulating the robot.

# To give actual commands to the physical robot(each in its own terminal window): 
ros2 launch annin_ar4_driver driver.launch.py calibrate:=True include_gripper:=True

ros2 launch annin_ar4_moveit_config moveit.launch.py use_sim_time:=true include_gripper:=True

ros2 run ar4_practice joint_commander.py



# To give commands for gazebo simulation(each in its own terminal window): 
ros2 launch annin_ar4_gazebo gazebo.launch.py

ros2 launch annin_ar4_moveit_config moveit.launch.py use_sim_time:=true include_gripper:=True

ros2 run ar4_practice joint_commander.py

