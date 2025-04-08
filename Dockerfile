FROM ros:jazzy

# Set environment variables to avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install VSCode (code-server) and other utilities
RUN apt-get update && apt-get install -y \
    curl \
    git \
    wget \
    python3-pip \
    && curl -fsSL https://code-server.dev/install.sh | sh \
    && rm -rf /var/lib/apt/lists/*

# Install Gazebo through ros-gz packages (simplified installation)
RUN apt-get update && apt-get install -y \
    ros-${ROS_DISTRO}-ros-gz \
    ros-${ROS_DISTRO}-robot-state-publisher \
    ros-${ROS_DISTRO}-joint-state-publisher \
    ros-${ROS_DISTRO}-xacro \
    ros-${ROS_DISTRO}-rviz2 \
    && rm -rf /var/lib/apt/lists/*

# Create workspace and clone all repositories
WORKDIR /ar4_ws
RUN mkdir -p src && cd src && \
    git clone https://github.com/ycheng517/ar4_ros_driver && \
    git clone https://github.com/ycheng517/ar4_hand_eye_calibration && \
    git clone https://github.com/koghalai123/ar4_practice

# Install dependencies (grouped for better caching)
RUN apt-get update && \
    rosdep update && \
    rosdep install --from-paths src --ignore-src -r -y && \
    apt-get install -y \
    ros-${ROS_DISTRO}-librealsense2* \
    ros-${ROS_DISTRO}-realsense2-* \
    && rm -rf /var/lib/apt/lists/*

# Import repos and build
RUN cd src && \
    . /opt/ros/${ROS_DISTRO}/setup.sh && \
    vcs import . --input ar4_hand_eye_calibration/hand_eye_calibration.repos && \
    cd /ar4_ws && \
    colcon build

# Set up environment sourcing
RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> /root/.bashrc && \
    echo "source /ar4_ws/install/setup.bash" >> /root/.bashrc && \
    echo "export GZ_SIM_RESOURCE_PATH=/ar4_ws/src/ar4_practice/models:\$GZ_SIM_RESOURCE_PATH" >> /root/.bashrc

# Expose VSCode port
EXPOSE 8080
# Expose Gazebo ports
EXPOSE 11345
EXPOSE 6080

# Default command (start VSCode and bash)
CMD ["/bin/sh", "-c", "code-server --auth none --bind-addr 0.0.0.0:8080 /ar4_ws & /bin/bash"]