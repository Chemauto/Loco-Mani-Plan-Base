FROM ros:jazzy-ros-base

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    libgl1 libegl1 libgles2 mesa-utils \
    ros-jazzy-rviz2 ros-jazzy-robot-state-publisher \
    && rm -rf /var/lib/apt/lists/*

RUN pip install mujoco numpy pyyaml --break-system-packages

RUN echo "source /opt/ros/jazzy/setup.bash" >> /root/.bashrc
