# The ZED camera source for the Thor: Stereolabs' runtime image for this L4T (ZED SDK + pyzed, CUDA from the host
# through the nvidia runtime) plus ROS 2 Jazzy's Python client and CycloneDDS. Jazzy, not Humble, because the base
# is Ubuntu 24.04; the messages it publishes (Image, CameraInfo, TF) are identical across distros, and with
# CycloneDDS on both sides the Humble containers read them.
#   docker build -t genesis-rl-zed:local zed/      (on the Thor; ZED_TAG matches the host's L4T: r39.2 = JetPack 7.2)
ARG ZED_TAG=5.5-py-runtime-l4t-r39.2
FROM stereolabs/zed:${ZED_TAG}
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends curl gnupg2 lsb-release ca-certificates \
    && curl -fsSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" > /etc/apt/sources.list.d/ros2.list \
    && apt-get update && apt-get install -y --no-install-recommends \
       ros-jazzy-ros-core ros-jazzy-rclpy ros-jazzy-sensor-msgs ros-jazzy-geometry-msgs ros-jazzy-tf2-ros-py \
       ros-jazzy-rmw-cyclonedds-cpp python3-numpy \
    && rm -rf /var/lib/apt/lists/*
COPY zed_source.py /opt/zed_source.py
COPY entrypoint.sh /opt/entrypoint.sh
RUN chmod +x /opt/entrypoint.sh /opt/zed_source.py
ENTRYPOINT ["/opt/entrypoint.sh"]
CMD ["python3", "/opt/zed_source.py"]
