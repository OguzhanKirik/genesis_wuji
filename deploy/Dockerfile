# The grasp application image: genesis_rl's deploy packages on top of adam-moveit's robot image.
#
# adam-moveit's image carries ROS 2 (from conda, so it builds for x86 and for the Jetson Thor's aarch64),
# the genesis_interfaces message package (MoveTo, Stop) this application calls, and already pins pinocchio,
# onnxruntime and onnx for the policy runtime. This image adds only what the application needs and builds
# the two packages here. It never runs adam-moveit's stack: that is the other container.
#
#   docker build -f deploy/Dockerfile --build-arg ADAM_IMAGE_TAG=v0.3.0 -t genesis-rl-deploy:dev .
#   docker buildx build --platform linux/arm64 ...   # for the Thor, or build on the Thor itself
# BASE_IMAGE overrides the registry image: on the bench, adam-moveit:latest (the dev image, whose workspace is
# mounted at /root/ws rather than built in; the entrypoint sources it when present).
ARG ADAM_IMAGE_TAG=unset
ARG BASE_IMAGE=ghcr.io/revel-robotics/adam-moveit:${ADAM_IMAGE_TAG}
FROM ${BASE_IMAGE}

ARG DEPLOY_WS=/opt/genesis_rl_ws
ENV DEPLOY_WS=${DEPLOY_WS}

# scipy for the cylinder fit; ultralytics only for the YOLO segmenter (CPU wheel of torch on x86; on the Thor
# install NVIDIA's torch wheel for JetPack instead and keep ultralytics without its torch dependency)
RUN pip install --no-cache-dir "scipy>=1.10" "pyyaml>=6" "numpy<2"

COPY deploy/genesis_perception ${DEPLOY_WS}/src/genesis_perception
COPY deploy/genesis_policy ${DEPLOY_WS}/src/genesis_policy
# the exported policies (actor.onnx + level.yaml per name); docker-compose can mount a newer set over it
COPY handoff/policy /handoff/policy

COPY deploy/entrypoint.sh /usr/local/bin/genesis_rl_entrypoint.sh
SHELL ["/bin/bash", "-lc"]
RUN cd ${DEPLOY_WS} && colcon build --symlink-install --event-handlers console_direct+ \
    && chmod +x /usr/local/bin/genesis_rl_entrypoint.sh

WORKDIR ${DEPLOY_WS}
ENTRYPOINT ["/usr/local/bin/genesis_rl_entrypoint.sh"]
CMD ["bash"]
