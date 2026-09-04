# Adam-U RL — Isaac Lab 2.3.2 / Isaac Sim 5.1.0 training image.
#
# Structure copied from adam-wbc: built from the pip distribution rather than
# NVIDIA's NGC base image, so no NGC account or API key is needed, and every
# version is pinned from a verified working environment
# (docker/requirements.lock.txt). The four pins that are easy to get wrong:
#
#   rsl-rl-lib==3.0.1             2.3.1 makes Isaac Lab's train script exit silently
#   antlr4-python3-runtime==4.9.3 newer breaks omegaconf's ATN deserializer
#   tokenizers==0.22.2            transformers 4.57.6 needs <=0.23.0
#   flatdict==4.0.1               needs setuptools<81 and --no-build-isolation
#
# The lock also carries the Pink IK stack (pin, pin-pink, qpsolvers, daqp and
# the cmeel-* binary wheels). adam_u_grasp_env_cfg.py imports
# `pink.tasks.FrameTask` at module scope, so `pink` has to be importable even
# though the env's action space is currently plain joint-position control.
#
# Verified on an RTX PRO 6000 Blackwell (sm_120). torch 2.7.0+cu128 ships sm_120.
# NOTE Isaac Sim 5.1's RTX renderer CRASHES on this GPU -- headless only here.
# See README.md; that is why play.py's --video default was flipped off.

FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    OMNI_KIT_ACCEPT_EULA=YES \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# --- system libraries -------------------------------------------------------
# Isaac Sim needs the X/GL/Vulkan stack even in headless mode.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl git git-lfs \
        libgl1 libglu1-mesa libglib2.0-0 libsm6 libice6 libxt6 \
        libxrandr2 libxinerama1 libxcursor1 libxi6 libxkbcommon0 \
        libxrender1 libxext6 libfontconfig1 libfreetype6 \
        libvulkan1 vulkan-tools mesa-vulkan-drivers libegl1 \
        libgomp1 xvfb \
    && rm -rf /var/lib/apt/lists/*

# --- python 3.11 (Ubuntu 24.04 ships only 3.12; Isaac Lab wants 3.11) -------
ENV UV_INSTALL_DIR=/usr/local/bin
RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
    && uv python install 3.11 \
    && uv venv --python 3.11 /opt/venv
ENV VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH

# --- python dependencies ----------------------------------------------------
# flatdict 4.0.1 builds via pkg_resources, removed in setuptools 81.
COPY docker/requirements.lock.txt /tmp/requirements.lock.txt
RUN uv pip install --no-deps "setuptools<81" wheel \
    && uv pip install --no-deps --no-build-isolation \
        --extra-index-url https://pypi.nvidia.com \
        --extra-index-url https://download.pytorch.org/whl/cu128 \
        --index-strategy unsafe-best-match \
        -r /tmp/requirements.lock.txt \
    && rm /tmp/requirements.lock.txt

# --- Isaac Lab --------------------------------------------------------------
ARG ISAACLAB_TAG=v2.3.2
RUN git clone --depth 1 --branch ${ISAACLAB_TAG} \
        https://github.com/isaac-sim/IsaacLab.git /opt/IsaacLab \
    && for p in isaaclab isaaclab_assets isaaclab_mimic isaaclab_rl isaaclab_tasks; do \
         uv pip install --no-deps -e /opt/IsaacLab/source/$p ; \
       done
ENV ISAACLAB_PATH=/opt/IsaacLab

# --- project code -----------------------------------------------------------
# COPY gives a self-contained image for cluster runs; compose bind-mounts over
# this for local development so edits are live.
WORKDIR /workspace
COPY tools/ /opt/tools/
COPY adam_u_isaac_lab/ /workspace/

# The Adam-U Collada meshes carry USD-illegal ids -- 280 of them, with dots in
# node names ("L_middle_distal.001") and CJK material names ("灰色-material").
# Isaac Sim's URDF importer turns those into prim names, silently produces a
# null prim, and yields an articulation with ZERO rigid bodies. Repaired at
# build time so the fix is reproducible rather than a manual step. Idempotent;
# the host tree is already sanitized, and the entrypoint re-checks the mount.
RUN python /opt/tools/sanitize_dae.py \
        /workspace/assets/robots/adam_u/meshes

# train.py adds its own directories to sys.path, but set these too so the
# package also imports cleanly from an interactive shell or a notebook.
ENV PYTHONPATH=/workspace:/workspace/adam_u_rl:/workspace/adam_u_rl/scripts

COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["bash"]
