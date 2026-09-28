# Adam-U RL — Isaac Sim 6.0 / Isaac Lab 3.0. Trains AND renders.
#
# One environment for both jobs. Isaac Sim 5.1 cannot drive the RTX renderer on
# an RTX PRO 6000 Blackwell (sm_120) -- it segfaults at 2.64 s in
# librtx.scenedb.plugin.so -- so 6.0 is the only version here that can both
# train and show you the result. Verified: `--viz kit` reaches "app ready" in
# ~9 s and trains with the viewport open.
#
# The repo briefly carried a second Isaac Sim 5.1 / Isaac Lab 2.3.2 image for
# training. It is gone (see git history), but two things it left behind matter:
#
#   * The robot spawns from a PRE-CONVERTED FLAT USD, committed under
#     assets/robots/adam_u/usd/. Isaac Lab 3.0's URDF importer nests rigid
#     bodies up to 17 levels deep, so the env's explicit FrameTransformer prim
#     paths ({ENV_REGEX_NS}/Robot/waistRoll and friends) stop resolving and
#     every distance reward silently loses its frames. 2.3.2's converter emits
#     a flat hierarchy; that output is what ships. tools/export_usd.py can
#     regenerate it, but needs the 2.3.2 image restored from git history.
#   * Quaternions changed order in 3.0: InitialStateCfg.rot is (x,y,z,w), not
#     (w,x,y,z). Upstream's (0.707,0,0,0.707) reads as 90 deg about X here
#     instead of Z and spawns the robot on its side. See the env config.
#
# Isaac Lab 3.0 is beta and its release notes warn of continuing breaking
# changes; expect this image to need repair on update.

FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    OMNI_KIT_ACCEPT_EULA=YES \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# --- system libraries -------------------------------------------------------
# Same X/GL/Vulkan stack as the training image, but here it is load-bearing:
# this one actually renders.
# build-essential is needed here but not in the 5.1 image: a few of the
# rerun/viser dependencies ship no cp312 wheel (pyliblzfse, an optional LZFSE
# codec reached through trimesh, is the first to fail) and must compile.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates curl git git-lfs \
        libgl1 libglu1-mesa libglib2.0-0 libsm6 libice6 libxt6 \
        libxrandr2 libxinerama1 libxcursor1 libxi6 libxkbcommon0 \
        libxrender1 libxext6 libfontconfig1 libfreetype6 \
        libvulkan1 vulkan-tools mesa-vulkan-drivers libegl1 \
        libgomp1 xvfb \
    && rm -rf /var/lib/apt/lists/*

# --- python 3.12 ------------------------------------------------------------
ENV UV_INSTALL_DIR=/usr/local/bin
RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
    && uv python install 3.12 \
    && uv venv --python 3.12 /opt/venv
ENV VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH

# --- python dependencies ----------------------------------------------------
# Frozen from a verified working Isaac Sim 6.0 environment, plus rerun-sdk and
# viser (resolved against the frozen numpy/torch/scipy pins). Those two matter:
# both bypass RTX entirely, so they are the fallback if `--viz kit` ever
# regresses on this card the way 5.1 did. viser is browser-based, so it also
# works from another machine.
COPY docker/requirements.lock.txt /tmp/requirements.lock.txt
RUN uv pip install --no-deps "setuptools<81" wheel \
    && uv pip install --no-deps --no-build-isolation \
        --extra-index-url https://pypi.nvidia.com \
        --extra-index-url https://download.pytorch.org/whl/cu128 \
        --index-strategy unsafe-best-match \
        -r /tmp/requirements.lock.txt \
    && rm /tmp/requirements.lock.txt

# --- Isaac Lab 3.0 ----------------------------------------------------------
# Pinned to a commit, not a tag: 3.0 is beta and moves. This is the revision
# verified on this GPU. Ten source packages, up from five in 2.3.2.
ARG ISAACLAB30_REV=28a37cecdd433c22d9eabd6a5954add9f13a8951
RUN git clone --filter=blob:none https://github.com/isaac-sim/IsaacLab.git /opt/IsaacLab30 \
    && git -C /opt/IsaacLab30 checkout --detach ${ISAACLAB30_REV} \
    && for p in isaaclab isaaclab_assets isaaclab_newton isaaclab_ov isaaclab_ovphysx \
                isaaclab_physx isaaclab_ppisp isaaclab_rl isaaclab_tasks \
                isaaclab_visualizers; do \
         uv pip install --no-deps -e /opt/IsaacLab30/source/$p ; \
       done
ENV ISAACLAB_PATH=/opt/IsaacLab30

# --- project code -----------------------------------------------------------
WORKDIR /workspace
COPY tools/ /opt/tools/
COPY genesis_rl /workspace/genesis_rl
COPY assets /workspace/assets
COPY pyproject.toml /workspace/pyproject.toml

# Editable install so `import genesis_rl` does not depend on the working
# directory. The compose file bind-mounts the checkout over /workspace, and
# the egg-link still points there, so host edits stay live.
RUN uv pip install --no-deps -e /workspace

ENV PYTHONPATH=/workspace:/workspace/genesis_rl:/workspace/genesis_rl/scripts

COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["bash"]
