#!/usr/bin/env bash
# Adam-U RL visualization container entrypoint (Isaac Sim 6.0 / Isaac Lab 3.0).
set -euo pipefail

# Fail loudly and early rather than deep inside Isaac Sim.
if ! python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    echo "[entrypoint] ERROR: no CUDA device visible." >&2
    echo "             Run with '--gpus all' (docker) or the compose GPU reservation." >&2
    exit 1
fi

python - <<'PY'
import torch
cc = torch.cuda.get_device_capability(0)
print(f"[entrypoint] {torch.cuda.get_device_name(0)} (sm_{cc[0]}{cc[1]}), "
      f"torch {torch.__version__}")
arches = torch.cuda.get_arch_list()
if not any(f"sm_{cc[0]}{cc[1]}" == a for a in arches):
    print(f"[entrypoint] WARNING: torch lacks kernels for sm_{cc[0]}{cc[1]}; built for {arches}")
PY

# Isaac Lab 3.0's URDF importer nests rigid bodies under the kinematic tree,
# which breaks this env's explicit FrameTransformer prim paths. So this side
# needs the FLAT USD that Isaac Lab 2.3.2's converter emits. It is generated,
# not committed -- 109 MB is not worth carrying in git forever.
USD="/workspace/assets/robots/adam_u/usd/adam_u.usd"
if [ ! -f "$USD" ]; then
    cat >&2 <<'MSG'
[entrypoint] ERROR: the pre-converted USD is missing:
               assets/robots/adam_u/usd/adam_u.usd

             Isaac Lab 3.0 cannot import the URDF usably (it nests rigid
             bodies, breaking the FrameTransformer prim paths), so this image
             needs the flat USD that the 2.3.2 training image produces.

             Generate it once, from the repo root on the host:

               docker compose run --rm adam_rl python /opt/tools/export_usd.py

             Then bring this service up again.
MSG
    exit 1
fi

# X11 sanity: this image exists to render, so a missing display is fatal here
# rather than a warning, and the message says what to do about it.
if [ -n "${DISPLAY:-}" ] && [ ! -e "/tmp/.X11-unix/X${DISPLAY#*:}" ]; then
    echo "[entrypoint] WARNING: DISPLAY=$DISPLAY but /tmp/.X11-unix/X${DISPLAY#*:} is not mounted." >&2
    echo "             GUI (--viz kit) will fail. On the host: xhost +local:root" >&2
    echo "             RTX-free alternatives: --viz rerun  or  --viz viser" >&2
fi

exec "$@"
