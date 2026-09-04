#!/usr/bin/env bash
# Adam-U RL container entrypoint.
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
name = torch.cuda.get_device_name(0)
arches = torch.cuda.get_arch_list()
print(f"[entrypoint] {name} (sm_{cc[0]}{cc[1]}), torch {torch.__version__}")
if not any(f"sm_{cc[0]}{cc[1]}" == a for a in arches):
    print(f"[entrypoint] WARNING: torch lacks kernels for sm_{cc[0]}{cc[1]}; "
          f"built for {arches}")
PY

# Compose bind-mounts the host checkout over /workspace, shadowing the meshes
# repaired at build time. A fresh `git clone`, or a `git checkout -- .`, brings
# the USD-illegal ids back and the failure is deeply misleading:
#
#   ValueError: No contact sensors added to the prim ... no rigid bodies present
#
# so re-check the mount rather than trusting it. sanitize_dae.py is idempotent,
# and a clean tree costs one pass over 33 small files.
MESHES="/workspace/assets/robots/adam_u/meshes"
if [ -d "$MESHES" ]; then
    if python /opt/tools/sanitize_dae.py "$MESHES" --dry-run 2>/dev/null \
         | grep -q "would sanitize 0 ids"; then
        :  # already clean
    else
        echo "[entrypoint] repairing USD-illegal Collada ids in the mounted meshes"
        python /opt/tools/sanitize_dae.py "$MESHES" || \
            echo "[entrypoint] WARNING: mesh sanitization failed" >&2
    fi
fi

exec "$@"
