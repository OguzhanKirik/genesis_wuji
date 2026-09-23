# Workcell assets

Copied verbatim from adam-moveit `ws/src/adam_u_isaac/assets/workcell/` at
commit 0acbd2c on 2026-09-19.

| file | what |
|---|---|
| `workcell.json` | table, bin and 40-60 bolt pile, as adam-moveit's Isaac runner authors them |
| `workcell_single.json` | same table, one bolt on the bare table, no bin |
| `rod-with-threads-0001/` | the bolt: 327k-face mesh, convex-hull collider, steel material with normal and roughness textures |

Coordinates are adam-moveit's world frame: robot root at the origin facing +x,
floor at z=0, table top at z=0.95. `adam_u_wuji_grasp_env_cfg.py` reads these
files and mirrors y for the right arm (see `WORKCELL_SIDE` there).

sha256:
    b20389aa257def0c510ece0838bf272b78a99865d924b9c7be4186f3b080059f  workcell.json
    cf80cb434cfa4ea4cd6cb140fda97cd5195c696fb6e0b60148ab903e119b70d3  workcell_single.json
    ebbc23bdc67c67e02ddecf3a913053fc26a6678fc264e8e2a15b0a927a488e22  rod-with-threads-0001/rod-with-threads-0001.usdc
