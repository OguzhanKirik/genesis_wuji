"""Which point of the rod its reported root pose is: a hull rod dropped standing (+z up) and one dropped upside down.

    docker compose exec -T adam_rl python /opt/tools/probe_rod_origin.py
Root height above the table at rest: ~17.5 mm both ways = the centre; ~0 / ~35 mm = the asset's origin (one end).
"""
import sys
from isaaclab.app import AppLauncher
app = AppLauncher(headless=True).app
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg
import genesis_rl.tasks.wuji_reach_grasp.env_cfg as E
from genesis_rl.tasks.wuji_pipeline import env_cfg as P
from genesis_rl.tasks.wuji_reach.env_cfg import LAYOUT, _t
sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1 / 120, device="cuda:0"))
cfg = InteractiveSceneCfg(num_envs=1, env_spacing=1.5)
cfg.ground = AssetBaseCfg(prim_path="/World/GroundPlane", spawn=GroundPlaneCfg())
cfg.table = P._static_box("Table", LAYOUT["table_center"], LAYOUT["table_size"], LAYOUT["table_color"])
for k, coll in enumerate(("hull", "hull", "cd", "cyl")):
    setattr(cfg, f"rod_{k}", E._neighbor_cfg(k, coll))
scene = InteractiveScene(cfg); sim.reset()
qs = [(0, 0, 0, 1), (1, 0, 0, 0), (0, 0, 0, 1), (0, 0, 0, 1)]          # xyzw: upright, upside down (180 about x)
for k in range(4):
    r = scene[f"rod_{k}"]
    p = torch.tensor([[E.BOLT_XY[0] + 0.06 * k, E.BOLT_XY[1], E.TABLE_TOP + 0.05]], device="cuda:0") + scene.env_origins
    r.write_root_pose_to_sim_index(root_pose=torch.cat([p, torch.tensor([qs[k]], device="cuda:0", dtype=torch.float)], dim=1),
                                   env_ids=torch.tensor([0], device="cuda:0"))
scene.write_data_to_sim()
for _ in range(120):
    sim.step(render=False); scene.update(1 / 120)
for k, name in enumerate(("hull +z up", "hull +z down", "cd +z up", "cyl +z up")):
    r = scene[f"rod_{k}"]
    print(f"{name:14s} root z above table {1000 * (float(_t(r.data.root_pos_w)[0, 2]) - E.TABLE_TOP):6.1f} mm,  com offset (body) "
          f"{[round(1000 * float(v), 1) for v in _t(r.data.body_com_pos_b)[0, 0]]} mm")
app.close()
