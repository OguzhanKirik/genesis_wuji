#!/usr/bin/env python3
"""Labelled head-camera frames of the reach-to-grasp scenes, for training the rod segmenter.

Runs in the training container (Isaac Sim 6.0 / Isaac Lab 3.0, the same image that trains). Builds the
reach-to-grasp env at the level in ADAM_U_RG_LEVEL (52: the full bin with 5-60 rod heaps; 44: one rod in the
bin; 27: the pile on the table), adds a head camera to every env at the pose the robot's head camera has
(neckPitch link + genesis_isaac's offset, 87 deg HFOV, 640x480, looking at the rods) with a little jitter per
frame, and after every reset writes

    <out>/rgb/<frame>.png            colour
    <out>/depth/<frame>.npy          float16 metres (distance to the image plane, as a RealSense reports it)
    <out>/inst/<frame>.png           uint16 instance ids, 0 = background, k = the k-th rod of that frame
    <out>/meta/<frame>.json          camera K and world pose, rod poses (world, xyzw), level, env

deploy/genesis_perception/scripts/make_yolo_labels.py turns `inst/` into YOLO-seg polygons on the host.

    ADAM_U_RG_LEVEL=52 python tools/render_rod_dataset.py --num_envs 8 --frames 400 --out datasets/rods_L52 \
        --headless --enable_cameras

The hand starts each episode at its standoff above the chosen rod, so part of the heap is behind it in some
frames: that is what the camera will see during the approach, keep it. Images are written with PIL (in the
Isaac image); no cv2 is needed here.
"""
import argparse
import json
import math
import os
import re
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--frames", type=int, default=200, help="total frames to write (over all envs)")
parser.add_argument("--out", default="datasets/rods")
parser.add_argument("--width", type=int, default=640)
parser.add_argument("--height", type=int, default=480)
parser.add_argument("--hfov_deg", type=float, default=87.0, help="RealSense D435/D455 depth FOV, as genesis_isaac")
parser.add_argument("--settle_steps", type=int, default=3, help="env steps after a reset before the frame")
parser.add_argument("--eye_jitter_m", type=float, default=0.02)
parser.add_argument("--look_jitter_m", type=float, default=0.03)
parser.add_argument("--seed", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.sensors import CameraCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402

from genesis_rl.envs import wuji_reach_grasp_env_cfg as G  # noqa: E402

ROD_PRIM = re.compile(r"/(Bolt|Neighbor\d+|Bed\d+)$")
# genesis_isaac/scripts/isaac_runner.py: the head camera's offset on neckPitch (unmeasured, the URDF's value)
HEAD_CAMERA_TRANSLATE_M = np.array([0.10, 0.00, 0.08])
APERTURE_MM = 20.955


def _t(x):
    return x.torch if hasattr(x, "torch") else x


focal_mm = APERTURE_MM / (2 * math.tan(math.radians(args.hfov_deg) / 2))
base = G.WujiReachGraspEnvCfg()


@configclass
class RenderSceneCfg(G.WujiReachGraspSceneCfg):
    head_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/HeadCam",
        update_period=0.0,
        height=args.height,
        width=args.width,
        data_types=["rgb", "distance_to_image_plane", "instance_segmentation_fast"],
        colorize_instance_segmentation=False,
        spawn=sim_utils.PinholeCameraCfg(focal_length=focal_mm, horizontal_aperture=APERTURE_MM,
                                         clipping_range=(0.05, 4.0)),
    )


@configclass
class RenderEnvCfg(G.WujiReachGraspEnvCfg):
    scene: RenderSceneCfg = RenderSceneCfg(num_envs=args.num_envs, env_spacing=base.scene.env_spacing)


cfg = RenderEnvCfg()
cfg.scene.num_envs = args.num_envs
cfg.sim.device = "cuda:0"
gym.register(id="RenderReachGrasp", entry_point="genesis_rl.envs.wuji_reach_grasp_env_cfg:ReachGraspEnv",
             disable_env_checker=True, kwargs={"env_cfg_entry_point": RenderEnvCfg})
env = gym.make("RenderReachGrasp", cfg=cfg)
u = env.unwrapped
wenv = RslRlVecEnvWrapper(env)
cam = u.scene["head_cam"]
robot = u.scene["robot"]
neck_id = robot.find_bodies("neckPitch")[0][0]
rng = np.random.default_rng(args.seed)
level = int(os.environ.get("ADAM_U_RG_LEVEL", "0"))

for sub in ("rgb", "depth", "inst", "meta"):
    os.makedirs(os.path.join(args.out, sub), exist_ok=True)


def rod_prims():
    """Every rigid rod in the scene: (name, asset)."""
    out = []
    for name, asset in u.scene.rigid_objects.items():
        if ROD_PRIM.search("/" + name) or name in ("object",):
            out.append((name, asset))
    for name, coll in getattr(u.scene, "rigid_object_collections", {}).items():
        out.append((name, coll))
    return out


def rod_poses_world(env_i):
    """(M, 7) world poses (xyz, xyzw) of the rods of env `env_i`, from the scene's rigid objects."""
    poses = []
    for name, asset in rod_prims():
        if hasattr(asset.data, "object_state_w"):     # a collection: (N, K, 13)
            st = _t(asset.data.object_state_w)[env_i, :, :7].cpu().numpy()
            poses.extend(st.tolist())
        else:
            st = _t(asset.data.root_state_w)[env_i, :7].cpu().numpy()
            poses.append(st.tolist())
    P = np.asarray(poses, float).reshape(-1, 7)
    # Isaac Lab stores quaternions wxyz; publish xyzw like the perception contract
    return np.concatenate([P[:, :3], P[:, 4:7], P[:, 3:4]], axis=1)


def place_cameras():
    """Eye at the head camera's position on neckPitch (plus jitter), looking at the rods' centroid."""
    pos = _t(robot.data.body_pos_w)[:, neck_id].cpu().numpy()
    quat = _t(robot.data.body_quat_w)[:, neck_id].cpu().numpy()   # wxyz
    eyes, targets = [], []
    for i in range(args.num_envs):
        w, x, y, z = quat[i]
        R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                      [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                      [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
        eye = pos[i] + R @ HEAD_CAMERA_TRANSLATE_M + rng.uniform(-args.eye_jitter_m, args.eye_jitter_m, 3)
        rods = rod_poses_world(i)[:, :3]
        target = rods.mean(axis=0) + rng.uniform(-args.look_jitter_m, args.look_jitter_m, 3)
        eyes.append(eye)
        targets.append(target)
    dev = u.device
    cam.set_world_poses_from_view(torch.tensor(np.asarray(eyes), dtype=torch.float32, device=dev),
                                  torch.tensor(np.asarray(targets), dtype=torch.float32, device=dev))


def instance_to_rod_ids(seg, info):
    """Renderer instance ids -> 0 (background) or 1..k per rod prim, from the idToLabels table of this env."""
    table = info.get("idToLabels", {}) if isinstance(info, dict) else {}
    remap = np.zeros(int(seg.max()) + 1 if seg.size else 1, dtype=np.uint16)
    k = 0
    for sid, path in table.items():
        try:
            sid = int(sid)
        except ValueError:
            continue
        if sid < len(remap) and ROD_PRIM.search(str(path)):
            k += 1
            remap[sid] = k
    return remap[seg]


wenv.reset()
written = 0
zero = torch.zeros(args.num_envs, u.action_manager.total_action_dim, device=u.device)
while written < args.frames:
    place_cameras()
    for _ in range(args.settle_steps):
        wenv.step(zero)
    cam.update(dt=u.step_dt)
    rgb = _t(cam.data.output["rgb"]).cpu().numpy()
    depth = _t(cam.data.output["distance_to_image_plane"]).cpu().numpy()
    seg = _t(cam.data.output["instance_segmentation_fast"]).cpu().numpy()
    infos = cam.data.info
    K = _t(cam.data.intrinsic_matrices).cpu().numpy()
    cam_pos = _t(cam.data.pos_w).cpu().numpy()
    cam_quat = _t(cam.data.quat_w_ros).cpu().numpy()   # wxyz, ROS optical convention
    for i in range(args.num_envs):
        if written >= args.frames:
            break
        name = f"{written:06d}"
        Image.fromarray(np.ascontiguousarray(rgb[i, :, :, :3]).astype(np.uint8)).save(os.path.join(args.out, "rgb", name + ".png"))
        d = depth[i, :, :, 0] if depth.ndim == 4 else depth[i]
        np.save(os.path.join(args.out, "depth", name + ".npy"), d.astype(np.float16))
        s = seg[i, :, :, 0] if seg.ndim == 4 else seg[i]
        info_i = infos[i]["instance_segmentation_fast"] if isinstance(infos, (list, tuple)) else infos["instance_segmentation_fast"]
        inst = instance_to_rod_ids(s.astype(np.int64), info_i)
        Image.fromarray(inst.astype(np.uint16)).save(os.path.join(args.out, "inst", name + ".png"))
        qw = cam_quat[i]
        with open(os.path.join(args.out, "meta", name + ".json"), "w", encoding="utf-8") as fh:
            json.dump(dict(level=level, env=i, K=K[i].tolist(), cam_pos_w=cam_pos[i].tolist(),
                           cam_quat_w_xyzw=[float(qw[1]), float(qw[2]), float(qw[3]), float(qw[0])],
                           rods_w_xyz_xyzw=rod_poses_world(i).tolist(), n_rods=int(inst.max())), fh)
        written += 1
    wenv.reset()
    if written % 50 == 0:
        print(f"{written}/{args.frames} frames", flush=True)

print(f"wrote {written} frames to {args.out}")
env.close()
app.close()
