# adam_u_rl

## Overview

This repository provides a minimal example of loading the **adam_u robot** into [Isaac Lab](https://isaac-sim.github.io/IsaacLab/) and running simple RL training tasks.  
It is intended as a **starting point** for robot manipulation with Isaac Lab — both the RL algorithm and the environment design can (and should) be further extended.

<p align="center">
  <img src="docs/demo.gif" alt="Demo of adam_u grasping task" width="800"/>
</p>

## Features

- ✅ Load **adam_u robot** from URDF into Isaac Lab.  
- ✅ Simple RL environment for grasping using **RSL_RL**.  
- ✅ CLI arguments to select different environments.  

⚠️ **Note**: This repository is primarily a demonstration. The algorithms and environment design are simplified for clarity and should be re-designed for production research.

---

## Installation

Tested on:
- **Ubuntu 22.04**
- **Isaac Sim 5.0** (should also work with 4.0 / 4.5)

### Requirements

- [Isaac Lab (pip installation)](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html)  
  Please follow the official documentation for installation and troubleshooting.

Clone this repo:
```bash
git clone https://gitlab.com/pndbotics/manipulation.git
cd adam_u_rl
```

---

## Quick Start

### Train

Navigate to the `manipulation` folder and run:

```bash
python adam_u_rl/scripts/train.py --headless
```

### Evaluate

To evaluate a trained policy:

```bash
python adam_u_rl/scripts/play.py   --checkpoint_path logs/rsl_rl/adam_u_grasp/2025-06-27_17-04-25/model_1499.pt
```

---

# Tutorial: Adam-U Grasping Environment

This tutorial shows how to set up and run a simple **Adam-U grasping task** in Isaac Lab.

---

### 1. Scene Setup

The scene includes:
- A **table** (cuboid top + cylindrical leg).  
- A **target object** (5cm cube) placed on the table.  
- The **Adam-U robot** loaded from URDF with predefined initial joint positions.  
- **Ground plane** and **dome light** for physics and visualization.  

Example (table + cube object):
```python
table_top = sim_utils.CuboidCfg(size=(0.6, 0.5, 0.05))
object = sim_utils.CuboidCfg(size=(0.05, 0.05, 0.05))
robot = sim_utils.UrdfFileCfg(asset_path="assets/robots/adam_u/urdf/adam_u.urdf")
```

---

### 2. Action Space

The robot is controlled through **7 right-arm joints**:
- shoulderPitch_Right, shoulderRoll_Right, shoulderYaw_Right, elbow_Right, wristYaw_Right, wristPitch_Right, wristRoll_Right  

Example:
```python
actions = mdp.JointPositionActionCfg(
    asset_name="robot",
    joint_names=[...7 right arm joints...],
    scale=1.0
)
```

---

### 3. Observation Space

The policy receives observations including:
- Right arm joint positions & velocities  
- Right hand position & orientation  
- Finger joint positions & velocities  
- Object 3D position  
- Last actions & robot base position  

Example:
```python
right_arm_pos = ObsTerm(func=mdp.joint_pos, params={...})
object_position = ObsTerm(func=mdp.root_pos_w, params={...})
```

---

### 4. Reward Function

Rewards encourage:
- Staying alive  
- Approaching and lifting the object  
- Smooth actions and low joint velocity  

Example:
```python
distance_to_object = RewTerm(func=compute_distance_reward, weight=2.0)
object_height = RewTerm(func=compute_height_reward, weight=10.0)
```

---

