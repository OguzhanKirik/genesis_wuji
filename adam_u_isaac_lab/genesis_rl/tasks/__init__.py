"""The tasks this package can train.

A task is one environment class, one PPO runner config, one gym id and one
experiment directory. ``scripts/train.py`` and ``scripts/play.py`` only know
this table. Adding a training run is a new folder under ``tasks/`` plus one
``_add(...)`` call here. The trainer does not grow a new branch.

This module is safe to import before Isaac Sim starts. ``Task.load_env`` and
``Task.load_agent`` are what pull in Isaac Lab, and the scripts call them
after ``AppLauncher``.

``wuji_teleop`` is an environment (``tasks/wuji_teleop``) and is not registered.
Experiment 3 was built and never trained, so it has no runner config of its own.
"""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass
from typing import Callable


def _load(spec: str):
    module_name, _, attr = spec.partition(":")
    if not attr:
        raise ValueError(f"expected 'module:Class', got {spec!r}")
    return getattr(importlib.import_module(module_name), attr)


@dataclass(frozen=True)
class Task:
    """One trainable task. ``env_cfg`` and ``agent_cfg`` are ``module:Class`` strings."""

    id: str
    short: str
    env_cfg: str
    agent_cfg: str
    entry_point: str
    experiment: str | Callable[[], str]

    def experiment_name(self) -> str:
        if callable(self.experiment):
            return self.experiment()
        return self.experiment

    def load_env(self):
        return _load(self.env_cfg)

    def load_agent(self):
        return _load(self.agent_cfg)


# Short name and gym id both resolve to the same Task. A list is the source of
# truth so iteration does not see each task twice.
TASK_LIST: list[Task] = []
BY_KEY: dict[str, Task] = {}


def _add(task: Task) -> None:
    TASK_LIST.append(task)
    BY_KEY[task.id] = task
    BY_KEY[task.short] = task


_PPO = "genesis_rl.algos.rsl_rl.rsl_rl_ppo_cfg"
_ENV = "isaaclab.envs:ManagerBasedRLEnv"

_add(Task(
    id="Isaac-AdamUGrasp30",
    short="pnd_grasp",
    env_cfg="genesis_rl.tasks.pnd_grasp.env_cfg:AdamUGraspEnvCfg",
    agent_cfg=f"{_PPO}:RslRlAdamUGraspPPORunnerCfg",
    entry_point=_ENV,
    experiment="adam_u30_grasp",
))
_add(Task(
    id="Isaac-AdamUWujiGrasp30",
    short="wuji_reach",
    env_cfg="genesis_rl.tasks.wuji_reach.env_cfg:AdamUWujiGraspEnvCfg",
    agent_cfg=f"{_PPO}:RslRlAdamUGraspPPORunnerCfg",
    entry_point=_ENV,
    experiment="adam_u30_wuji_grasp",
))
_add(Task(
    id="Isaac-AdamUWujiGraspOnly30",
    short="wuji_grasp",
    env_cfg="genesis_rl.tasks.wuji_grasp.env_cfg:AdamUWujiGraspOnlyEnvCfg",
    agent_cfg=f"{_PPO}:RslRlAdamUGraspPPORunnerCfg",
    entry_point=_ENV,
    experiment="adam_u30_wuji_grasponly",
))
_add(Task(
    id="Isaac-WujiSynergyGrasp30",
    short="wuji_synergy",
    env_cfg="genesis_rl.tasks.wuji_synergy.env_cfg:WujiSynergyGraspEnvCfg",
    agent_cfg=f"{_PPO}:RslRlWujiSynergyPPORunnerCfg",
    entry_point=_ENV,
    # Finger count changes the asset and the observation width.
    experiment=lambda: f"adam_u30_wuji_synergy_f{os.environ.get('ADAM_U_FINGERS', '3')}",
))
_add(Task(
    id="Isaac-WujiPipeline30",
    short="wuji_pipeline",
    env_cfg="genesis_rl.tasks.wuji_pipeline.env_cfg:WujiPipelineEnvCfg",
    agent_cfg=f"{_PPO}:RslRlWujiPipelinePPORunnerCfg",
    entry_point=_ENV,
    # The env reads ADAM_U_LEVEL and ADAM_U_STAGE. Log directories have always
    # been keyed on ADAM_U_LEVEL; keep that so old runs stay in the same folder.
    experiment=lambda: f"adam_u30_wuji_pipeline_L{os.environ.get('ADAM_U_LEVEL', '0')}",
))
_add(Task(
    id="Isaac-WujiReachGrasp30",
    short="wuji_reach_grasp",
    env_cfg="genesis_rl.tasks.wuji_reach_grasp.env_cfg:WujiReachGraspEnvCfg",
    agent_cfg=f"{_PPO}:RslRlWujiReachGraspPPORunnerCfg",
    entry_point="genesis_rl.tasks.wuji_reach_grasp.env_cfg:ReachGraspEnv",
    experiment=lambda: f"adam_u30_wuji_reach_grasp_L{os.environ.get('ADAM_U_RG_LEVEL', '0')}",
))

# (--robot, --scenario) for the flags that predate --task. A pnd robot ignores
# --scenario, which is what train.py used to do.
_BY_SCENARIO = {
    ("wuji", "reach"): "wuji_reach",
    ("wuji", "grasp"): "wuji_grasp",
    ("wuji", "synergy"): "wuji_synergy",
    ("wuji", "pipeline"): "wuji_pipeline",
    ("wuji", "reach_grasp"): "wuji_reach_grasp",
}


def known_names() -> str:
    shorts = ", ".join(task.short for task in TASK_LIST)
    ids = ", ".join(task.id for task in TASK_LIST)
    return f"short names: {shorts}. gym ids: {ids}"


def resolve(task: str | None, robot: str, scenario: str) -> Task:
    """Pick a task from ``--task`` or from the older ``--robot`` / ``--scenario`` pair.

    ``--task`` wins when both are passed.
    """
    if task:
        found = BY_KEY.get(task)
        if found is None:
            raise SystemExit(f"unknown task {task!r}. {known_names()}")
        return found
    if robot == "wuji":
        short = _BY_SCENARIO.get((robot, scenario))
        if short is None:
            raise SystemExit(
                f"unknown scenario {scenario!r} for robot {robot!r}. {known_names()}"
            )
        return BY_KEY[short]
    return BY_KEY["pnd_grasp"]
