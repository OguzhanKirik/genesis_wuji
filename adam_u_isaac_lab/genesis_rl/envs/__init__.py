"""Compatibility package. New code should import ``genesis_rl.tasks``.

The env class is loaded on first use so importing a sibling module
(``genesis_rl.envs.wuji_synergy_grasp_env_cfg``) does not import every task.
"""


def __getattr__(name):
    if name == "AdamUGraspEnvCfg":
        from genesis_rl.tasks.pnd_grasp.env_cfg import AdamUGraspEnvCfg
        return AdamUGraspEnvCfg
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["AdamUGraspEnvCfg"]
