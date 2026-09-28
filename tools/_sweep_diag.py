import sys
CKPT = sys.argv[1]
sys.argv = [sys.argv[0]]
from isaaclab.app import AppLauncher
app = AppLauncher(headless=True).app
import torch, importlib.metadata as md
import gymnasium as gym
from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from genesis_rl.algos.rsl_rl.rsl_rl_ppo_cfg import RslRlWujiPipelinePPORunnerCfg
from genesis_rl.envs import wuji_pipeline_env_cfg as P
from genesis_rl.envs import wuji_synergy_grasp_env_cfg as C
from genesis_rl.envs.adam_u_wuji_grasp_env_cfg import _t

cfg = P.WujiPipelineEnvCfg(); cfg.scene.num_envs = 128; cfg.sim.device = "cuda:0"
gym.register(id="SweepDiag", entry_point="isaaclab.envs:ManagerBasedRLEnv",
             disable_env_checker=True, kwargs={"env_cfg_entry_point": P.WujiPipelineEnvCfg})
env = gym.make("SweepDiag", cfg=cfg); u = env.unwrapped
ac = RslRlWujiPipelinePPORunnerCfg(); ac.device = "cuda:0"
ac = handle_deprecated_rsl_rl_cfg(ac, md.version("rsl-rl-lib"))
w = RslRlVecEnvWrapper(env, clip_actions=ac.clip_actions)
r = OnPolicyRunner(w, ac.to_dict(), log_dir=None, device="cuda:0")
r.load(CKPT); policy = r.get_inference_policy(device="cuda:0")
robot = u.scene["robot"]; term = u.action_manager._terms["pipeline"]
wb, _ = robot.find_bodies(P.WRIST_BODY); wb = wb[0]

def wpos(): return _t(robot.data.body_link_pos_w)[:, wb, :] - u.scene.env_origins

obs = w.get_observations()
start = None; path = None; first_stable = torch.full((128,), -1, device=u.device)
after_stable = torch.zeros(128, device=u.device)
pos_at_stable = None
print("\n  %5s %10s %10s %10s %9s %9s" % ("step", "|dxy| cum", "path len", "dxy/step", "stable%", "closure"))
with torch.inference_mode():
    for t in range(P.GRASP_STEPS):
        a = policy(obs); obs, _, _, _ = w.step(a)
        p = wpos()
        if start is None:
            start = p.clone(); path = torch.zeros(128, device=u.device); prev = p.clone()
        else:
            path += torch.norm(p[:, :2] - prev[:, :2], dim=1); prev = p.clone()
        b = C._buffers(u)
        stable = b["thumb_stable"] & b["finger_stable"]
        newly = stable & (first_stable < 0)
        if bool(newly.any()):
            first_stable[newly] = t
            if pos_at_stable is None:
                pos_at_stable = p.clone()
            else:
                pos_at_stable[newly] = p[newly]
        if pos_at_stable is not None:
            got = first_stable >= 0
            after_stable = torch.where(got, torch.norm(p[:, :2] - pos_at_stable[:, :2], dim=1), after_stable)
        if t % 10 == 9 or t == P.GRASP_STEPS - 1:
            print("  %5d %8.1fmm %8.1fmm %8.2fmm %8.1f%% %9.2f" % (
                t + 1, torch.norm(p[:, :2] - start[:, :2], dim=1).mean() * 1000,
                path.mean() * 1000, (path / max(1, t)).mean() * 1000,
                float((first_stable >= 0).float().mean()) * 100, term.closure[:, 1].mean()))
got = first_stable >= 0
print("\n  episodes reaching stable opposing contact: %.1f%%" % (float(got.float().mean()) * 100))
print("  mean step of first stable contact:         %.1f of %d" % (first_stable[got].float().mean(), P.GRASP_STEPS))
print("  wrist path length over GRASP (horizontal): %.1f mm" % (path.mean() * 1000))
print("  net wrist displacement over GRASP:         %.1f mm" % (torch.norm(wpos()[:, :2] - start[:, :2], dim=1).mean() * 1000))
print("  wrist travel AFTER first stable contact:   %.1f mm" % (after_stable[got].mean() * 1000))
print("\n  => path length >> net displacement means wandering, not a direct correction.")
env.close(); app.close()
