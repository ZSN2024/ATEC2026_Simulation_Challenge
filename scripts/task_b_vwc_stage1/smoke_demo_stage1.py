"""Smoke test Task B Stage1 demo adapter against the official Task B env."""

import argparse
import os
import sys


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Smoke test Task B Stage1 demo solution.")
parser.add_argument("--task", type=str, default="ATEC-TaskB-B2wPiper", help="Official Task B env id.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of envs.")
parser.add_argument("--steps", type=int, default=64, help="Number of rollout steps.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import atec_rl_lab.tasks  # noqa: F401, E402
from demo.taskb_stage1.solution_stage1 import AlgSolution  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402


def main():
    print("SMOKE_STAGE=start", flush=True)
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=True,
    )
    print("SMOKE_STAGE=env_cfg_ready", flush=True)
    env = gym.make(args_cli.task, cfg=env_cfg)
    print("SMOKE_STAGE=env_created", flush=True)
    solution = AlgSolution()
    print("SMOKE_STAGE=solution_ready", flush=True)

    obs, _ = env.reset()
    print("SMOKE_STAGE=env_reset", flush=True)
    total_reward = 0.0
    last_action_shape = None
    for step in range(args_cli.steps):
        with torch.inference_mode():
            resp = solution.predicts(obs, total_reward)
        print(f"SMOKE_STAGE=predict_{step}", flush=True)
        action = torch.tensor(resp["action"], dtype=torch.float32, device=args_cli.device)
        last_action_shape = tuple(action.shape)
        obs, reward, terminated, truncated, _ = env.step(action)
        print(f"SMOKE_STAGE=step_{step}", flush=True)
        if isinstance(reward, torch.Tensor):
            total_reward += reward.mean().item()
        else:
            total_reward += float(reward)
        if torch.isnan(action).any() or torch.isinf(action).any():
            raise RuntimeError("Demo action contains NaN or Inf.")
        if bool(terminated.any()) or bool(truncated.any()):
            break

    print(f"task={args_cli.task}")
    print(f"steps_completed={step + 1}")
    print(f"last_action_shape={last_action_shape}")
    print(f"total_reward={total_reward:.4f}")
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
