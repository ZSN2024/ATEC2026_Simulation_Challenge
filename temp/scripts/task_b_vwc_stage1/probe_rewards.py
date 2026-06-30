"""Print Stage1 reward terms under zero policy action for quick sanity checks."""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.append(SCRIPTS_ROOT)


parser = argparse.ArgumentParser(description="Probe Task B Stage1 reward terms.")
parser.add_argument("--task", type=str, default="ATEC-TaskB-B2wPiper-VWC-Stage1-v0", help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=8, help="Number of environments to simulate.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from isaaclab_tasks.utils import parse_env_cfg

import atec_rl_lab.train.task_b_vwc_stage1  # noqa: F401  # isort: skip


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()

    unwrapped = env.unwrapped
    zero_action = torch.zeros((args_cli.num_envs, *env.action_space.shape), device=unwrapped.device)
    random_action = 2.0 * torch.rand_like(zero_action) - 1.0
    probes = [("zero", zero_action), ("random", random_action)]

    reward_manager = unwrapped.reward_manager
    for label, action in probes:
        env.step(action)
        print(f"[reward_probe_begin] action={label}")
        for name, term_cfg in reward_manager._term_cfgs.items():
            values = reward_manager._term_names_to_values[name]
            mean_value = values.mean().item()
            min_value = values.min().item()
            max_value = values.max().item()
            weight = float(term_cfg.weight)
            print(
                f"[reward_term] name={name} mean={mean_value:.6f} min={min_value:.6f} max={max_value:.6f} "
                f"weight={weight:.6f} weighted_mean={mean_value * weight:.6f}"
            )
        print(f"[reward_probe_end] action={label}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
