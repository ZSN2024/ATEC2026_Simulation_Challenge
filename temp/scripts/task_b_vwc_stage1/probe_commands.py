"""Probe Stage1 command generators for continuity and shape."""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.append(SCRIPTS_ROOT)

parser = argparse.ArgumentParser(description="Probe Task B Stage1 commands.")
parser.add_argument("--task", type=str, default="ATEC-TaskB-B2wPiper-VWC-Stage1-v0")
parser.add_argument("--num_envs", type=int, default=2)
parser.add_argument("--steps", type=int, default=10)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from isaaclab_tasks.utils import parse_env_cfg

import atec_rl_lab.train.task_b_vwc_stage1  # noqa: F401


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg)
    obs, _ = env.reset()
    unwrapped = env.unwrapped
    zero_action = torch.zeros((args_cli.num_envs, *env.action_space.shape), device=unwrapped.device)
    for step in range(args_cli.steps):
        base_cmd = unwrapped.command_manager.get_command("base_velocity")[0].detach().cpu().tolist()
        ee_cmd = unwrapped.command_manager.get_command("ee_goal")[0].detach().cpu().tolist()
        ee_rpy = unwrapped.command_manager.get_term("ee_goal").command_rpy[0].detach().cpu().tolist()
        ee_term = unwrapped.command_manager.get_term("ee_goal")
        use_ground = getattr(ee_term, "use_ground_goal", torch.zeros(args_cli.num_envs, device=unwrapped.device, dtype=torch.bool))
        ground_z_w = getattr(ee_term, "ground_goal_w", torch.zeros(args_cli.num_envs, 3, device=unwrapped.device))[:, 2]
        ee_goal_z_b = unwrapped.command_manager.get_command("ee_goal")[:, 2]
        phase = (ee_term.ee_goal_timer[0] / ee_term.ee_goal_total_time[0].clamp_min(1.0e-6)).item()
        traj_phase = (ee_term.ee_goal_timer[0] / ee_term.ee_goal_traj_time[0].clamp_min(1.0e-6)).clamp(0.0, 1.0).item()
        hold_phase = (
            (ee_term.ee_goal_timer[0] - ee_term.ee_goal_traj_time[0]).clamp_min(0.0)
            / (ee_term.ee_goal_total_time[0] - ee_term.ee_goal_traj_time[0]).clamp_min(1.0e-6)
        ).clamp(0.0, 1.0).item()
        base_term = unwrapped.command_manager.get_term("base_velocity")
        curriculum_phase = float(base_term.metrics.get("curriculum_phase", torch.zeros(1, device=unwrapped.device))[0].item())
        print(
            f"step={step} base={base_cmd} ee_pos={ee_cmd} ee_rpy={ee_rpy} "
            f"ground_ratio={use_ground.float().mean().item():.3f} "
            f"goal_z_b_min={ee_goal_z_b.min().item():.3f} goal_z_b_max={ee_goal_z_b.max().item():.3f} "
            f"ground_z_w_min={ground_z_w[use_ground].min().item() if torch.any(use_ground) else 0.0:.3f} "
            f"ground_z_w_max={ground_z_w[use_ground].max().item() if torch.any(use_ground) else 0.0:.3f} "
            f"traj_phase={traj_phase:.3f} hold_phase={hold_phase:.3f} total_phase={phase:.3f} "
            f"curriculum_phase={curriculum_phase:.1f}",
            flush=True,
        )
        env.step(zero_action)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
