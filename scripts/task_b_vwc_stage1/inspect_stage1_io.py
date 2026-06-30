"""Inspect Task B Stage1 env registration and basic IO."""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.append(SCRIPTS_ROOT)


parser = argparse.ArgumentParser(description="Inspect Task B Stage1 env IO.")
parser.add_argument("--task", type=str, default="ATEC-TaskB-B2wPiper-VWC-Stage1-v0", help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
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
    spec = gym.spec(args_cli.task)
    print(f"TASK={spec.id}")
    print(f"ENTRY={spec.entry_point}")
    print(f"KWARGS={sorted(spec.kwargs.keys())}")
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg)
    obs, _ = env.reset()
    if isinstance(obs, dict):
        for key, value in obs.items():
            if isinstance(value, torch.Tensor):
                print(f"OBS[{key}]={tuple(value.shape)}")
    print(f"ACTION_SPACE={env.action_space}")
    unwrapped = env.unwrapped
    robot_cfg = unwrapped.cfg.scene.robot
    print(f"ROBOT_JOINTS={list(getattr(robot_cfg, 'joint_names', []))}")
    print(f"LEG_JOINTS={list(getattr(robot_cfg, 'leg_joint_names', []))}")
    print(f"WHEEL_JOINTS={list(getattr(robot_cfg, 'wheel_joint_names', []))}")
    print(f"ARM_JOINTS={list(getattr(robot_cfg, 'arm_joint_names', []))}")
    print(f"POLICY_ACTION_DIM={getattr(unwrapped, '_policy_action_dim', None)}")
    print(f"ACTION_MANAGER_DIM={tuple(unwrapped.action_manager.action.shape)}")
    print(f"COMMANDS={list(unwrapped.command_manager.active_terms)}")
    for name in unwrapped.command_manager.active_terms:
        command = unwrapped.command_manager.get_command(name)
        if isinstance(command, torch.Tensor):
            print(f"COMMAND[{name}]={tuple(command.shape)}")
        term = unwrapped.command_manager.get_term(name)
        if hasattr(term, "metrics"):
            print(f"COMMAND_METRICS[{name}]={sorted(term.metrics.keys())}")
    print(f"REWARDS={list(unwrapped.reward_manager.active_terms)}")
    print(f"TERMINATIONS={list(unwrapped.termination_manager.active_terms)}")
    zero_action = torch.zeros(env.action_space.shape, device=unwrapped.device)
    obs, reward, terminated, truncated, _ = env.step(zero_action)
    print(f"STEP_REWARD={tuple(reward.shape) if isinstance(reward, torch.Tensor) else type(reward).__name__}")
    print(f"STEP_TERMINATED={tuple(terminated.shape)}")
    print(f"STEP_TRUNCATED={tuple(truncated.shape)}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
