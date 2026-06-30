"""Open-loop wheel response probe for Stage1 B2W."""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.append(SCRIPTS_ROOT)

parser = argparse.ArgumentParser(description="Probe Task B Stage1 wheel action response.")
parser.add_argument("--task", type=str, default="ATEC-TaskB-B2wPiper-VWC-Stage1-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=20)
parser.add_argument("--wheel_action", type=float, default=0.5)
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
    env.reset()
    unwrapped = env.unwrapped
    action = torch.zeros((args_cli.num_envs, *env.action_space.shape), device=unwrapped.device)
    action[:, -4:] = args_cli.wheel_action

    robot = unwrapped.scene["robot"]
    wheel_ids, _ = robot.find_joints(list(unwrapped._wheel_joint_names))
    contact_sensor = unwrapped.scene.sensors["contact_sensor"]
    wheel_body_ids = contact_sensor.find_bodies(".*_foot")[0]
    for step in range(args_cli.steps):
        env.step(action)
        wheel_vel = robot.data.joint_vel[:, wheel_ids][0].detach().cpu().tolist()
        base_lin = robot.data.root_lin_vel_b[0].detach().cpu().tolist()
        base_ang = robot.data.root_ang_vel_b[0].detach().cpu().tolist()
        base_height = robot.data.root_pos_w[0, 2].item()
        wheel_contact_force = (
            contact_sensor.data.net_forces_w_history[:, :, wheel_body_ids, :]
            .norm(dim=-1)
            .max(dim=1)[0][0]
            .detach()
            .cpu()
            .tolist()
        )
        print(
            f"step={step} wheel_vel={wheel_vel} base_lin={base_lin} "
            f"base_ang={base_ang} wheel_contact_force={wheel_contact_force} base_height={base_height:.4f}"
        )
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
