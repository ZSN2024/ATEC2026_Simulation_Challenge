"""Probe B2W+Piper arm radial reach without the Stage1 arm planner wrapper."""

import argparse
import csv
import math
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.append(SCRIPTS_ROOT)

parser = argparse.ArgumentParser(description="Probe raw B2W+Piper arm reach in base coordinates.")
parser.add_argument("--task", type=str, default="ATEC-TaskB-B2wPiper-VWC-Stage1-v0")
parser.add_argument("--out_dir", type=str, default="logs/task_b_vwc_stage1_arm_reachability")
parser.add_argument("--steps_per_target", type=int, default=240)
parser.add_argument("--strict_threshold", type=float, default=0.05)
parser.add_argument("--loose_threshold", type=float, default=0.12)
parser.add_argument("--radii", type=float, nargs="+", default=[0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80])
parser.add_argument("--command_type", type=str, choices=["position", "pose"], default="position")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from isaaclab_tasks.utils import parse_env_cfg

import atec_rl_lab.train.task_b_vwc_stage1  # noqa: F401
from atec_rl_lab.train.task_b_vwc_stage1.task_space import quat_from_rpy, world_to_base
from atec_rl_lab.utils.cartesian_controller import CartesianController


DIRECTIONS = {
    "forward": (1.0, 0.0, 0.0),
    "forward_down_15": (math.cos(math.radians(15.0)), 0.0, -math.sin(math.radians(15.0))),
    "forward_down_30": (math.cos(math.radians(30.0)), 0.0, -math.sin(math.radians(30.0))),
    "forward_down_45": (math.cos(math.radians(45.0)), 0.0, -math.sin(math.radians(45.0))),
    "down": (0.0, 0.0, -1.0),
    "left": (0.0, 1.0, 0.0),
    "right": (0.0, -1.0, 0.0),
}


def _ee_pos_b(robot, ee_body_id):
    return world_to_base(
        robot.data.root_pos_w,
        robot.data.root_quat_w,
        robot.data.body_pos_w[:, ee_body_id],
    )


def _step_ik(env, controller, robot, arm_ids, target_b, target_quat_b, steps_per_target: int):
    unwrapped = env.unwrapped
    for _ in range(steps_per_target):
        if args_cli.command_type == "pose":
            arm_target = controller.compute_base(target_b, ee_quat_b=target_quat_b)
        else:
            arm_target = controller.compute_base(target_b)
        robot.set_joint_position_target(arm_target, joint_ids=arm_ids)
        unwrapped.scene.write_data_to_sim()
        unwrapped.sim.step(render=False)
        unwrapped.scene.update(unwrapped.physics_dt)


def main():
    out_dir = Path(args_cli.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    env = gym.make(args_cli.task, cfg=env_cfg)
    obs, _ = env.reset()
    del obs
    unwrapped = env.unwrapped
    robot = unwrapped.scene["robot"]
    arm_names = list(unwrapped.cfg.scene.robot.arm_joint_names)
    arm_ids, _ = robot.find_joints(arm_names)
    ee_body_id = robot.find_bodies("gripper_base")[0][0]
    controller = CartesianController(
        robot=robot,
        ee_body_name="gripper_base",
        arm_joint_names=arm_names,
        num_envs=1,
        device=unwrapped.device,
        command_type=args_cli.command_type,
        max_joint_delta=0.05,
    )

    start_ee_b = _ee_pos_b(robot, ee_body_id)[0].detach().clone()
    root_height = float(robot.data.root_pos_w[0, 2].item())
    target_quat_b = quat_from_rpy(torch.zeros((1, 3), dtype=torch.float32, device=unwrapped.device))
    rows = []

    for direction_name, direction_tuple in DIRECTIONS.items():
        direction = torch.tensor(direction_tuple, dtype=torch.float32, device=unwrapped.device)
        direction = direction / torch.linalg.norm(direction).clamp_min(1.0e-6)
        for radius in args_cli.radii:
            env.reset()
            controller.reset()
            target_b = (start_ee_b + direction * float(radius)).view(1, 3)
            _step_ik(env, controller, robot, arm_ids, target_b, target_quat_b, args_cli.steps_per_target)
            final_ee_b = _ee_pos_b(robot, ee_body_id)[0].detach()
            pos_err = float(torch.linalg.norm(target_b[0] - final_ee_b).item())
            achieved_delta = final_ee_b - start_ee_b
            achieved_along = float(torch.dot(achieved_delta, direction).item())
            row = {
                "command_type": args_cli.command_type,
                "direction": direction_name,
                "radius": radius,
                "target_x_b": float(target_b[0, 0].item()),
                "target_y_b": float(target_b[0, 1].item()),
                "target_z_b": float(target_b[0, 2].item()),
                "final_x_b": float(final_ee_b[0].item()),
                "final_y_b": float(final_ee_b[1].item()),
                "final_z_b": float(final_ee_b[2].item()),
                "pos_err": pos_err,
                "reachable_5cm": pos_err <= args_cli.strict_threshold,
                "reachable_12cm": pos_err <= args_cli.loose_threshold,
                "achieved_along_direction": achieved_along,
                "root_height_w": root_height,
                "start_ee_x_b": float(start_ee_b[0].item()),
                "start_ee_y_b": float(start_ee_b[1].item()),
                "start_ee_z_b": float(start_ee_b[2].item()),
            }
            rows.append(row)
            print(
                f"{args_cli.command_type} {direction_name:>15s} r={radius:.2f} "
                f"target=({row['target_x_b']:.3f},{row['target_y_b']:.3f},{row['target_z_b']:.3f}) "
                f"final=({row['final_x_b']:.3f},{row['final_y_b']:.3f},{row['final_z_b']:.3f}) "
                f"err={pos_err:.4f} ok5={int(row['reachable_5cm'])} ok12={int(row['reachable_12cm'])}",
                flush=True,
            )

    csv_path = out_dir / f"b2w_piper_radial_reach_{args_cli.command_type}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary_lines = [
        "# B2W+Piper Raw Arm Radial Reach",
        "",
        f"- command_type: {args_cli.command_type}",
        f"- start_ee_b: [{start_ee_b[0].item():.4f}, {start_ee_b[1].item():.4f}, {start_ee_b[2].item():.4f}]",
        f"- root_height_w: {root_height:.4f}",
        f"- strict threshold: {args_cli.strict_threshold:.3f} m",
        f"- loose threshold: {args_cli.loose_threshold:.3f} m",
        "",
        "| direction | max_radius_5cm | max_radius_12cm | max_achieved_along |",
        "|---|---:|---:|---:|",
    ]
    for direction_name in DIRECTIONS:
        bucket = [row for row in rows if row["direction"] == direction_name]
        ok5 = [row["radius"] for row in bucket if row["reachable_5cm"]]
        ok12 = [row["radius"] for row in bucket if row["reachable_12cm"]]
        max_along = max(row["achieved_along_direction"] for row in bucket)
        summary_lines.append(
            f"| {direction_name} | {max(ok5) if ok5 else 0.0:.2f} | {max(ok12) if ok12 else 0.0:.2f} | {max_along:.3f} |"
        )

    summary_path = out_dir / f"b2w_piper_radial_reach_{args_cli.command_type}.md"
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print(f"WROTE {csv_path}")
    print(f"WROTE {summary_path}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
