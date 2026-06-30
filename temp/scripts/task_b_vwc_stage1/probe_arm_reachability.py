"""Probe B2W+Piper arm reachable workspace in robot base coordinates."""

import argparse
import csv
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.append(SCRIPTS_ROOT)

parser = argparse.ArgumentParser(description="Probe Task B Stage1 arm reachability in base frame.")
parser.add_argument("--task", type=str, default="ATEC-TaskB-B2wPiper-VWC-Stage1-v0")
parser.add_argument("--out_dir", type=str, default="logs/task_b_vwc_stage1_arm_reachability")
parser.add_argument("--steps_per_target", type=int, default=160)
parser.add_argument("--pos_err_threshold", type=float, default=0.12)
parser.add_argument("--x_values", type=float, nargs="+", default=[0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75])
parser.add_argument("--y_values", type=float, nargs="+", default=[-0.35, -0.20, 0.0, 0.20, 0.35])
parser.add_argument(
    "--z_values",
    type=float,
    nargs="+",
    default=[-0.70, -0.60, -0.50, -0.40, -0.30, -0.20, -0.10, 0.0, 0.10, 0.20, 0.30, 0.40],
)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from isaaclab_tasks.utils import parse_env_cfg

import atec_rl_lab.train.task_b_vwc_stage1  # noqa: F401
from atec_rl_lab.train.task_b_vwc_stage1.cartesian_arm_action import CartesianArmAction
from atec_rl_lab.train.task_b_vwc_stage1.task_space import world_to_base


def _reset_robot_to_nominal(env):
    obs, _ = env.reset()
    return obs


def _measure_reachability(env, controller, robot, arm_ids, ee_body_id, target_b, steps_per_target: int):
    unwrapped = env.unwrapped
    device = unwrapped.device
    target = torch.tensor([target_b], dtype=torch.float32, device=device)
    controller.reset()

    for _ in range(steps_per_target):
        arm_target = controller.compute_base(target)
        robot.set_joint_position_target(arm_target, joint_ids=arm_ids)
        unwrapped.scene.write_data_to_sim()
        unwrapped.sim.step(render=False)
        unwrapped.scene.update(unwrapped.physics_dt)

    ee_pos_b = world_to_base(
        robot.data.root_pos_w,
        robot.data.root_quat_w,
        robot.data.body_pos_w[:, ee_body_id],
    )
    pos_err = torch.linalg.norm(target - ee_pos_b, dim=-1).item()
    final = ee_pos_b[0].detach().cpu().tolist()
    arm_pos = robot.data.joint_pos[:, arm_ids][0].detach().cpu().tolist()
    return pos_err, final, arm_pos


def _summarize(rows, pos_err_threshold: float):
    reachable = [r for r in rows if r["reachable"]]
    by_xy = {}
    for row in rows:
        key = (row["target_x_b"], row["target_y_b"])
        bucket = by_xy.setdefault(key, [])
        bucket.append(row)

    lines = [
        "# Task B Stage1 B2W+Piper Arm Reachability",
        "",
        f"- Samples: {len(rows)}",
        f"- Reachable threshold: position error <= {pos_err_threshold:.3f} m",
        f"- Reachable samples: {len(reachable)} / {len(rows)}",
        "",
        "## Highest Reachable/Lowest Reachable Z By XY",
        "",
        "| x_b | y_b | reachable_count | min_reachable_z_b | max_reachable_z_b | best_err |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for (x, y), bucket in sorted(by_xy.items()):
        ok = [r for r in bucket if r["reachable"]]
        best_err = min(r["pos_err"] for r in bucket)
        if ok:
            min_z = min(r["target_z_b"] for r in ok)
            max_z = max(r["target_z_b"] for r in ok)
            lines.append(f"| {x:.3f} | {y:.3f} | {len(ok)} | {min_z:.3f} | {max_z:.3f} | {best_err:.4f} |")
        else:
            lines.append(f"| {x:.3f} | {y:.3f} | 0 | n/a | n/a | {best_err:.4f} |")
    return "\n".join(lines) + "\n"


def main():
    out_dir = Path(args_cli.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    env = gym.make(args_cli.task, cfg=env_cfg)
    _reset_robot_to_nominal(env)
    unwrapped = env.unwrapped
    robot = unwrapped.scene["robot"]
    arm_names = list(unwrapped.cfg.scene.robot.arm_joint_names)
    arm_ids, _ = robot.find_joints(arm_names)
    ee_body_id = robot.find_bodies("gripper_base")[0][0]
    controller = CartesianArmAction(
        robot=robot,
        ee_body_name="gripper_base",
        arm_joint_names=arm_names,
        num_envs=1,
        device=unwrapped.device,
        command_type="position",
        max_joint_delta=0.05,
    )

    rows = []
    for x in args_cli.x_values:
        for y in args_cli.y_values:
            for z in args_cli.z_values:
                _reset_robot_to_nominal(env)
                pos_err, final, arm_pos = _measure_reachability(
                    env, controller, robot, arm_ids, ee_body_id, (x, y, z), args_cli.steps_per_target
                )
                reachable = pos_err <= args_cli.pos_err_threshold
                row = {
                    "target_x_b": x,
                    "target_y_b": y,
                    "target_z_b": z,
                    "pos_err": pos_err,
                    "reachable": reachable,
                    "final_x_b": final[0],
                    "final_y_b": final[1],
                    "final_z_b": final[2],
                }
                for idx, value in enumerate(arm_pos):
                    row[f"arm_joint_{idx}_pos"] = value
                rows.append(row)
                print(
                    f"target=({x:.3f},{y:.3f},{z:.3f}) err={pos_err:.4f} "
                    f"reachable={int(reachable)} final=({final[0]:.3f},{final[1]:.3f},{final[2]:.3f})",
                    flush=True,
                )

    csv_path = out_dir / "b2w_piper_base_reachability.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary_path = out_dir / "b2w_piper_base_reachability.md"
    summary_path.write_text(_summarize(rows, args_cli.pos_err_threshold), encoding="utf-8")
    print(f"WROTE {csv_path}")
    print(f"WROTE {summary_path}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
