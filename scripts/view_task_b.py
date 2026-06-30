# Created by skywoodsz on 2026/01/28.
import argparse
import itertools
import torch
from isaaclab.app import AppLauncher

# create argparser
parser = argparse.ArgumentParser(description="View ATEC Task B.")
parser.add_argument(
    "--num_envs", type=int, default=1, help="Number of environments to spawn."
)
parser.add_argument(
    "--robot",
    type=str,
    default="b2",
    choices=("b2", "b2w"),
    help="Task B robot configuration to view.",
)
parser.add_argument(
    "--report-after-steps",
    type=int,
    default=300,
    help="Report root height and joint state after this many zero-action steps.",
)
parser.add_argument(
    "--exit-after-report",
    action="store_true",
    default=False,
    help="Exit immediately after printing the report.",
)
parser.add_argument(
    "--report-file",
    type=str,
    default="",
    help="Optional file path to write the same report text.",
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""
from isaaclab.envs import ManagerBasedRLEnv

from atec_rl_lab.tasks.task_b.env_cfg import TaskBEnvB2Cfg, TaskBEnvB2WCfg


TASK_B_ENV_CFGS = {
    "b2": TaskBEnvB2Cfg,
    "b2w": TaskBEnvB2WCfg,
}


def emit_report(lines: list[str]) -> None:
    text = "\n".join(lines)
    print(text)
    if args_cli.report_file:
        with open(args_cli.report_file, "w", encoding="utf-8") as f:
            f.write(text + "\n")


def main():
    env_cfg = TASK_B_ENV_CFGS[args_cli.robot]()
    env_cfg.scene.num_envs = args_cli.num_envs

    env = ManagerBasedRLEnv(env_cfg)

    for name, articulation in env.scene.articulations.items():
        print("-" * 100)
        print("Robot name:", name)
        print("Bodies:", articulation.num_bodies, "->", articulation.body_names)
        print("Joints:", articulation.num_joints, "->", articulation.joint_names)
        articulation.set_joint_position_target(articulation.data.default_joint_pos)

    action_space = env.action_space
    obs, info = env.reset()
    report_printed = False

    for i in itertools.count():
        if (not args_cli.exit_after_report) and (not simulation_app.is_running()):
            break
        action = torch.zeros(action_space.shape, device=env.device)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated | truncated

        if (not report_printed) and (i + 1 >= args_cli.report_after_steps):
            robot = env.scene["robot"]
            env_id = 0
            report_lines = [
                "=" * 100,
                f"Report step: {i + 1}",
                f"Robot cfg: {args_cli.robot}",
                f"Root pos w: {robot.data.root_pos_w[env_id].detach().cpu().tolist()}",
                f"Root quat w: {robot.data.root_quat_w[env_id].detach().cpu().tolist()}",
                f"Root lin vel b: {robot.data.root_lin_vel_b[env_id].detach().cpu().tolist()}",
                f"Root ang vel b: {robot.data.root_ang_vel_b[env_id].detach().cpu().tolist()}",
                f"Root height: {robot.data.root_pos_w[env_id, 2].item():.6f}",
                f"Joint names: {robot.joint_names}",
                f"Joint pos: {robot.data.joint_pos[env_id].detach().cpu().tolist()}",
                f"Joint vel: {robot.data.joint_vel[env_id].detach().cpu().tolist()}",
            ]
            emit_report(report_lines)
            report_printed = True
            if args_cli.exit_after_report:
                break

        if done.any():
            env_ids = done.nonzero(as_tuple=False).squeeze(-1)
            env.reset(env_ids=env_ids)


if __name__ == "__main__":
    main()
    # close sim app
    simulation_app.close()
