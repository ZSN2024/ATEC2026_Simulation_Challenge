"""Play Task B Stage1 checkpoints with RSL-RL."""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.append(SCRIPTS_ROOT)
sys.path.append(os.path.join(SCRIPTS_ROOT, "rsl_rl"))

import cli_args  # isort: skip


parser = argparse.ArgumentParser(description="Play Task B Stage1 checkpoint.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during play.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="ATEC-TaskB-B2wPiper-VWC-Stage1-v0", help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--use_pretrained_checkpoint", action="store_true", help="Use the pre-trained checkpoint.")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time if possible.")
parser.add_argument("--export", action="store_true", default=False, help="Export JIT/ONNX policy before play.")
parser.add_argument("--print_metrics", action="store_true", default=False, help="Print live play metrics.")
parser.add_argument("--print_interval", type=int, default=50, help="Metric print interval in simulation steps.")
parser.add_argument("--show_command_markers", action="store_true", default=False, help="Visualize EE goal command markers.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import importlib.metadata as metadata
import os
import time

import gymnasium as gym
import torch
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import atec_rl_lab.train.task_b_vwc_stage1  # noqa: F401  # isort: skip
from atec_rl_lab.train.task_b_vwc_stage1.task_space import (  # noqa: E402
    base_to_world,
    ee_orientation_error_rpy,
    quat_from_rpy,
    world_to_base,
)
import isaaclab.utils.math as math_utils  # noqa: E402


def _get_policy_export_module(runner):
    """Return a module accepted by IsaacLab's RSL-RL exporter, if available."""
    alg = runner.alg
    for attr in ("actor_critic", "policy", "actor"):
        module = getattr(alg, attr, None)
        if module is not None:
            return module
    return alg


def _get_policy_normalizer(policy_module):
    if hasattr(policy_module, "actor") and hasattr(policy_module.actor, "obs_normalizer"):
        return policy_module.actor.obs_normalizer
    for attr in ("obs_normalizer", "actor_obs_normalizer", "student_obs_normalizer"):
        normalizer = getattr(policy_module, attr, None)
        if normalizer is not None:
            return normalizer
    return None


def _maybe_create_goal_marker(enabled: bool):
    if not enabled:
        return None
    try:
        from isaaclab.markers import VisualizationMarkers
        from isaaclab.markers.config import POSITION_GOAL_MARKER_CFG

        return VisualizationMarkers(POSITION_GOAL_MARKER_CFG.replace(prim_path="/Visuals/TaskBStage1/ee_goal"))
    except Exception as exc:
        print(f"[WARN] Failed to create EE goal marker: {exc}")
        return None


_MARKER_WARNED = False
_METRICS_WARNED = False
_GROUND_GOAL_WARNED = False

GROUND_GOAL_RESAMPLE_TIME_S = 3.0
GROUND_GOAL_X_RANGE = (0.35, 0.85)
GROUND_GOAL_Y_RANGE = (-0.35, 0.35)
GROUND_GOAL_Z_W = 0.10


def _get_robot(unwrapped_env):
    base_env = _get_manager_env(unwrapped_env)
    robot = getattr(base_env, "_robot", None)
    if robot is not None:
        return robot
    return base_env.scene["robot"]


def _get_manager_env(env):
    """Find the wrapped Isaac env that owns command_manager/scene."""
    current = env
    visited = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        if hasattr(current, "command_manager"):
            return current
        for attr in ("unwrapped", "env", "_env"):
            child = getattr(current, attr, None)
            if child is not None and child is not current:
                current = child
                break
        else:
            break
    raise AttributeError("Could not find wrapped env with command_manager")


def _visualize_ee_goal(marker, play_context):
    global _MARKER_WARNED
    if marker is None:
        return
    try:
        robot = play_context["robot"]
        ee_goal_b = play_context["command_manager"].get_command("ee_goal")
        ee_goal_w = base_to_world(robot.data.root_pos_w, robot.data.root_quat_w, ee_goal_b)
        marker.visualize(translations=ee_goal_w)
    except Exception as exc:
        if not _MARKER_WARNED:
            print(f"[WARN] Failed to update EE goal marker; disabling marker updates: {exc}")
            _MARKER_WARNED = True


def _sample_near_ground_ee_goals(play_context, force: bool = False):
    """Override play-only EE commands with world-frame near-ground target points."""
    global _GROUND_GOAL_WARNED
    try:
        robot = play_context["robot"]
        command_manager = play_context["command_manager"]
        term = command_manager.get_term("ee_goal")
        device = robot.data.root_pos_w.device
        num_envs = robot.data.root_pos_w.shape[0]

        if "ground_goal_timer" not in play_context:
            play_context["ground_goal_timer"] = torch.full(
                (num_envs,),
                float(GROUND_GOAL_RESAMPLE_TIME_S),
                dtype=torch.float32,
                device=device,
            )
            play_context["ground_goal_w"] = torch.zeros((num_envs, 3), dtype=torch.float32, device=device)
            play_context["ground_goal_rpy"] = torch.zeros((num_envs, 3), dtype=torch.float32, device=device)

        timer = play_context["ground_goal_timer"]
        if force:
            resample = torch.ones_like(timer, dtype=torch.bool)
        else:
            resample = timer >= float(GROUND_GOAL_RESAMPLE_TIME_S)
        if torch.any(resample):
            ids = torch.nonzero(resample, as_tuple=False).squeeze(-1)
            local_xy = torch.zeros((ids.numel(), 3), dtype=torch.float32, device=device)
            local_xy[:, 0] = torch.empty(ids.numel(), device=device).uniform_(*GROUND_GOAL_X_RANGE)
            local_xy[:, 1] = torch.empty(ids.numel(), device=device).uniform_(*GROUND_GOAL_Y_RANGE)
            xy_w = base_to_world(robot.data.root_pos_w[ids], robot.data.root_quat_w[ids], local_xy)[:, :2]
            play_context["ground_goal_w"][ids, :2] = xy_w
            play_context["ground_goal_w"][ids, 2] = float(GROUND_GOAL_Z_W)
            play_context["ground_goal_rpy"][ids] = 0.0
            timer[ids] = 0.0

        term.ee_goal_b[:] = world_to_base(
            robot.data.root_pos_w,
            robot.data.root_quat_w,
            play_context["ground_goal_w"],
        )
        term.ee_goal_rpy_b[:] = play_context["ground_goal_rpy"]
        term.ee_goal_quat_b[:] = quat_from_rpy(term.ee_goal_rpy_b)
    except Exception as exc:
        if not _GROUND_GOAL_WARNED:
            print(f"[WARN] Failed to override near-ground EE goals; using env commands: {exc}")
            _GROUND_GOAL_WARNED = True


def _advance_near_ground_goal_timer(play_context, dt: float):
    timer = play_context.get("ground_goal_timer")
    if timer is not None:
        timer += float(dt)


def _resample_done_near_ground_goals(play_context, dones):
    timer = play_context.get("ground_goal_timer")
    if timer is not None and isinstance(dones, torch.Tensor):
        timer[dones.bool()] = float(GROUND_GOAL_RESAMPLE_TIME_S)


def _print_play_metrics(play_context, reward, dones, timestep: int):
    global _METRICS_WARNED
    try:
        robot = play_context["robot"]
        command_manager = play_context["command_manager"]
        num_envs = play_context["num_envs"]
        ee_body_id = robot.find_bodies("gripper_base")[0][0]
        base_cmd = command_manager.get_command("base_velocity")
        ee_goal_b = command_manager.get_command("ee_goal")
        ee_goal_rpy = command_manager.get_term("ee_goal").command_rpy
        ee_goal_quat = command_manager.get_term("ee_goal").command_quat
        ee_pos_b = world_to_base(robot.data.root_pos_w, robot.data.root_quat_w, robot.data.body_pos_w[:, ee_body_id])
        ee_err = torch.linalg.norm(ee_goal_b - ee_pos_b, dim=-1)
        ee_quat_b = math_utils.quat_mul(math_utils.quat_conjugate(robot.data.root_quat_w), robot.data.body_quat_w[:, ee_body_id])
        ee_orn_err = torch.linalg.norm(ee_orientation_error_rpy(ee_goal_quat, ee_quat_b), dim=-1)
        root_height = robot.data.root_pos_w[:, 2]
        root_ang = robot.data.root_ang_vel_b
        root_lin = robot.data.root_lin_vel_b
        reward_mean = reward.mean().item() if isinstance(reward, torch.Tensor) else float(reward)
        done_count = int(dones.sum().item()) if isinstance(dones, torch.Tensor) else int(dones)
        print(
            "[PLAY_METRICS] "
            f"step={timestep} reward_mean={reward_mean:.4f} done={done_count}/{num_envs} "
            f"height_mean={root_height.mean().item():.3f} height_min={root_height.min().item():.3f} "
            f"ee_err_mean={ee_err.mean().item():.3f} ee_err_env0={ee_err[0].item():.3f} "
            f"ee_orn_err_mean={ee_orn_err.mean().item():.3f} ee_orn_err_env0={ee_orn_err[0].item():.3f} "
            f"base_lin_env0={root_lin[0].detach().cpu().tolist()} "
            f"base_ang_env0={root_ang[0].detach().cpu().tolist()} "
            f"base_cmd_env0={base_cmd[0].detach().cpu().tolist()} "
            f"ee_goal_env0={ee_goal_b[0].detach().cpu().tolist()} "
            f"ee_goal_rpy_env0={ee_goal_rpy[0].detach().cpu().tolist()} "
            f"ee_pos_env0={ee_pos_b[0].detach().cpu().tolist()}",
            flush=True,
        )
    except Exception as exc:
        if not _METRICS_WARNED:
            print(f"[WARN] Failed to print play metrics; disabling metric prints: {exc}")
            _METRICS_WARNED = True


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    task_name = args_cli.task.split(":")[-1]
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else 64

    from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg

    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    if hasattr(env_cfg.scene, "terrain"):
        env_cfg.scene.terrain.max_init_terrain_level = None
        if env_cfg.scene.terrain.terrain_generator is not None:
            env_cfg.scene.terrain.terrain_generator.num_rows = 5
            env_cfg.scene.terrain.terrain_generator.num_cols = 5
            env_cfg.scene.terrain.terrain_generator.curriculum = False

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", task_name)
        if not resume_path:
            print("[INFO] No pre-trained checkpoint is available for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)
    env_cfg.log_dir = log_dir
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during play.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    manager_env = env.unwrapped
    play_context = {
        "robot": _get_robot(manager_env),
        "command_manager": manager_env.command_manager,
        "num_envs": manager_env.num_envs,
    }
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    policy = runner.get_inference_policy(device=env.unwrapped.device)
    policy_nn = _get_policy_export_module(runner)

    if args_cli.export:
        normalizer = _get_policy_normalizer(policy_nn)
        export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
        try:
            export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
            export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")
        except Exception as exc:
            print(f"[WARN] Policy export failed; continuing play without export: {exc}")

    dt = env.unwrapped.step_dt
    _sample_near_ground_ee_goals(play_context, force=True)
    obs = env.get_observations()
    timestep = 0
    goal_marker = _maybe_create_goal_marker(args_cli.show_command_markers)
    while simulation_app.is_running():
        start_time = time.time()
        with torch.inference_mode():
            actions = policy(obs)
            obs, reward, dones, _ = env.step(actions)
            _advance_near_ground_goal_timer(play_context, dt)
            _resample_done_near_ground_goals(play_context, dones)
            _sample_near_ground_ee_goals(play_context)
            obs = env.get_observations()
            if hasattr(policy_nn, "reset"):
                policy_nn.reset(dones)
        _visualize_ee_goal(goal_marker, play_context)
        if args_cli.print_metrics and timestep % max(args_cli.print_interval, 1) == 0:
            _print_play_metrics(play_context, reward, dones, timestep)
        timestep += 1
        if args_cli.video:
            if timestep == args_cli.video_length:
                break
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
