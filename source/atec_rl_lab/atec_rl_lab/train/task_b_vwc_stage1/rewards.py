import math

import torch
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils import configclass

import atec_rl_lab.train.locomotion.velocity.mdp as loco_mdp

from .task_space import ee_orientation_error_rpy, world_to_base


def alive(env) -> torch.Tensor:
    return torch.ones(env.num_envs, device=env.device)


def track_lin_vel_x_exp(
    env,
    std: float,
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    error = torch.square(env.command_manager.get_command(command_name)[:, 0] - asset.data.root_lin_vel_b[:, 0])
    reward = torch.exp(-error / std**2)
    reward *= torch.clamp(-asset.data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def track_ee_position_exp(
    env,
    std: float,
    command_name: str = "ee_goal",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    ee_body_id = asset.find_bodies("gripper_base")[0][0]
    ee_goal_b = env.command_manager.get_command(command_name)
    ee_pos_b = world_to_base(
        asset.data.root_pos_w,
        asset.data.root_quat_w,
        asset.data.body_pos_w[:, ee_body_id],
    )
    error = torch.sum(torch.square(ee_goal_b - ee_pos_b), dim=1)
    reward = torch.exp(-error / std**2)
    reward *= torch.clamp(-asset.data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def _upright_scale(env) -> torch.Tensor:
    asset: Articulation = env.scene["robot"]
    return torch.clamp(-asset.data.projected_gravity_b[:, 2], 0, 0.7) / 0.7


def _ee_pos_b(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    ee_body_id = asset.find_bodies("gripper_base")[0][0]
    return world_to_base(
        asset.data.root_pos_w,
        asset.data.root_quat_w,
        asset.data.body_pos_w[:, ee_body_id],
    )


def ee_goal_distance(env, command_name: str = "ee_goal") -> torch.Tensor:
    ee_goal_b = env.command_manager.get_command(command_name)
    return torch.norm(ee_goal_b - _ee_pos_b(env), dim=1)


def low_ee_goal_intensity(
    env,
    command_name: str = "ee_goal",
    start_z_b: float = -0.30,
    full_z_b: float = -0.65,
) -> torch.Tensor:
    ee_goal_b = env.command_manager.get_command(command_name)
    return torch.clamp((start_z_b - ee_goal_b[:, 2]) / (start_z_b - full_z_b), 0.0, 1.0)


def ee_approach_progress(
    env,
    command_name: str = "ee_goal",
    scale: float = 20.0,
    max_progress: float = 0.05,
    goal_change_threshold: float = 0.02,
) -> torch.Tensor:
    curr_dist = ee_goal_distance(env, command_name=command_name)
    goal = env.command_manager.get_command(command_name)

    needs_init = (
        not hasattr(env, "_prev_ee_goal_dist")
        or env._prev_ee_goal_dist.shape[0] != env.num_envs
        or env._prev_ee_goal_dist.device != curr_dist.device
    )
    if needs_init:
        env._prev_ee_goal_dist = curr_dist.detach().clone()
        env._prev_ee_goal_b = goal.detach().clone()
        return torch.zeros_like(curr_dist)

    goal_changed = torch.norm(goal - env._prev_ee_goal_b, dim=1) > goal_change_threshold
    episode_reset = getattr(env, "episode_length_buf", torch.ones_like(curr_dist)) <= 1
    prev_dist = torch.where(goal_changed | episode_reset, curr_dist, env._prev_ee_goal_dist)
    progress = torch.clamp(prev_dist - curr_dist, 0.0, max_progress)
    reward = torch.tanh(scale * progress) * _upright_scale(env)

    env._prev_ee_goal_dist = curr_dist.detach().clone()
    env._prev_ee_goal_b = goal.detach().clone()
    return reward


def track_ee_orientation_exp(
    env,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    ee_body_id = asset.find_bodies("gripper_base")[0][0]
    goal_quat_b = env.command_manager.get_term("ee_goal").command_quat
    current_quat_b = math_utils.quat_mul(
        math_utils.quat_conjugate(asset.data.root_quat_w),
        asset.data.body_quat_w[:, ee_body_id],
    )
    error = torch.sum(torch.abs(ee_orientation_error_rpy(goal_quat_b, current_quat_b)), dim=1)
    reward = torch.exp(-error / std)
    reward *= torch.clamp(-asset.data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def adaptive_leg_posture_exp(
    env,
    std: float,
    low_scale: float = 0.40,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=".*_(hip|thigh|calf)_joint"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    joint_error = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    reward = torch.exp(-torch.mean(torch.square(joint_error), dim=1) / std)
    low = low_ee_goal_intensity(env)
    reward *= (1.0 - low) + low_scale * low
    reward *= torch.clamp(-asset.data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def wheel_contact_required(
    env,
    threshold: float,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_sensor", body_names=[".*_foot"]),
) -> torch.Tensor:
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > threshold
    reward = contacts.float().mean(dim=1)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def stand_still_zero_cmd(
    env,
    command_name: str = "base_velocity",
    joint_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=".*_(hip|thigh|calf)_joint"),
    wheel_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=".*_foot_joint"),
    lin_vel_threshold: float = 0.2,
    yaw_threshold: float = 0.5,
    std: float = 0.25,
) -> torch.Tensor:
    asset: Articulation = env.scene[joint_cfg.name]
    command = env.command_manager.get_command(command_name)
    standing = (torch.abs(command[:, 0]) < lin_vel_threshold) & (torch.abs(command[:, 2]) < yaw_threshold)

    leg_error = torch.mean(
        torch.square(asset.data.joint_pos[:, joint_cfg.joint_ids] - asset.data.default_joint_pos[:, joint_cfg.joint_ids]),
        dim=1,
    )
    wheel_vel = torch.mean(torch.square(asset.data.joint_vel[:, wheel_cfg.joint_ids]), dim=1)
    base_penalty = (
        torch.square(asset.data.root_lin_vel_b[:, 0])
        + torch.square(asset.data.root_lin_vel_b[:, 1])
        + 0.5 * torch.square(asset.data.root_ang_vel_b[:, 2])
        + 0.1 * wheel_vel
        + 0.5 * leg_error
    )
    reward = torch.exp(-base_penalty / std)
    reward *= standing.float()
    reward *= torch.clamp(-asset.data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def side_slip_l2(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    reward = torch.square(asset.data.root_lin_vel_b[:, 1])
    reward *= torch.clamp(-asset.data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def _action_rate_l2_slice(env, start: int, end: int) -> torch.Tensor:
    action = env.action_manager.action[:, start:end]
    prev_action = env.action_manager.prev_action[:, start:end]
    return torch.sum(torch.square(action - prev_action), dim=1)


def leg_action_rate_l2(env, leg_action_dim: int = 12) -> torch.Tensor:
    return _action_rate_l2_slice(env, 0, leg_action_dim)


def wheel_action_rate_l2(env, leg_action_dim: int = 12) -> torch.Tensor:
    return _action_rate_l2_slice(env, leg_action_dim, env.action_manager.action.shape[1])


def leg_action_l2(env, leg_action_dim: int = 12) -> torch.Tensor:
    return torch.sum(torch.square(env.action_manager.action[:, :leg_action_dim]), dim=1)


def hip_pos_l2(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=".*_hip_joint"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    error = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    reward = torch.sum(torch.square(error), dim=1)
    reward *= torch.clamp(-asset.data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def adaptive_flat_orientation_l2(
    env,
    low_scale: float = 0.35,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    low = low_ee_goal_intensity(env)
    reward = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)
    reward *= (1.0 - low) + low_scale * low
    reward *= torch.clamp(-asset.data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def adaptive_base_height_l2(
    env,
    standing_height: float = 0.78,
    low_height: float = 0.48,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    low = low_ee_goal_intensity(env)
    target_height = standing_height * (1.0 - low) + low_height * low
    reward = torch.square(asset.data.root_pos_w[:, 2] - target_height)
    reward *= torch.clamp(-asset.data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def min_base_height_l2(
    env,
    min_height: float = 0.35,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    reward = torch.square(torch.clamp(min_height - asset.data.root_pos_w[:, 2], min=0.0))
    reward *= torch.clamp(-asset.data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


@configclass
class RewardsCfg:
    """Stage1 rewards with base and EE tracking."""

    alive = RewTerm(func=alive, weight=0.5)
    track_lin_vel_x_exp = RewTerm(
        func=track_lin_vel_x_exp,
        weight=1.5,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    track_ang_vel_z_exp = RewTerm(
        func=loco_mdp.track_ang_vel_z_exp,
        weight=0.5,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    track_ee_position_exp = RewTerm(
        func=track_ee_position_exp,
        weight=1.0,
        params={"command_name": "ee_goal", "std": math.sqrt(0.12)},
    )
    track_ee_orientation_exp = RewTerm(
        func=track_ee_orientation_exp,
        weight=0.3,
        params={"std": 1.0},
    )
    ee_approach_progress = RewTerm(
        func=ee_approach_progress,
        weight=1.2,
        params={
            "command_name": "ee_goal",
            "scale": 20.0,
            "max_progress": 0.05,
            "goal_change_threshold": 0.02,
        },
    )
    adaptive_leg_posture_exp = RewTerm(
        func=adaptive_leg_posture_exp,
        weight=0.8,
        params={
            "std": 0.35,
            "low_scale": 0.40,
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*_(hip|thigh|calf)_joint"),
        },
    )
    wheel_contact_required = RewTerm(
        func=wheel_contact_required,
        weight=0.8,
        params={"threshold": 1.0, "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=[".*_foot"])},
    )
    stand_still_zero_cmd = RewTerm(
        func=stand_still_zero_cmd,
        weight=0.6,
        params={
            "command_name": "base_velocity",
            "joint_cfg": SceneEntityCfg("robot", joint_names=".*_(hip|thigh|calf)_joint"),
            "wheel_cfg": SceneEntityCfg("robot", joint_names=".*_foot_joint"),
            "lin_vel_threshold": 0.2,
            "yaw_threshold": 0.5,
            "std": 0.25,
        },
    )
    side_slip_l2 = RewTerm(func=side_slip_l2, weight=-0.5)
    leg_action_l2 = RewTerm(func=leg_action_l2, weight=-0.003, params={"leg_action_dim": 12})
    leg_action_rate_l2 = RewTerm(func=leg_action_rate_l2, weight=-0.015, params={"leg_action_dim": 12})
    wheel_action_rate_l2 = RewTerm(func=wheel_action_rate_l2, weight=-0.001, params={"leg_action_dim": 12})
    joint_acc_l2 = RewTerm(
        func=loco_mdp.joint_acc_l2,
        weight=-5.0e-7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_(hip|thigh|calf)_joint")},
    )
    joint_torques_l2 = RewTerm(
        func=loco_mdp.joint_torques_l2,
        weight=-1.0e-5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_(hip|thigh|calf)_joint")},
    )
    joint_pos_limits = RewTerm(
        func=loco_mdp.joint_pos_limits,
        weight=-5.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_(hip|thigh|calf)_joint")},
    )
    hip_pos_l2 = RewTerm(
        func=hip_pos_l2,
        weight=-0.15,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_hip_joint")},
    )
    adaptive_flat_orientation_l2 = RewTerm(
        func=adaptive_flat_orientation_l2,
        weight=-4.0,
        params={"low_scale": 0.35, "asset_cfg": SceneEntityCfg("robot")},
    )
    adaptive_base_height_l2 = RewTerm(
        func=adaptive_base_height_l2,
        weight=-3.0,
        params={"standing_height": 0.78, "low_height": 0.48, "asset_cfg": SceneEntityCfg("robot")},
    )
    min_base_height_l2 = RewTerm(
        func=min_base_height_l2,
        weight=-12.0,
        params={"min_height": 0.35, "asset_cfg": SceneEntityCfg("robot")},
    )
    lin_vel_z_l2 = RewTerm(func=loco_mdp.lin_vel_z_l2, weight=-1.5)
    ang_vel_xy_l2 = RewTerm(func=loco_mdp.ang_vel_xy_l2, weight=-0.2)
    undesired_contacts = RewTerm(
        func=loco_mdp.undesired_contacts,
        weight=-5.0,
        params={
            "threshold": 1.0,
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=["base.*", ".*_hip", ".*_thigh"]),
        },
    )
