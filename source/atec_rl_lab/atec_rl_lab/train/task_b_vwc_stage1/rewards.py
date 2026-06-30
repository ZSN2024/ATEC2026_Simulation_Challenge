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


def leg_posture_exp(
    env,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=".*_(hip|thigh|calf)_joint"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    joint_error = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    reward = torch.exp(-torch.mean(torch.square(joint_error), dim=1) / std)
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
        weight=0.8,
        params={"command_name": "ee_goal", "std": math.sqrt(0.05)},
    )
    track_ee_orientation_exp = RewTerm(
        func=track_ee_orientation_exp,
        weight=0.3,
        params={"std": 1.0},
    )
    leg_posture_exp = RewTerm(
        func=leg_posture_exp,
        weight=0.8,
        params={"std": 0.35, "asset_cfg": SceneEntityCfg("robot", joint_names=".*_(hip|thigh|calf)_joint")},
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
        weight=-0.3,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_hip_joint")},
    )
    flat_orientation_l2 = RewTerm(func=loco_mdp.flat_orientation_l2, weight=-4.0)
    base_height_l2 = RewTerm(
        func=loco_mdp.base_height_l2,
        weight=-4.0,
        params={"target_height": 0.78},
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
