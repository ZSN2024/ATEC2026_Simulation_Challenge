import torch
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils import configclass

from .task_space import ee_orientation_error_rpy, world_to_base


JOINT_VEL_SCALE = 0.05
OBS_CLIP = 100.0
ROOT_VEL_CLIP = 10.0
EE_VEL_CLIP = 10.0
CONTACT_FORCE_SCALE = 100.0
CONTACT_FORCE_CLIP = 10.0


def _sanitize(tensor: torch.Tensor, clip: float = OBS_CLIP) -> torch.Tensor:
    return torch.nan_to_num(tensor, nan=0.0, posinf=clip, neginf=-clip).clamp(-clip, clip)


def _robot(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> Articulation:
    return env.scene[asset_cfg.name]


def ee_goal_command(env) -> torch.Tensor:
    return env.command_manager.get_command("ee_goal")


def ee_goal_orientation_command(env) -> torch.Tensor:
    return env.command_manager.get_term("ee_goal").command_rpy


def ee_current_pos_b(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset = _robot(env, asset_cfg)
    ee_body_id = asset.find_bodies("gripper_base")[0][0]
    return world_to_base(
        asset.data.root_pos_w,
        asset.data.root_quat_w,
        asset.data.body_pos_w[:, ee_body_id],
    )


def ee_current_vel_b(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset = _robot(env, asset_cfg)
    ee_body_id = asset.find_bodies("gripper_base")[0][0]
    if not hasattr(asset.data, "body_lin_vel_w"):
        return torch.zeros((env.num_envs, 3), device=env.device)
    root_lin_vel_w = asset.data.root_lin_vel_w
    ee_lin_vel_w = asset.data.body_lin_vel_w[:, ee_body_id]
    return world_to_base(
        torch.zeros_like(asset.data.root_pos_w),
        asset.data.root_quat_w,
        ee_lin_vel_w - root_lin_vel_w,
    )


def ee_position_error_b(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    return env.command_manager.get_command("ee_goal") - ee_current_pos_b(env, asset_cfg=asset_cfg)


def ee_current_quat_b(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset = _robot(env, asset_cfg)
    ee_body_id = asset.find_bodies("gripper_base")[0][0]
    root_pos_w = asset.data.root_pos_w
    root_quat_w = asset.data.root_quat_w
    ee_quat_w = asset.data.body_quat_w[:, ee_body_id]
    return math_utils.quat_mul(math_utils.quat_conjugate(root_quat_w), ee_quat_w)


def ee_orientation_error_obs(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    current_quat_b = ee_current_quat_b(env, asset_cfg=asset_cfg)
    goal_quat_b = env.command_manager.get_term("ee_goal").command_quat
    return ee_orientation_error_rpy(goal_quat_b, current_quat_b)


def ee_command_phase(env) -> torch.Tensor:
    term = env.command_manager.get_term("ee_goal")
    phase = (term.ee_goal_timer / term.ee_goal_total_time.clamp_min(1.0e-6)).unsqueeze(-1)
    return phase.clamp(0.0, 1.0)


def ee_traj_phase(env) -> torch.Tensor:
    term = env.command_manager.get_term("ee_goal")
    phase = (term.ee_goal_timer / term.ee_goal_traj_time.clamp_min(1.0e-6)).unsqueeze(-1)
    return phase.clamp(0.0, 1.0)


def ee_hold_phase(env) -> torch.Tensor:
    term = env.command_manager.get_term("ee_goal")
    hold_timer = (term.ee_goal_timer - term.ee_goal_traj_time).clamp_min(0.0)
    hold_time = (term.ee_goal_total_time - term.ee_goal_traj_time).clamp_min(1.0e-6)
    return (hold_timer / hold_time).unsqueeze(-1).clamp(0.0, 1.0)


def wheel_contact_summary(
    env,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_sensor", body_names=[".*_foot"]),
    threshold: float = 1.0,
) -> torch.Tensor:
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0]
    count = torch.mean((forces > threshold).float(), dim=1, keepdim=True)
    max_force = torch.max(forces, dim=1, keepdim=True)[0] / CONTACT_FORCE_SCALE
    return torch.cat([count, max_force.clamp(0.0, CONTACT_FORCE_CLIP)], dim=-1)


def last_policy_action(env) -> torch.Tensor:
    if hasattr(env, "_policy_action_dim"):
        return env.action_manager.action[:, : env._policy_action_dim]
    return env.action_manager.action


def current_policy_obs(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset = _robot(env, asset_cfg)
    joint_ids = asset_cfg.joint_ids
    joint_pos = asset.data.joint_pos[:, joint_ids] - asset.data.default_joint_pos[:, joint_ids]
    joint_vel = asset.data.joint_vel[:, joint_ids] * JOINT_VEL_SCALE
    obs = torch.cat(
        [
            asset.data.root_ang_vel_b,
            asset.data.projected_gravity_b,
            env.command_manager.get_command("base_velocity"),
            env.command_manager.get_command("ee_goal"),
            ee_goal_orientation_command(env),
            joint_pos,
            joint_vel,
            last_policy_action(env),
        ],
        dim=-1,
    )
    return _sanitize(obs)


def policy_obs_with_history(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    obs = current_policy_obs(env, asset_cfg=asset_cfg)
    if hasattr(env, "update_policy_history") and hasattr(env, "_policy_history"):
        return env.update_policy_history(obs)
    history_len = getattr(env, "history_len", 10)
    return torch.cat([obs, torch.zeros((env.num_envs, obs.shape[-1] * history_len), device=obs.device)], dim=-1)


def undesired_contact_summary(
    env,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_sensor", body_names=["base.*", ".*_hip", ".*_thigh"]),
    threshold: float = 1.0,
) -> torch.Tensor:
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0]
    body_count = max(len(sensor_cfg.body_ids), 1)
    count = torch.sum((forces > threshold).float(), dim=1, keepdim=True) / float(body_count)
    max_force = torch.max(forces, dim=1, keepdim=True)[0] / CONTACT_FORCE_SCALE
    mean_force = torch.mean(forces, dim=1, keepdim=True) / CONTACT_FORCE_SCALE
    contact_obs = torch.cat(
        [
            count.clamp(0.0, 1.0),
            max_force.clamp(0.0, CONTACT_FORCE_CLIP),
            mean_force.clamp(0.0, CONTACT_FORCE_CLIP),
        ],
        dim=-1,
    )
    return torch.nan_to_num(contact_obs, nan=0.0, posinf=CONTACT_FORCE_CLIP, neginf=0.0)


def root_state_privileged(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset = _robot(env, asset_cfg)
    return torch.cat(
        [
            asset.data.root_lin_vel_b.clamp(-ROOT_VEL_CLIP, ROOT_VEL_CLIP),
            asset.data.root_ang_vel_b.clamp(-ROOT_VEL_CLIP, ROOT_VEL_CLIP),
            asset.data.projected_gravity_b,
            asset.data.root_pos_w[:, 2:3].clamp(-5.0, 5.0),
        ],
        dim=-1,
    )


def critic_privileged_obs(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    contact_sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_sensor", body_names=["base.*", ".*_hip", ".*_thigh"]),
) -> torch.Tensor:
    critic_obs = torch.cat(
        [
            current_policy_obs(env, asset_cfg=asset_cfg),
            root_state_privileged(env, asset_cfg=asset_cfg),
            ee_current_pos_b(env, asset_cfg=asset_cfg),
            ee_current_vel_b(env, asset_cfg=asset_cfg).clamp(-EE_VEL_CLIP, EE_VEL_CLIP),
            ee_position_error_b(env, asset_cfg=asset_cfg),
            ee_orientation_error_obs(env, asset_cfg=asset_cfg),
            ee_traj_phase(env),
            ee_hold_phase(env),
            ee_command_phase(env),
            wheel_contact_summary(env, sensor_cfg=SceneEntityCfg("contact_sensor", body_names=[".*_foot"])),
            undesired_contact_summary(env, sensor_cfg=contact_sensor_cfg),
        ],
        dim=-1,
    )
    return _sanitize(critic_obs)


@configclass
class ObservationsCfg:
    """Stage1 observation groups with VWC-style 10-frame policy history."""

    @configclass
    class PolicyCfg(ObsGroup):
        policy = ObsTerm(
            func=policy_obs_with_history,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        critic = ObsTerm(
            func=critic_privileged_obs,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True),
                "contact_sensor_cfg": SceneEntityCfg(
                    "contact_sensor", body_names=["base.*", ".*_hip", ".*_thigh"]
                ),
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()
