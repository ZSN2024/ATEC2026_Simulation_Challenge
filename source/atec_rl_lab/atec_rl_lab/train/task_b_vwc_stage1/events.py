import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass


def reset_root_state_vwc_stage1(
    env,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset root pose around TaskB's actual per-env spawn positions.

    TaskB's generated terrain places cloned robots on a grid, but ``scene.env_origins``
    is not a valid reset offset in this environment. Cache the real spawn positions
    once and reset to them with the same perturbations used by VWC.
    """
    asset: Articulation = env.scene[asset_cfg.name]

    cache_name = "_task_b_vwc_stage1_base_root_pos_w"
    if not hasattr(env, cache_name):
        setattr(env, cache_name, asset.data.root_pos_w.detach().clone())
    base_root_pos_w = getattr(env, cache_name)

    root_states = asset.data.default_root_state[env_ids].clone()

    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    pose_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)

    positions = base_root_pos_w[env_ids].clone()
    positions += pose_samples[:, 0:3]

    orientations_delta = math_utils.quat_from_euler_xyz(
        pose_samples[:, 3], pose_samples[:, 4], pose_samples[:, 5]
    )
    orientations = math_utils.quat_mul(root_states[:, 3:7], orientations_delta)

    range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    velocity_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)
    velocities = root_states[:, 7:13] + velocity_samples

    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)


@configclass
class EventsCfg:
    """Stage1 event switches."""

    enable_domain_randomization = False
