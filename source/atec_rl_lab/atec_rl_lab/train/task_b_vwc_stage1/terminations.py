import torch
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg


def root_height_below_minimum(
    env,
    minimum_height: float = 0.1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_pos_w[:, 2] < minimum_height


def roll_pitch_exceeded(
    env,
    roll_limit: float = 0.8,
    pitch_limit: float = 0.8,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """VWC low-level termination: reset when base roll or pitch exceeds the configured radian limits."""
    asset: Articulation = env.scene[asset_cfg.name]
    quat = asset.data.root_quat_w
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]

    roll = torch.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = torch.asin(torch.clamp(2.0 * (w * y - z * x), -1.0, 1.0))
    return (torch.abs(roll) > roll_limit) | (torch.abs(pitch) > pitch_limit)
