"""Task D training: standard rough terrain + forward rewards."""
import torch
from isaaclab.utils import configclass
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg

from atec_rl_lab.train.locomotion.velocity.config.quadruped.unitree_b2.rough_env_cfg import (
    UnitreeB2RoughEnvCfg,
)


def robot_forward_progress(env, asset_cfg):
    robot = env.scene[asset_cfg.name]
    return torch.clamp(robot.data.root_pos_w[:, 0] + 3.0, min=0.0) * 1.0


def robot_crossed_obstacle(env, asset_cfg):
    robot = env.scene[asset_cfg.name]
    return (robot.data.root_pos_w[:, 0] > 2.0).float() * 10.0


@configclass
class UnitreeB2TaskDEnvCfg(UnitreeB2RoughEnvCfg):
    """B2 obstacle traversal — proven rough terrain + forward/cross rewards."""

    def __post_init__(self):
        super().__post_init__()

        # Keep the inherited rough terrain (proven working with 4096 envs)

        # Add Task D rewards
        self.rewards.forward_progress = RewTerm(
            func=robot_forward_progress, weight=1.0,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        self.rewards.crossed_obstacle = RewTerm(
            func=robot_crossed_obstacle, weight=1.0,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )

        self.disable_zero_weight_rewards()
