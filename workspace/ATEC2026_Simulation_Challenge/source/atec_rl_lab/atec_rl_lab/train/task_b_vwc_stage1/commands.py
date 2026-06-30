import math
from collections.abc import Sequence

import torch
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass

import atec_rl_lab.train.locomotion.velocity.mdp as loco_mdp


class UniformEeGoalCommand(CommandTerm):
    """Uniformly sampled base-frame EE position goal."""

    cfg: "UniformEeGoalCommandCfg"

    def __init__(self, cfg: "UniformEeGoalCommandCfg", env):
        super().__init__(cfg, env)
        self.ee_goal_b = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self.ee_goal_b

    def _update_metrics(self):
        return

    def _resample_command(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return
        env_ids_t = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        x = torch.empty(len(env_ids), device=self.device).uniform_(*self.cfg.ranges.pos_x)
        y = torch.empty(len(env_ids), device=self.device).uniform_(*self.cfg.ranges.pos_y)
        z = torch.empty(len(env_ids), device=self.device).uniform_(*self.cfg.ranges.pos_z)
        self.ee_goal_b[env_ids_t, 0] = x
        self.ee_goal_b[env_ids_t, 1] = y
        self.ee_goal_b[env_ids_t, 2] = z

    def _update_command(self):
        return


@configclass
class UniformEeGoalCommandCfg(CommandTermCfg):
    class_type: type = UniformEeGoalCommand

    @configclass
    class Ranges:
        pos_x: tuple[float, float] = (0.45, 0.65)
        pos_y: tuple[float, float] = (-0.20, 0.20)
        pos_z: tuple[float, float] = (0.15, 0.45)

    resampling_time_range: tuple[float, float] = (1.0, 3.0)
    debug_vis: bool = False
    ranges: Ranges = Ranges()


@configclass
class CommandsCfg:
    """Stage1 commands: base velocity plus EE goal."""

    base_velocity = loco_mdp.UniformThresholdVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(3.0, 3.0),
        rel_standing_envs=0.05,
        rel_heading_envs=0.0,
        heading_command=False,
        debug_vis=False,
        ranges=loco_mdp.UniformThresholdVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.8, 0.8),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(-1.0, 1.0),
            heading=(-math.pi, math.pi),
        ),
    )

    ee_goal = UniformEeGoalCommandCfg()
