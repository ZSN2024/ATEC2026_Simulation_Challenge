import math
from collections.abc import Sequence

import torch
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass

import atec_rl_lab.train.locomotion.velocity.mdp as loco_mdp

from .task_space import base_to_world, quat_from_rpy, sphere_to_cartesian, world_to_base


class VwcWheelVelocityCommand(loco_mdp.UniformThresholdVelocityCommand):
    """Stage1 wheel-base curriculum command following the VWC-style progression."""

    cfg: "VwcWheelVelocityCommandCfg"

    def _update_metrics(self):
        super()._update_metrics()
        phase = self._curriculum_phase()
        phase_ranges = self._phase_ranges(phase)
        standing = (
            (torch.abs(self.vel_command_b[:, 0]) < self.cfg.lin_vel_x_clip)
            & (torch.abs(self.vel_command_b[:, 2]) < self.cfg.ang_vel_z_clip)
        ).float()
        self.metrics["curriculum_phase"] = torch.full(
            (self.num_envs,),
            float(phase),
            dtype=torch.float32,
            device=self.device,
        )
        self.metrics["lin_vel_x_min"] = torch.full((self.num_envs,), float(phase_ranges.lin_vel_x[0]), device=self.device)
        self.metrics["lin_vel_x_max"] = torch.full((self.num_envs,), float(phase_ranges.lin_vel_x[1]), device=self.device)
        self.metrics["ang_vel_z_min"] = torch.full((self.num_envs,), float(phase_ranges.ang_vel_z[0]), device=self.device)
        self.metrics["ang_vel_z_max"] = torch.full((self.num_envs,), float(phase_ranges.ang_vel_z[1]), device=self.device)
        self.metrics["standing_mask"] = standing
        self.metrics["walking_mask"] = 1.0 - standing

    def _resample_command(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return
        phase = self._curriculum_phase()
        ranges = self._phase_ranges(phase)

        env_ids_t = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        self.vel_command_b[env_ids_t, 0] = torch.empty(len(env_ids), device=self.device).uniform_(*ranges.lin_vel_x)
        self.vel_command_b[env_ids_t, 1] = 0.0
        self.vel_command_b[env_ids_t, 2] = torch.empty(len(env_ids), device=self.device).uniform_(*ranges.ang_vel_z)

        if self.cfg.heading_command and hasattr(self, "heading_target"):
            self.heading_target[env_ids_t] = 0.0

        standing_env_ids = env_ids_t[
            torch.rand(len(env_ids), device=self.device) <= float(self.cfg.rel_standing_envs)
        ]
        if standing_env_ids.numel() > 0:
            self.vel_command_b[standing_env_ids] = 0.0

        self.vel_command_b[env_ids_t, :2] *= (
            torch.norm(self.vel_command_b[env_ids_t, :2], dim=1) > self.cfg.lin_vel_x_clip
        ).unsqueeze(1)
        self.vel_command_b[env_ids_t, 2] *= (
            torch.abs(self.vel_command_b[env_ids_t, 2]) > self.cfg.ang_vel_z_clip
        )

    def _curriculum_phase(self) -> int:
        progress = self._training_progress()
        if progress < self.cfg.stage_a_until:
            return 0
        if progress < self.cfg.stage_b_until:
            return 1
        return 2

    def _training_progress(self) -> float:
        total_steps = max(int(self.cfg.curriculum_total_steps), 1)
        step = int(getattr(self._env, "common_step_counter", 0))
        return min(max(step / total_steps, 0.0), 1.0)

    def _phase_ranges(self, phase: int) -> "VwcWheelVelocityCommandCfg.Ranges":
        if phase == 0:
            return self.cfg.stage_a_ranges
        if phase == 1:
            return self.cfg.stage_b_ranges
        return self.cfg.stage_c_ranges


class UniformEeGoalCommand(CommandTerm):
    """VWC-style interpolated base-frame EE pose goal."""

    cfg: "UniformEeGoalCommandCfg"

    def __init__(self, cfg: "UniformEeGoalCommandCfg", env):
        super().__init__(cfg, env)
        self._robot = env.scene["robot"]
        self._goal_center_offset = torch.tensor(
            [self.cfg.sphere_center.x_offset, self.cfg.sphere_center.y_offset, self.cfg.sphere_center.z_offset],
            dtype=torch.float32,
            device=self.device,
        )
        self._collision_lower_limits = torch.tensor(
            self.cfg.collision_lower_limits,
            dtype=torch.float32,
            device=self.device,
        )
        self._collision_upper_limits = torch.tensor(
            self.cfg.collision_upper_limits,
            dtype=torch.float32,
            device=self.device,
        )
        self._collision_check_t = torch.linspace(
            0.0,
            1.0,
            int(self.cfg.num_collision_check_samples),
            dtype=torch.float32,
            device=self.device,
        ).view(1, -1, 1)
        self.ee_goal_b = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self.ee_goal_quat_b = torch.zeros((self.num_envs, 4), dtype=torch.float32, device=self.device)
        self.ee_goal_rpy_b = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self.ee_start_sphere = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self.ee_target_sphere = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self.ground_goal_w = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self.use_ground_goal = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        self.ee_goal_timer = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
        self.ee_goal_traj_time = torch.ones((self.num_envs,), dtype=torch.float32, device=self.device)
        self.ee_goal_total_time = torch.ones((self.num_envs,), dtype=torch.float32, device=self.device)
        self._resample_command(list(range(self.num_envs)))

    @property
    def command(self) -> torch.Tensor:
        return self.ee_goal_b

    @property
    def command_quat(self) -> torch.Tensor:
        return self.ee_goal_quat_b

    @property
    def command_rpy(self) -> torch.Tensor:
        return self.ee_goal_rpy_b

    def _update_metrics(self):
        traj_phase = (self.ee_goal_timer / self.ee_goal_traj_time.clamp_min(1.0e-6)).clamp(0.0, 1.0)
        hold_phase = (
            (self.ee_goal_timer - self.ee_goal_traj_time).clamp_min(0.0)
            / (self.ee_goal_total_time - self.ee_goal_traj_time).clamp_min(1.0e-6)
        ).clamp(0.0, 1.0)
        self.metrics["goal_timer"] = self.ee_goal_timer
        self.metrics["traj_time"] = self.ee_goal_traj_time
        self.metrics["total_time"] = self.ee_goal_total_time
        self.metrics["traj_phase"] = traj_phase
        self.metrics["hold_phase"] = hold_phase
        self.metrics["ground_goal_ratio"] = self.use_ground_goal.float()
        self.metrics["goal_z_b"] = self.ee_goal_b[:, 2]
        self.metrics["ground_goal_z_w"] = self.ground_goal_w[:, 2]

    def _resample_command(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return
        env_ids_t = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if torch.count_nonzero(self.ee_goal_timer[env_ids_t]) > 0:
            self.ee_start_sphere[env_ids_t] = self.ee_target_sphere[env_ids_t]
        else:
            self.ee_start_sphere[env_ids_t, 0] = torch.empty(len(env_ids), device=self.device).uniform_(
                *self.cfg.ranges.init_radius
            )
            self.ee_start_sphere[env_ids_t, 1] = torch.empty(len(env_ids), device=self.device).uniform_(
                *self.cfg.ranges.init_pitch
            )
            self.ee_start_sphere[env_ids_t, 2] = torch.empty(len(env_ids), device=self.device).uniform_(
                *self.cfg.ranges.init_yaw
            )

        ground_goal_mask = torch.rand(len(env_ids), device=self.device) < float(self.cfg.ground_goal_ratio)
        self.use_ground_goal[env_ids_t] = ground_goal_mask
        if torch.any(ground_goal_mask):
            ground_ids = env_ids_t[ground_goal_mask]
            self.ground_goal_w[ground_ids] = self._sample_ground_goal_w(ground_ids)
            self.ee_target_sphere[ground_ids] = 0.0
        if torch.any(~ground_goal_mask):
            sphere_ids = env_ids_t[~ground_goal_mask]
            self.ee_target_sphere[sphere_ids] = self._sample_target_sphere(sphere_ids)

        sphere_ids = env_ids_t[~ground_goal_mask]
        if sphere_ids.numel() > 0:
            self.ee_goal_rpy_b[sphere_ids, 0] = torch.empty(sphere_ids.numel(), device=self.device).uniform_(
                *self.cfg.ranges.delta_roll
            )
            self.ee_goal_rpy_b[sphere_ids, 1] = torch.empty(sphere_ids.numel(), device=self.device).uniform_(
                *self.cfg.ranges.delta_pitch
            )
            self.ee_goal_rpy_b[sphere_ids, 2] = torch.empty(sphere_ids.numel(), device=self.device).uniform_(
                *self.cfg.ranges.delta_yaw
            )
        if torch.any(ground_goal_mask):
            self._sample_ground_goal_rpy(env_ids_t[ground_goal_mask])
        self.ee_goal_timer[env_ids_t] = 0.0
        self.ee_goal_traj_time[env_ids_t] = torch.empty(len(env_ids), device=self.device).uniform_(
            *self.cfg.traj_time_range
        )
        self.ee_goal_total_time[env_ids_t] = self.ee_goal_traj_time[env_ids_t] + torch.empty(
            len(env_ids), device=self.device
        ).uniform_(*self.cfg.hold_time_range)
        self._update_current_goal(env_ids_t)

    def _update_command(self):
        self.ee_goal_timer += self._env.step_dt
        t = torch.clamp(self.ee_goal_timer / self.ee_goal_traj_time.clamp_min(1.0e-6), 0.0, 1.0)
        current_sphere = torch.lerp(self.ee_start_sphere, self.ee_target_sphere, t.unsqueeze(-1))
        self.ee_goal_b[:] = self._sphere_to_goal_cart(current_sphere)
        if torch.any(self.use_ground_goal):
            ground_ids = torch.nonzero(self.use_ground_goal, as_tuple=False).squeeze(-1)
            self.ee_goal_b[ground_ids] = self._ground_goal_to_base(ground_ids)
        self.ee_goal_quat_b[:] = quat_from_rpy(self.ee_goal_rpy_b)

        resample_ids = torch.nonzero(self.ee_goal_timer >= self.ee_goal_total_time, as_tuple=False).squeeze(-1)
        if resample_ids.numel() > 0:
            self._resample_command(resample_ids.tolist())

    def _update_current_goal(self, env_ids_t: torch.Tensor):
        self.ee_goal_b[env_ids_t] = self._sphere_to_goal_cart(self.ee_target_sphere[env_ids_t])
        ground_ids = env_ids_t[self.use_ground_goal[env_ids_t]]
        if ground_ids.numel() > 0:
            self.ee_goal_b[ground_ids] = self._ground_goal_to_base(ground_ids)
        self.ee_goal_quat_b[env_ids_t] = quat_from_rpy(self.ee_goal_rpy_b[env_ids_t])

    def _sample_ground_goal_w(self, env_ids_t: torch.Tensor) -> torch.Tensor:
        root_pos_w = self._robot.data.root_pos_w[env_ids_t]
        root_quat_w = self._robot.data.root_quat_w[env_ids_t]
        local_xy = torch.zeros((len(env_ids_t), 3), dtype=torch.float32, device=self.device)
        local_xy[:, 0] = torch.empty(len(env_ids_t), device=self.device).uniform_(*self.cfg.ground_goal_x_range)
        local_xy[:, 1] = torch.empty(len(env_ids_t), device=self.device).uniform_(*self.cfg.ground_goal_y_range)
        xy_w = base_to_world(root_pos_w, root_quat_w, local_xy)[:, :2]
        goal_w = torch.zeros((len(env_ids_t), 3), dtype=torch.float32, device=self.device)
        goal_w[:, :2] = xy_w
        goal_w[:, 2] = torch.empty(len(env_ids_t), device=self.device).uniform_(*self.cfg.ground_goal_z_range_w)
        return goal_w

    def _sample_ground_goal_rpy(self, env_ids_t: torch.Tensor):
        goal_b = self._ground_goal_to_base(env_ids_t)
        yaw = torch.atan2(goal_b[:, 1], goal_b[:, 0])
        yaw += torch.empty(len(env_ids_t), device=self.device).uniform_(*self.cfg.ground_goal_yaw_noise)
        self.ee_goal_rpy_b[env_ids_t, 0] = 0.0
        self.ee_goal_rpy_b[env_ids_t, 1] = torch.empty(len(env_ids_t), device=self.device).uniform_(
            *self.cfg.ground_goal_pitch_noise
        )
        self.ee_goal_rpy_b[env_ids_t, 2] = yaw

    def _ground_goal_to_base(self, env_ids_t: torch.Tensor) -> torch.Tensor:
        return world_to_base(
            self._robot.data.root_pos_w[env_ids_t],
            self._robot.data.root_quat_w[env_ids_t],
            self.ground_goal_w[env_ids_t],
        )

    def _sample_target_sphere(self, env_ids_t: torch.Tensor) -> torch.Tensor:
        target = torch.zeros((len(env_ids_t), 3), dtype=torch.float32, device=self.device)
        valid = torch.zeros((len(env_ids_t),), dtype=torch.bool, device=self.device)
        for _ in range(8):
            candidate = torch.zeros_like(target)
            candidate[:, 0] = torch.empty(len(env_ids_t), device=self.device).uniform_(*self.cfg.ranges.radius)
            candidate[:, 1] = torch.empty(len(env_ids_t), device=self.device).uniform_(*self.cfg.ranges.pitch)
            candidate[:, 2] = torch.empty(len(env_ids_t), device=self.device).uniform_(*self.cfg.ranges.yaw)
            cart = self._sphere_to_goal_cart(candidate)
            candidate_valid = (
                (cart[:, 0] >= self.cfg.min_goal_x)
                & (cart[:, 0] <= self.cfg.max_goal_x)
                & (torch.abs(cart[:, 1]) <= self.cfg.max_goal_y_abs)
                & (cart[:, 2] >= self.cfg.min_goal_z)
                & (cart[:, 2] <= self.cfg.max_goal_z)
            )
            if torch.any(valid):
                start_cart = self._sphere_to_goal_cart(self.ee_start_sphere[env_ids_t])
            else:
                start_cart = self._sphere_to_goal_cart(self.ee_start_sphere[env_ids_t])
            collision_free = ~self._collision_check(start_cart, cart)
            candidate_valid &= collision_free
            newly_valid = (~valid) & candidate_valid
            if torch.any(newly_valid):
                target[newly_valid] = candidate[newly_valid]
                valid[newly_valid] = True
            if torch.all(valid):
                break
        if not torch.all(valid):
            target[~valid, 0] = 0.55
            target[~valid, 1] = 0.15
            target[~valid, 2] = 0.0
        return target

    def _sphere_to_goal_cart(self, sphere: torch.Tensor) -> torch.Tensor:
        return sphere_to_cartesian(sphere) + self._goal_center_offset.unsqueeze(0)

    def _collision_check(self, start_cart: torch.Tensor, goal_cart: torch.Tensor) -> torch.Tensor:
        interp = torch.lerp(start_cart.unsqueeze(1), goal_cart.unsqueeze(1), self._collision_check_t)
        inside_collision_box = torch.logical_and(
            torch.all(interp < self._collision_upper_limits.view(1, 1, 3), dim=-1),
            torch.all(interp > self._collision_lower_limits.view(1, 1, 3), dim=-1),
        ).any(dim=1)
        underground = (interp[..., 2] < self.cfg.underground_limit).any(dim=1)
        return inside_collision_box | underground


@configclass
class UniformEeGoalCommandCfg(CommandTermCfg):
    class_type: type = UniformEeGoalCommand

    @configclass
    class SphereCenter:
        x_offset: float = 0.30
        y_offset: float = 0.0
        z_offset: float = 0.20

    @configclass
    class Ranges:
        init_radius: tuple[float, float] = (0.50, 0.70)
        init_pitch: tuple[float, float] = (0.0, math.pi / 8.0)
        init_yaw: tuple[float, float] = (0.0, 0.0)
        radius: tuple[float, float] = (0.35, 0.70)
        pitch: tuple[float, float] = (-0.8, 0.5)
        yaw: tuple[float, float] = (-0.8, 0.8)
        delta_roll: tuple[float, float] = (-0.35, 0.35)
        delta_pitch: tuple[float, float] = (-0.35, 0.35)
        delta_yaw: tuple[float, float] = (-0.35, 0.35)

    resampling_time_range: tuple[float, float] = (3.0, 3.0)
    traj_time_range: tuple[float, float] = (1.0, 3.0)
    hold_time_range: tuple[float, float] = (0.5, 2.0)
    collision_lower_limits: tuple[float, float, float] = (-0.10, -0.22, -0.25)
    collision_upper_limits: tuple[float, float, float] = (0.28, 0.22, 0.18)
    underground_limit: float = -0.10
    num_collision_check_samples: int = 10
    min_goal_x: float = 0.15
    max_goal_x: float = 0.95
    max_goal_y_abs: float = 0.55
    min_goal_z: float = 0.05
    max_goal_z: float = 0.65
    ground_goal_ratio: float = 0.70
    ground_goal_x_range: tuple[float, float] = (0.25, 0.75)
    ground_goal_y_range: tuple[float, float] = (-0.35, 0.35)
    ground_goal_z_range_w: tuple[float, float] = (0.08, 0.14)
    ground_goal_yaw_noise: tuple[float, float] = (-0.35, 0.35)
    ground_goal_pitch_noise: tuple[float, float] = (-0.20, 0.20)
    debug_vis: bool = False
    sphere_center: SphereCenter = SphereCenter()
    ranges: Ranges = Ranges()


@configclass
class VwcWheelVelocityCommandCfg(loco_mdp.UniformThresholdVelocityCommandCfg):
    class_type: type = VwcWheelVelocityCommand

    @configclass
    class Ranges:
        lin_vel_x: tuple[float, float] = (0.0, 0.4)
        lin_vel_y: tuple[float, float] = (0.0, 0.0)
        ang_vel_z: tuple[float, float] = (0.0, 0.0)
        heading: tuple[float, float] = (-math.pi, math.pi)

    curriculum_total_steps: int = 45000 * 24
    stage_a_until: float = 0.10
    stage_b_until: float = 0.30
    lin_vel_x_clip: float = 0.2
    ang_vel_z_clip: float = 0.5
    stage_a_ranges: Ranges = Ranges(
        lin_vel_x=(0.0, 0.4),
        lin_vel_y=(0.0, 0.0),
        ang_vel_z=(0.0, 0.0),
        heading=(-math.pi, math.pi),
    )
    stage_b_ranges: Ranges = Ranges(
        lin_vel_x=(-0.4, 0.6),
        lin_vel_y=(0.0, 0.0),
        ang_vel_z=(-0.4, 0.4),
        heading=(-math.pi, math.pi),
    )
    stage_c_ranges: Ranges = Ranges(
        lin_vel_x=(-0.8, 0.8),
        lin_vel_y=(0.0, 0.0),
        ang_vel_z=(-1.0, 1.0),
        heading=(-math.pi, math.pi),
    )
    ranges: Ranges = Ranges(
        lin_vel_x=(-0.8, 0.8),
        lin_vel_y=(0.0, 0.0),
        ang_vel_z=(-1.0, 1.0),
        heading=(-math.pi, math.pi),
    )


@configclass
class CommandsCfg:
    """Stage1 commands: base velocity plus EE goal."""

    base_velocity = VwcWheelVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(3.0, 3.0),
        rel_standing_envs=0.05,
        rel_heading_envs=0.0,
        heading_command=False,
        debug_vis=False,
        ranges=VwcWheelVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.8, 0.8),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(-1.0, 1.0),
            heading=(-math.pi, math.pi),
        ),
    )

    ee_goal = UniformEeGoalCommandCfg()
