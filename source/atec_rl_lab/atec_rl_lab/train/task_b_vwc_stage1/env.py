import numpy as np
import torch
from gymnasium import spaces

from atec_rl_lab.tasks.task_base.envs_base import BaseRLEnv

from .cartesian_arm_action import CartesianArmAction


class TaskBVwcStage1Env(BaseRLEnv):
    """Stage1 env wrapper that exposes leg/wheel policy actions and applies arm IK separately."""

    history_len = 10

    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg=cfg, render_mode=render_mode, **kwargs)

        self._robot = self.scene["robot"]
        robot_cfg = self.cfg.scene.robot

        self._leg_joint_names = list(cfg.actions.joint_leg.joint_names) if cfg.actions.joint_leg is not None else []
        self._wheel_joint_names = list(cfg.actions.joint_wheel.joint_names) if cfg.actions.joint_wheel is not None else []
        self._arm_joint_names = list(getattr(robot_cfg, "arm_joint_names", []))

        self._policy_action_dim = len(self._leg_joint_names) + len(self._wheel_joint_names)
        self._arm_action_scale = 0.5
        self._single_policy_obs_dim = (
            3  # base_ang_vel
            + 3  # projected_gravity
            + 3  # base velocity command
            + 3  # ee goal
            + 3  # ee goal orientation rpy
            + len(robot_cfg.joint_names)  # joint_pos
            + len(robot_cfg.joint_names)  # joint_vel
            + self._policy_action_dim  # last policy action
        )
        self._policy_history = torch.zeros(
            self.num_envs,
            self.history_len,
            self._single_policy_obs_dim,
            device=self.device,
        )

        self._arm_joint_ids, _ = self._robot.find_joints(self._arm_joint_names)
        self._arm_controller = CartesianArmAction(
            robot=self._robot,
            ee_body_name="gripper_base",
            arm_joint_names=self._arm_joint_names,
            num_envs=self.num_envs,
            device=self.device,
            command_type="pose",
        )

        # Expose the policy-facing action space: leg + wheel only.
        self.single_action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self._policy_action_dim,),
            dtype=np.float32,
        )
        self.action_space = self.single_action_space

    def reset(self, seed: int | None = None, options: dict | None = None):
        self._policy_history.zero_()
        obs, info = super().reset(seed=seed, options=options)
        self._arm_controller.reset()
        return obs, info

    def update_policy_history(self, current_policy_obs: torch.Tensor) -> torch.Tensor:
        """Return VWC-style current observation plus 10 frames of previous policy observations."""
        history_flat = self._policy_history.reshape(self.num_envs, -1)
        self._policy_history = torch.cat(
            [self._policy_history[:, 1:], current_policy_obs.unsqueeze(1)],
            dim=1,
        )
        return torch.cat([current_policy_obs, history_flat], dim=-1)

    def reset_policy_history(self, env_ids: torch.Tensor):
        if env_ids.numel() > 0:
            self._policy_history[env_ids] = 0.0

    def _compute_full_action(self, action: torch.Tensor) -> torch.Tensor:
        if action.ndim == 1:
            action = action.unsqueeze(0)
        action_dim = int(action.shape[-1])
        if action_dim != self._policy_action_dim:
            raise ValueError(
                f"Expected policy action dim {self._policy_action_dim}, got {action_dim}."
            )

        full_action = action
        ee_goal_b = self.command_manager.get_command("ee_goal")
        ee_goal_quat_b = getattr(self.command_manager.get_term("ee_goal"), "command_quat")
        arm_joint_target = self._arm_controller.compute_base(ee_goal_b, ee_quat_b=ee_goal_quat_b)
        return full_action, arm_joint_target

    def step(self, action: torch.Tensor):
        full_action, arm_joint_target = self._compute_full_action(action.to(self.device))
        self._robot.set_joint_position_target(arm_joint_target, joint_ids=self._arm_joint_ids)
        obs, reward, terminated, truncated, info = super().step(full_action)
        done = torch.logical_or(terminated, truncated)
        if torch.any(done):
            env_ids = torch.nonzero(done, as_tuple=False).squeeze(-1)
            self._arm_controller.reset(env_ids=env_ids)
            self.reset_policy_history(env_ids)
        return obs, reward, terminated, truncated, info
