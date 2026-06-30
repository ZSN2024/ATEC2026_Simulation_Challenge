import torch
from isaaclab.utils.math import subtract_frame_transforms

from atec_rl_lab.utils.cartesian_controller import CartesianController


class CartesianArmAction:
    """Thin wrapper around the shared cartesian controller for planned Stage1 IK use."""

    def __init__(self, *args, **kwargs):
        self.robot = kwargs.get("robot")
        self.num_envs = kwargs.get("num_envs")
        self.device = kwargs.get("device")
        pose_kwargs = dict(kwargs)
        position_kwargs = dict(kwargs)
        pose_kwargs["command_type"] = "pose"
        position_kwargs["command_type"] = "position"
        self.pose_controller = CartesianController(*args, **pose_kwargs)
        self.position_controller = CartesianController(*args, **position_kwargs)
        self.controller = self.pose_controller
        self.direct_reach_threshold = 0.20
        self.far_target_step = 0.10

    def reset(self, env_ids=None):
        self.pose_controller.reset(env_ids=env_ids)
        self.position_controller.reset(env_ids=env_ids)

    def compute_base(self, ee_pos_b, ee_quat_b=None):
        planned_pos_b, direct_mask = self._plan_position_target(ee_pos_b)
        arm_target = self.position_controller.compute_base(planned_pos_b)
        if ee_quat_b is not None and torch.any(direct_mask):
            direct_arm_target = self.pose_controller.compute_base(ee_pos_b, ee_quat_b=ee_quat_b)
            arm_target[direct_mask] = direct_arm_target[direct_mask]
        return arm_target

    def _current_ee_pos_b(self):
        root_pose_w = self.robot.data.root_pose_w
        ee_pose_w = self.robot.data.body_pose_w[:, self.pose_controller.ee_idx]
        ee_pos_b, _ = subtract_frame_transforms(
            root_pose_w[:, :3],
            root_pose_w[:, 3:],
            ee_pose_w[:, :3],
            ee_pose_w[:, 3:],
        )
        return ee_pos_b

    def _plan_position_target(self, ee_pos_b):
        current_ee_pos_b = self._current_ee_pos_b()
        delta = ee_pos_b - current_ee_pos_b
        dist = torch.linalg.norm(delta, dim=-1, keepdim=True).clamp_min(1.0e-6)
        direct_mask = (dist.squeeze(-1) <= self.direct_reach_threshold)
        step = torch.minimum(dist, torch.full_like(dist, self.far_target_step))
        planned_pos_b = current_ee_pos_b + delta / dist * step
        planned_pos_b[direct_mask] = ee_pos_b[direct_mask]
        return planned_pos_b, direct_mask
