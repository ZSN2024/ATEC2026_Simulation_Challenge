import torch
import numpy as np

from atec_rl_lab.utils.cartesian_controller import CartesianController
from demo.constants import (
    LEG_LOCK_POS,
    LEG_DIM, WHEEL_DIM, ARM_DIM, GRIPPER_DIM,
    LEG_START, WHEEL_START, ARM_START, GRIPPER_START,
    TRACK_WIDTH, WHEEL_RADIUS, MAX_LIN_VEL, MAX_ANG_VEL, GRASP_RANGE,
)


class WheelController:
    """Differential-drive controller for 4 wheel joints."""

    def __init__(self, device="cuda"):
        self.device = device
        self.track_width = TRACK_WIDTH
        self.wheel_radius = WHEEL_RADIUS
        self.max_vel = MAX_LIN_VEL
        self.max_omega = MAX_ANG_VEL
        self.grasp_range = GRASP_RANGE

    def zero(self) -> torch.Tensor:
        return torch.zeros(WHEEL_DIM, device=self.device)

    def compute(self, target_pos_b: torch.Tensor) -> torch.Tensor:
        """
        target_pos_b: (3,) target position in base frame (x-forward, y-left, z-up)
        Returns: (4,) wheel velocity tensor [FR, FL, RR, RL]
        Align to target first if angle is large, then drive forward.
        """
        dx, dy = target_pos_b[0].item(), target_pos_b[1].item()
        dist = np.sqrt(dx * dx + dy * dy)

        if dist < self.grasp_range:
            return self.zero()

        angle_to_target = np.arctan2(dy, dx)
        angle_deg = abs(np.degrees(angle_to_target))

        if angle_deg > 5:
            omega = np.clip(angle_to_target * 3.0, -self.max_omega, self.max_omega)
            v = 0.0
        else:
            omega = np.clip(angle_to_target * 2.0, -self.max_omega, self.max_omega)
            v = min(dist * 0.5, self.max_vel)

        vel_right = (v + omega * self.track_width / 2) / self.wheel_radius
        vel_left = (v - omega * self.track_width / 2) / self.wheel_radius

        return torch.tensor([vel_right, vel_left, vel_right, vel_left],
                            device=self.device, dtype=torch.float32)

    def compute_toward_bin(self, dx: float, dy: float) -> torch.Tensor:
        """Drive toward bin center given (dx, dy) in approximate world frame."""
        dist = np.sqrt(dx * dx + dy * dy)
        if dist < 1.5:
            return self.zero()
        angle = np.arctan2(dy, dx)
        v = min(dist * 0.5, self.max_vel)
        omega = np.clip(angle * 2.5, -self.max_omega, self.max_omega)
        vr = (v + omega * self.track_width / 2) / self.wheel_radius
        vl = (v - omega * self.track_width / 2) / self.wheel_radius
        return torch.tensor([vr, vl, vr, vl], device=self.device, dtype=torch.float32)

    def rotate(self, omega: float = 0.5) -> torch.Tensor:
        """Pure rotation."""
        vr = omega * self.track_width / 2 / self.wheel_radius
        vl = -vr
        return torch.tensor([vr, vl, vr, vl], device=self.device, dtype=torch.float32)


class GraspController:
    """IK-based arm + gripper controller using CartesianController."""

    ARM_JOINT_NAMES = [
        "arm_joint1", "arm_joint2", "arm_joint3",
        "arm_joint4", "arm_joint5", "arm_joint6",
    ]
    GRIPPER_JOINT_NAMES = ["arm_joint7", "arm_joint8"]
    EE_BODY_NAME = "gripper_base"

    def __init__(self, robot, device="cuda", num_envs=1, command_type="position"):
        self.robot = robot
        self.device = device
        self.num_envs = num_envs

        self.arm_ids, _ = robot.find_joints(self.ARM_JOINT_NAMES)
        self.gripper_ids, _ = robot.find_joints(self.GRIPPER_JOINT_NAMES)

        self.cart_ctrl = CartesianController(
            robot=robot,
            ee_body_name=self.EE_BODY_NAME,
            arm_joint_names=self.ARM_JOINT_NAMES,
            num_envs=num_envs,
            device=device,
            command_type=command_type,
            max_joint_delta=0.05,
        )

        self.rest_arm_pos = torch.zeros(num_envs, ARM_DIM, device=device)
        self.gripper_open = torch.zeros(num_envs, GRIPPER_DIM, device=device)
        self.gripper_close = torch.full((num_envs, GRIPPER_DIM), 0.3, device=device)

        print(f"[GraspController] arm_ids={self.arm_ids}, gripper_ids={self.gripper_ids}")
        print(f"[GraspController] arm joints: {self.ARM_JOINT_NAMES}")
        print(f"[GraspController] gripper joints: {self.GRIPPER_JOINT_NAMES}")
        print(f"[GraspController] EE body: {self.EE_BODY_NAME}, fixed_base={robot.is_fixed_base}")

    def reset(self):
        self.cart_ctrl.reset()

    def ee_pos_w(self) -> torch.Tensor:
        return self.cart_ctrl.ee_pos_w

    def ee_pose_w(self) -> torch.Tensor:
        """Return (num_envs, 7) [x,y,z, qw,qx,qy,qz] in world frame."""
        return self.robot.data.body_pose_w[:, self.cart_ctrl.ee_idx]

    def compute_rest(self) -> torch.Tensor:
        """Return (num_envs, 8) arm + gripper position for rest pose (open)."""
        arm_pos = self.robot.data.default_joint_pos[:, self.arm_ids]
        return torch.cat([arm_pos, self.gripper_open], dim=-1)

    def compute_grab(self, target_pos_b: torch.Tensor,
                     close_gripper: bool = True) -> torch.Tensor:
        """
        target_pos_b: (num_envs, 3) target EE pos in base frame.
        Returns: (num_envs, 8) arm(6) + gripper(2) joint positions.
        """
        if target_pos_b is None:
            gripper = self.gripper_open
            arm = self.robot.data.default_joint_pos[:, self.arm_ids]
        else:
            arm = self.cart_ctrl.compute_base(target_pos_b)
            gripper = self.gripper_close if close_gripper else self.gripper_open

        return torch.cat([arm, gripper], dim=-1)

    def compute_open(self) -> torch.Tensor:
        """Hold arm still, open gripper."""
        arm = self.robot.data.joint_pos[:, self.arm_ids]
        return torch.cat([arm, self.gripper_open], dim=-1)


class LegPDController:
    """Direct position controller for leg joints.

    Returns offset from default joint positions.
    """

    def __init__(self, device="cuda", kp=150.0, kd=10.0):
        self.device = device
        self.offset = torch.zeros(LEG_DIM, device=device)

    def set_offset(self, offset: torch.Tensor):
        self.offset = offset.clone().to(self.device)

    def compute(self) -> torch.Tensor:
        return self.offset.to(self.device)
