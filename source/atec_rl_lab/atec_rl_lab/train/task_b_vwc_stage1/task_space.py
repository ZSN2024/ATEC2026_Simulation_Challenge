import torch
import isaaclab.utils.math as math_utils
from isaaclab.utils.math import quat_rotate, quat_rotate_inverse


def world_to_base(
    root_pos_w: torch.Tensor,
    root_quat_w: torch.Tensor,
    world_pos_w: torch.Tensor,
) -> torch.Tensor:
    return quat_rotate_inverse(root_quat_w, world_pos_w - root_pos_w)


def base_to_world(
    root_pos_w: torch.Tensor,
    root_quat_w: torch.Tensor,
    base_pos: torch.Tensor,
) -> torch.Tensor:
    return root_pos_w + quat_rotate(root_quat_w, base_pos)


def sphere_to_cartesian(sphere: torch.Tensor) -> torch.Tensor:
    radius = sphere[:, 0]
    pitch = sphere[:, 1]
    yaw = sphere[:, 2]
    cos_pitch = torch.cos(pitch)
    return torch.stack(
        [
            radius * cos_pitch * torch.cos(yaw),
            radius * cos_pitch * torch.sin(yaw),
            radius * torch.sin(pitch),
        ],
        dim=-1,
    )


def cartesian_to_sphere(cart: torch.Tensor) -> torch.Tensor:
    radius = torch.linalg.norm(cart, dim=-1).clamp_min(1.0e-6)
    pitch = torch.asin((cart[:, 2] / radius).clamp(-1.0, 1.0))
    yaw = torch.atan2(cart[:, 1], cart[:, 0])
    return torch.stack([radius, pitch, yaw], dim=-1)


def quat_from_rpy(rpy: torch.Tensor) -> torch.Tensor:
    return math_utils.quat_from_euler_xyz(rpy[:, 0], rpy[:, 1], rpy[:, 2])


def ee_orientation_error_rpy(goal_quat: torch.Tensor, current_quat: torch.Tensor) -> torch.Tensor:
    quat_error = math_utils.quat_mul(goal_quat, math_utils.quat_conjugate(current_quat))
    return math_utils.axis_angle_from_quat(quat_error)
