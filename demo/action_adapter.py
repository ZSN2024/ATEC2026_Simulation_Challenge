"""Action adapter: assemble full 24D action from PPO policy output + IK arm + gripper.

Replaces the old scripted arm action with Cartesian IK results.
Output format: [leg(12) | wheel(4) | arm_ik(6) | gripper(2)]
"""

import torch


def adapt_action(
    policy_action: torch.Tensor,
    arm_joint_pos: torch.Tensor,
    gripper_pos: torch.Tensor,
    default_arm_gripper: torch.Tensor,
    official_action_dim: int = 24,
    arm_scale: float = 0.5,
) -> torch.Tensor:
    """Assemble full official action tensor.

    Policy action (leg + wheel) is already in the environment's expected
    format.  Arm + gripper are computed via Cartesian IK and scaled to
    match the env action spec:  action = (pos - default) / arm_scale.

    Args:
        policy_action: (1, 16) or (16,) — leg(12) + wheel(4) from PPO.
        arm_joint_pos: (1, 6) or (6,) — arm joint positions from Cartesian IK.
        gripper_pos: (1, 2) or (2,) — gripper joint positions (0=open, 0.3=close).
        default_arm_gripper: (8,) default arm(6)+gripper(2) from default_joint_pos.
        official_action_dim: full action dim (24 for B2wPiper).
        arm_scale: action scale for the arm control group (default 0.5).

    Returns:
        (1, official_action_dim) full action tensor.
    """
    if policy_action.ndim == 1:
        policy_action = policy_action.unsqueeze(0)
    if arm_joint_pos.ndim == 1:
        arm_joint_pos = arm_joint_pos.unsqueeze(0)
    if gripper_pos.ndim == 1:
        gripper_pos = gripper_pos.unsqueeze(0)
    if default_arm_gripper.ndim == 1:
        default_arm_gripper = default_arm_gripper.unsqueeze(0)

    num_envs = policy_action.shape[0]
    device = policy_action.device
    dtype = policy_action.dtype

    action = torch.zeros(num_envs, official_action_dim, device=device, dtype=dtype)

    # Leg(12) + Wheel(4) from PPO — already in env format
    action[:, :16] = policy_action[:, :16]

    # Arm(6) + Gripper(2) from IK — scale to env format
    arm_gripper = torch.cat([arm_joint_pos, gripper_pos], dim=-1)
    def_ag = default_arm_gripper.to(device=device, dtype=dtype)
    action[:, 16:24] = (arm_gripper - def_ag) / arm_scale

    return action
