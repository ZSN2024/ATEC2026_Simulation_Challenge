import torch


ARM_ACTION_SCALE = 0.5
ARM_TARGET_REL = torch.tensor(
    [0.0, 0.5, -1.0, 0.0, 1.5, 0.0, 0.035, -0.035],
    dtype=torch.float32,
)


def _scripted_arm_action_from_proprio(proprio: torch.Tensor, official_action_dim: int) -> torch.Tensor:
    if official_action_dim != 24:
        raise ValueError(f"TaskB Stage1 demo currently supports B2wPiper action dim 24, got {official_action_dim}.")
    idx = 12  # skip base lin/ang velocity, command, projected gravity
    joint_pos = proprio[:, idx:idx + official_action_dim]
    arm_joint_pos = joint_pos[:, 16:24]
    target = ARM_TARGET_REL.to(device=proprio.device, dtype=proprio.dtype).view(1, -1)
    target = target.repeat(proprio.shape[0], 1)
    return torch.clamp((target - arm_joint_pos) / ARM_ACTION_SCALE, -1.0, 1.0)


def adapt_action(
    policy_action: torch.Tensor,
    official_action_dim: int,
    policy_action_dim: int | None = None,
    proprio: torch.Tensor | None = None,
) -> torch.Tensor:
    if policy_action.ndim == 1:
        policy_action = policy_action.unsqueeze(0)
    if policy_action_dim is None:
        policy_action_dim = int(policy_action.shape[1])
    if official_action_dim != 24 or policy_action_dim != 16:
        raise ValueError(
            f"TaskB Stage1 demo expects B2wPiper dims official=24, policy=16; "
            f"got official={official_action_dim}, policy={policy_action_dim}."
        )
    action = torch.zeros(
        (policy_action.shape[0], official_action_dim),
        device=policy_action.device,
        dtype=policy_action.dtype,
    )
    action[:, :policy_action_dim] = policy_action[:, :policy_action_dim]
    if proprio is not None:
        action[:, 16:24] = _scripted_arm_action_from_proprio(
            proprio.to(device=policy_action.device, dtype=policy_action.dtype),
            official_action_dim,
        )
    return action
