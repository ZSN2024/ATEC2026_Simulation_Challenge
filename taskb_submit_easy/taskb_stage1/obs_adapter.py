import torch


HISTORY_LEN = 10
SINGLE_POLICY_OBS_DIM = 79
JOINT_VEL_SCALE = 0.05
OBS_CLIP = 100.0

_history: torch.Tensor | None = None


def reset_history():
    global _history
    _history = None


def _current_policy_obs(
    obs: dict,
    velocity_command: torch.Tensor,
    ee_goal_command: torch.Tensor,
    ee_goal_orientation_command: torch.Tensor,
    policy_action_dim: int,
) -> torch.Tensor:
    proprio = obs["proprio"]
    official_action_dim = (int(proprio.shape[-1]) - 12) // 3
    if official_action_dim != 24:
        raise ValueError(f"TaskB Stage1 demo currently supports B2wPiper action dim 24, got {official_action_dim}.")
    if policy_action_dim != 16:
        raise ValueError(f"TaskB Stage1 B2wPiper policy action dim must be 16, got {policy_action_dim}.")

    idx = 0
    idx += 3  # base_lin_vel, actor does not use it.
    base_ang_vel = proprio[:, idx:idx + 3]
    idx += 3
    idx += 3  # official velocity command, replaced by the local scheduler command.
    projected_gravity = proprio[:, idx:idx + 3]
    idx += 3
    joint_pos = proprio[:, idx:idx + official_action_dim]
    idx += official_action_dim
    joint_vel = proprio[:, idx:idx + official_action_dim] * JOINT_VEL_SCALE
    idx += official_action_dim
    last_action = proprio[:, idx:idx + policy_action_dim]

    current = torch.cat(
        [
            base_ang_vel,
            projected_gravity,
            velocity_command.to(device=proprio.device, dtype=proprio.dtype),
            ee_goal_command.to(device=proprio.device, dtype=proprio.dtype),
            ee_goal_orientation_command.to(device=proprio.device, dtype=proprio.dtype),
            joint_pos,
            joint_vel,
            last_action,
        ],
        dim=-1,
    )
    return torch.nan_to_num(current, nan=0.0, posinf=OBS_CLIP, neginf=-OBS_CLIP).clamp(-OBS_CLIP, OBS_CLIP)


def adapt_obs(
    obs: dict,
    velocity_command: torch.Tensor,
    ee_goal_command: torch.Tensor,
    ee_goal_orientation_command: torch.Tensor,
    expected_policy_obs_dim: int | None = None,
    policy_action_dim: int | None = None,
) -> torch.Tensor:
    global _history
    if policy_action_dim is None:
        policy_action_dim = 16

    current = _current_policy_obs(
        obs,
        velocity_command,
        ee_goal_command,
        ee_goal_orientation_command,
        policy_action_dim,
    )
    if int(current.shape[-1]) != SINGLE_POLICY_OBS_DIM:
        raise ValueError(f"Current policy obs dim mismatch: expected {SINGLE_POLICY_OBS_DIM}, got {current.shape[-1]}.")

    num_envs = current.shape[0]
    if _history is None or _history.shape[0] != num_envs or _history.device != current.device:
        _history = torch.zeros(
            num_envs,
            HISTORY_LEN,
            SINGLE_POLICY_OBS_DIM,
            device=current.device,
            dtype=current.dtype,
        )

    history_flat = _history.reshape(num_envs, -1)
    policy_obs = torch.cat([current, history_flat], dim=-1)
    _history = torch.cat([_history[:, 1:], current.unsqueeze(1)], dim=1)

    if expected_policy_obs_dim is not None and int(policy_obs.shape[-1]) != int(expected_policy_obs_dim):
        raise ValueError(
            f"Policy observation dim mismatch: expected {expected_policy_obs_dim}, got {int(policy_obs.shape[-1])}."
        )
    return policy_obs
