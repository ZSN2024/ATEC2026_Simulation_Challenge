import torch


def next_command(num_envs: int, device: str = "cuda") -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    base_velocity = torch.tensor([[0.3, 0.0, 0.0]], device=device, dtype=torch.float32).repeat(num_envs, 1)
    ee_goal = torch.tensor([[0.55, 0.0, 0.30]], device=device, dtype=torch.float32).repeat(num_envs, 1)
    ee_goal_rpy = torch.tensor([[0.0, 0.0, 0.0]], device=device, dtype=torch.float32).repeat(num_envs, 1)
    return base_velocity, ee_goal, ee_goal_rpy
