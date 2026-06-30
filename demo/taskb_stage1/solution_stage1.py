import torch

from .action_adapter import adapt_action
from .command_scheduler import next_command
from .obs_adapter import adapt_obs, reset_history
from .policy_loader import load_policy, load_policy_meta


class AlgSolution:
    def __init__(self):
        self.device = "cuda"
        self.policy = load_policy(device=self.device)
        self.policy_meta = load_policy_meta()
        self._initialized = False
        self.official_action_dim = 0
        self.policy_action_dim = int(self.policy_meta["action_dim"])
        reset_history()

    def predicts(self, obs, total_reward):
        proprio = obs["proprio"].to(self.device)
        if not self._initialized:
            self.official_action_dim = (int(proprio.shape[-1]) - 12) // 3
            self._initialized = True

        velocity_command, ee_goal_command, ee_goal_orientation_command = next_command(
            proprio.shape[0], device=self.device
        )
        policy_obs = adapt_obs(
            {"proprio": proprio},
            velocity_command,
            ee_goal_command,
            ee_goal_orientation_command,
            expected_policy_obs_dim=int(self.policy_meta["policy_obs_dim"]),
            policy_action_dim=self.policy_action_dim,
        )
        with torch.inference_mode():
            policy_action = self.policy(policy_obs)
        action = adapt_action(
            policy_action,
            official_action_dim=self.official_action_dim,
            policy_action_dim=self.policy_action_dim,
            proprio=proprio,
        )
        return {"giveup": False, "action": action.cpu().tolist()}
