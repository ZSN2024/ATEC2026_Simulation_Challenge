import os
from typing import Any

import torch


class _Stage1RawActor(torch.nn.Module):
    def __init__(self, obs_dim: int, action_dim: int):
        super().__init__()
        from rsl_rl.models.mlp_model import MLPModel

        self.model = MLPModel(
            obs={"policy": torch.zeros(1, obs_dim, device="cuda")},
            obs_groups={"actor": ["policy"]},
            obs_set="actor",
            output_dim=action_dim,
            hidden_dims=[128, 128],
            activation="elu",
            obs_normalization=False,
            distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0},
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.model({"policy": obs})


class _Stage1BaseRuntime:
    HISTORY_LEN = 10
    SINGLE_OBS_DIM = 79
    POLICY_OBS_DIM = SINGLE_OBS_DIM * (HISTORY_LEN + 1)
    POLICY_ACTION_DIM = 16
    OFFICIAL_ACTION_DIM = 24
    JOINT_VEL_SCALE = 0.05
    OBS_CLIP = 100.0
    ARM_SCALE = 0.5

    ARM_HOME_REL = torch.tensor(
        [0.0, 0.7, -1.2, 0.0, 1.2, 0.0, 0.035, -0.035],
        dtype=torch.float32,
    )

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.policy = self._load_policy()
        self.history: torch.Tensor | None = None
        self.step_count = 0

    def _candidate_policy_paths(self) -> list[str]:
        demo_dir = os.path.dirname(os.path.abspath(__file__))
        return [
            os.path.join(demo_dir, "taskb_stage1_8400.pt"),
            os.path.join(demo_dir, "taskb_stage1", "policy.pt"),
            os.path.join(
                os.path.dirname(demo_dir),
                "temp",
                "logs",
                "rsl_rl",
                "task_b_vwc_stage1",
                "2026-06-30_19-07-12",
                "model_8400.pt",
            ),
            os.path.join(
                os.path.dirname(demo_dir),
                "logs",
                "rsl_rl",
                "task_b_vwc_stage1",
                "2026-06-30_19-07-12",
                "model_8400.pt",
            ),
        ]

    def _load_policy(self):
        for path in self._candidate_policy_paths():
            if not os.path.exists(path):
                continue
            try:
                policy = torch.jit.load(path, map_location=self.device)
                policy.eval()
                print(f"[taskb-eazy] loaded Stage1 JIT policy: {path}")
                return policy
            except Exception:
                actor = self._load_raw_checkpoint(path)
                actor.eval()
                actor.to(self.device)
                print(f"[taskb-eazy] loaded Stage1 raw checkpoint: {path}")
                return actor
        raise FileNotFoundError(
            "No Stage1 policy/checkpoint found. Expected one of: "
            + ", ".join(self._candidate_policy_paths())
        )

    def _load_raw_checkpoint(self, path: str) -> torch.nn.Module:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        if "actor_state_dict" in ckpt:
            actor_sd = ckpt["actor_state_dict"]
        elif "model_state_dict" in ckpt:
            actor_sd = {}
            for key, value in ckpt["model_state_dict"].items():
                if key.startswith("actor."):
                    actor_sd["mlp." + key[len("actor.") :]] = value
                elif key == "std":
                    actor_sd["distribution.std_param"] = value
        else:
            raise ValueError(f"Unknown Stage1 checkpoint format: {list(ckpt.keys())}")

        actor = _Stage1RawActor(self.POLICY_OBS_DIM, self.POLICY_ACTION_DIM)
        missing, unexpected = actor.model.load_state_dict(actor_sd, strict=False)
        if unexpected:
            print(f"[taskb-eazy] unexpected checkpoint keys: {unexpected[:8]}")
        if missing:
            print(f"[taskb-eazy] missing checkpoint keys: {missing[:8]}")
        return actor

    def reset(self):
        self.history = None
        self.step_count = 0

    def _commands(self, num_envs: int, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cycle = self.step_count % 1200
        if self.step_count < 50:
            cmd = [0.0, 0.0, 0.0]
        elif cycle < 400:
            cmd = [0.20, 0.0, 0.0]
        elif cycle < 600:
            cmd = [0.05, 0.0, 0.35]
        elif cycle < 1000:
            cmd = [0.20, 0.0, 0.0]
        else:
            cmd = [0.05, 0.0, -0.35]

        base_cmd = torch.tensor(cmd, device=self.device, dtype=dtype).view(1, 3).repeat(num_envs, 1)
        ee_goal = torch.tensor([0.35, 0.0, 0.35], device=self.device, dtype=dtype).view(1, 3).repeat(num_envs, 1)
        ee_rpy = torch.zeros(num_envs, 3, device=self.device, dtype=dtype)
        return base_cmd, ee_goal, ee_rpy

    def _policy_obs(
        self,
        proprio: torch.Tensor,
        base_cmd: torch.Tensor,
        ee_goal: torch.Tensor,
        ee_rpy: torch.Tensor,
    ) -> torch.Tensor:
        official_action_dim = (int(proprio.shape[-1]) - 12) // 3
        if official_action_dim != self.OFFICIAL_ACTION_DIM:
            raise ValueError(f"TaskB eazy supports B2wPiper action_dim=24, got {official_action_dim}.")

        idx = 0
        idx += 3
        base_ang_vel = proprio[:, idx : idx + 3]
        idx += 3
        idx += 3
        projected_gravity = proprio[:, idx : idx + 3]
        idx += 3
        joint_pos = proprio[:, idx : idx + official_action_dim]
        idx += official_action_dim
        joint_vel = proprio[:, idx : idx + official_action_dim] * self.JOINT_VEL_SCALE
        idx += official_action_dim
        last_policy_action = proprio[:, idx : idx + self.POLICY_ACTION_DIM]

        current = torch.cat(
            [
                base_ang_vel,
                projected_gravity,
                base_cmd,
                ee_goal,
                ee_rpy,
                joint_pos,
                joint_vel,
                last_policy_action,
            ],
            dim=-1,
        )
        current = torch.nan_to_num(
            current,
            nan=0.0,
            posinf=self.OBS_CLIP,
            neginf=-self.OBS_CLIP,
        ).clamp(-self.OBS_CLIP, self.OBS_CLIP)

        if int(current.shape[-1]) != self.SINGLE_OBS_DIM:
            raise ValueError(f"Stage1 current obs dim mismatch: {current.shape[-1]} != {self.SINGLE_OBS_DIM}")

        num_envs = proprio.shape[0]
        if self.history is None or self.history.shape[0] != num_envs or self.history.device != proprio.device:
            self.history = torch.zeros(
                num_envs,
                self.HISTORY_LEN,
                self.SINGLE_OBS_DIM,
                device=proprio.device,
                dtype=proprio.dtype,
            )

        history_flat = self.history.reshape(num_envs, -1)
        obs = torch.cat([current, history_flat], dim=-1)
        self.history = torch.cat([self.history[:, 1:], current.unsqueeze(1)], dim=1)
        return obs

    def _fixed_arm_action(self, proprio: torch.Tensor) -> torch.Tensor:
        # Base-only smoke: leave the manipulator to the official default action target.
        return torch.zeros(proprio.shape[0], 8, device=proprio.device, dtype=proprio.dtype)

    def step(self, obs: dict) -> torch.Tensor:
        proprio = obs["proprio"].to(self.device)
        self.step_count += 1

        base_cmd, ee_goal, ee_rpy = self._commands(proprio.shape[0], proprio.dtype)
        policy_obs = self._policy_obs(proprio, base_cmd, ee_goal, ee_rpy)
        with torch.inference_mode():
            policy_action = self.policy(policy_obs)
        if not isinstance(policy_action, torch.Tensor):
            policy_action = torch.as_tensor(policy_action, device=self.device, dtype=proprio.dtype)
        if policy_action.ndim == 1:
            policy_action = policy_action.unsqueeze(0)
        policy_action = policy_action.to(device=self.device, dtype=proprio.dtype)
        policy_action = torch.clamp(policy_action, -1.0, 1.0)

        action = torch.zeros(
            proprio.shape[0],
            self.OFFICIAL_ACTION_DIM,
            device=self.device,
            dtype=proprio.dtype,
        )
        action[:, : self.POLICY_ACTION_DIM] = policy_action[:, : self.POLICY_ACTION_DIM]
        action[:, 16:24] = self._fixed_arm_action(proprio)

        if self.step_count <= 5 or self.step_count % 200 == 0:
            print(
                "[taskb-eazy] "
                f"step={self.step_count} base_cmd={base_cmd[0].detach().cpu().tolist()} "
                f"policy_min={policy_action.min().item():.3f} policy_max={policy_action.max().item():.3f} "
                f"arm_min={action[:, 16:24].min().item():.3f} arm_max={action[:, 16:24].max().item():.3f}",
                flush=True,
            )
        return action


class AlgSolution:
    """Task B eazy: Stage1 leg/wheel policy, fixed arm, no IK."""

    def __init__(self):
        self.device = "cuda"
        self._runtime: _Stage1BaseRuntime | None = None
        self._initialized = False

    def get_action_spec(self) -> dict[str, dict[str, Any]] | None:
        return {}

    def reset(self, **kwargs):
        if self._runtime is not None:
            self._runtime.reset()
        self._initialized = False

    def _ensure_runtime(self, obs: dict):
        if self._runtime is None:
            self._runtime = _Stage1BaseRuntime(device=self.device)
        proprio = obs["proprio"].to(self.device)
        action_dim = (int(proprio.shape[-1]) - 12) // 3
        if action_dim != _Stage1BaseRuntime.OFFICIAL_ACTION_DIM:
            raise ValueError(f"taskb-eazy expects B2wPiper action_dim=24, got {action_dim}.")
        self._initialized = True

    def predicts(self, obs: dict, current_score: float) -> dict:
        self._ensure_runtime(obs)
        action = self._runtime.step(obs)
        return {"action": action.cpu().tolist(), "giveup": False}
