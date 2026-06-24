import os
import torch
from typing import Any


class AlgSolution:
    """RL-based solution for ATEC Task A (Off-road Navigation).

    Supports two modes (auto-detected):
      1. Raw checkpoint (model_*.pt) — preferred, loads directly
      2. JIT policy (policy.pt) — fallback

    Priority: model_200.pt > model_100.pt > policy.pt
    """

    ACTION_SCALE = 0.5
    _CKPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                              "logs", "rsl_rl", "unitree_b2_rough")

    _ROBOT_DIM_TABLE: dict[int, dict[str, int]] = {
        20: {"leg": 12, "arm": 8, "wheel": 0},   # B2Piper
        24: {"leg": 12, "arm": 8, "wheel": 4},   # B2wPiper
        33: {"leg": 33, "arm": 0, "wheel": 0},   # G1
        16: {"leg": 6, "arm": 8, "wheel": 2},    # Tron1Piper
        18: {"leg": 10, "arm": 8, "wheel": 0},   # Tron2A Legged
    }

    def __init__(self):
        self.device = "cuda"
        self._initialized = False
        self._use_raw_ckpt = False

        policy_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "policy.pt")

        # 1) Use JIT policy.pt first (official baseline)
        if os.path.exists(policy_path):
            print(f"[solution] Loading JIT policy: {policy_path}")
            self.policy = torch.jit.load(policy_path, map_location=self.device)
            self.policy.eval()
            self._use_raw_ckpt = False
        # 2) Fall back to raw training checkpoint
        elif (raw_ckpt := self._find_latest_checkpoint()) is not None:
            print(f"[solution] Loading raw checkpoint: {raw_ckpt}")
            self._load_raw_checkpoint(raw_ckpt)
            self._use_raw_ckpt = True
        else:
            raise FileNotFoundError(
                "No policy found. Place policy.pt in demo/ or train a policy first."
            )

        # Forward velocity command (training range: [-1.0, 1.0])
        # Using 0.5 m/s — safe for the baseline flat-terrain policy
        self.fixed_velocity_commands = torch.tensor(
            [0.5, 0.0, 0.0],
            device=self.device,
            dtype=torch.float32,
        ).view(1, 3)

        # Deferred init
        self.action_dim: int = 0
        self.leg_action_dim: int = 0
        self.arm_action_dim: int = 0
        self.leg_joint_indices: list[int] = []
        self.arm_joint_indices: list[int] = []
        self.train_to_env_action_scale: torch.Tensor | None = None
        self.env_to_train_action_scale: torch.Tensor | None = None
        self.arm_default_action: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # Checkpoint discovery
    # ------------------------------------------------------------------
    def _find_latest_checkpoint(self) -> str | None:
        """Find the latest raw training checkpoint."""
        candidates = []
        # Search demo/ directory
        demo_dir = os.path.dirname(os.path.abspath(__file__))
        for f in os.listdir(demo_dir):
            if f.startswith("model_") and f.endswith(".pt"):
                candidates.append(os.path.join(demo_dir, f))

        # Search training log dir
        if os.path.isdir(self._CKPT_DIR):
            for run_dir in sorted(os.listdir(self._CKPT_DIR), reverse=True):
                run_path = os.path.join(self._CKPT_DIR, run_dir)
                if not os.path.isdir(run_path):
                    continue
                for f in sorted(os.listdir(run_path), reverse=True):
                    if f.startswith("model_") and f.endswith(".pt"):
                        candidates.append(os.path.join(run_path, f))

        if not candidates:
            return None
        candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        return candidates[0]

    def _load_raw_checkpoint(self, ckpt_path: str) -> None:
        """Load a raw RSL-RL checkpoint and extract the actor network."""
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)

        if "actor_state_dict" in ckpt:
            actor_sd = ckpt["actor_state_dict"]
            self._ckpt_iter = ckpt.get("iter", "?")
        elif "model_state_dict" in ckpt:
            actor_sd = {}
            for k, v in ckpt["model_state_dict"].items():
                if k.startswith("actor."):
                    actor_sd["mlp." + k[len("actor."):]] = v
                elif k == "std":
                    actor_sd["distribution.std_param"] = v
            self._ckpt_iter = ckpt.get("iter", "?")
        else:
            raise ValueError(f"Unknown checkpoint format. Keys: {list(ckpt.keys())}")

        print(f"[solution] Checkpoint iteration: {self._ckpt_iter}")
        print(f"[solution] Actor state dict keys: {len(actor_sd)} params")

        # Build MLPModel matching training architecture
        from rsl_rl.models.mlp_model import MLPModel

        self._actor_model = MLPModel(
            obs={"policy": torch.zeros(1, 45, device=self.device)},
            obs_groups={"actor": ["policy"]},
            obs_set="actor",
            output_dim=12,
            hidden_dims=[512, 256, 128],
            activation="elu",
            obs_normalization=False,
            distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0},
        )
        self._actor_model.load_state_dict(actor_sd, strict=False)
        self._actor_model.eval()
        self._actor_model.to(self.device)
        print(f"[solution] Actor model loaded successfully")

    # ------------------------------------------------------------------
    # Optional: custom action spec
    # ------------------------------------------------------------------
    def get_action_spec(self) -> dict[str, dict[str, Any]] | None:
        return {}

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------
    def _init_from_obs(self, proprio: torch.Tensor) -> None:
        self.action_dim = (int(proprio.shape[-1]) - 12) // 3

        robot_info = self._ROBOT_DIM_TABLE.get(self.action_dim)
        if robot_info is None:
            print(
                f"[solution] WARNING: Unknown action_dim={self.action_dim}. "
                f"Assuming all {self.action_dim} joints are leg joints."
            )
            robot_info = {"leg": self.action_dim, "arm": 0, "wheel": 0}

        self.leg_action_dim = robot_info["leg"]
        self.arm_action_dim = robot_info["arm"]
        self.leg_joint_indices = list(range(self.leg_action_dim))
        self.arm_joint_indices = list(
            range(self.leg_action_dim, self.leg_action_dim + self.arm_action_dim)
        )

        self._build_action_scales()
        self.arm_default_action = torch.zeros(
            (1, self.arm_action_dim), device=self.device, dtype=torch.float32
        )
        self._initialized = True

        print(
            f"[solution] Initialized: action_dim={self.action_dim}, "
            f"leg_dim={self.leg_action_dim}, arm_dim={self.arm_action_dim}"
        )

    def _build_action_scales(self) -> None:
        leg_dim = self.leg_action_dim
        if leg_dim == 12:
            train_scales = [0.125, 0.25, 0.25] * 4
        elif leg_dim == 8:
            train_scales = [0.125, 0.25] * 4
        else:
            train_scales = [0.25] * leg_dim

        train_scales_t = torch.tensor(train_scales, device=self.device, dtype=torch.float32).view(1, -1)
        self.train_to_env_action_scale = train_scales_t / self.ACTION_SCALE
        self.env_to_train_action_scale = self.ACTION_SCALE / train_scales_t

    # ------------------------------------------------------------------
    # Observation processing
    # ------------------------------------------------------------------
    def _get_velocity_commands(self, proprio: torch.Tensor) -> torch.Tensor:
        num_envs = proprio.shape[0]
        cmd = self.fixed_velocity_commands.to(dtype=proprio.dtype, device=self.device)
        if num_envs > 1:
            cmd = cmd.repeat(num_envs, 1)
        return cmd

    def _extract_policy_obs(self, obs: dict, action_dim: int) -> torch.Tensor:
        proprio = obs["proprio"].to(self.device)

        idx = 0
        idx += 3  # base_lin_vel (not used by policy)
        base_ang_vel = proprio[:, idx:idx + 3]; idx += 3
        idx += 3  # velocity_commands (overridden below)
        projected_gravity = proprio[:, idx:idx + 3]; idx += 3

        joint_pos_all = proprio[:, idx:idx + action_dim]; idx += action_dim
        joint_vel_all = proprio[:, idx:idx + action_dim]; idx += action_dim
        actions_all = proprio[:, idx:idx + action_dim]

        joint_pos_leg = joint_pos_all[:, self.leg_joint_indices]
        joint_vel_leg = joint_vel_all[:, self.leg_joint_indices]
        actions_env_leg = actions_all[:, self.leg_joint_indices]

        actions_train_leg = (
            actions_env_leg * self.env_to_train_action_scale.to(dtype=proprio.dtype)
        )
        velocity_commands = self._get_velocity_commands(proprio)

        policy_obs = torch.cat(
            [
                base_ang_vel * 0.25,
                projected_gravity,
                velocity_commands,
                joint_pos_leg,
                joint_vel_leg * 0.05,
                actions_train_leg,
            ],
            dim=-1,
        )
        return policy_obs

    # ------------------------------------------------------------------
    # Action mapping
    # ------------------------------------------------------------------
    def _map_policy_action_to_env_action(
        self, action_train: torch.Tensor, action_dim: int
    ) -> torch.Tensor:
        num_envs = action_train.shape[0]
        leg_action_env = action_train * self.train_to_env_action_scale.to(
            dtype=action_train.dtype
        )

        action_env = torch.zeros((num_envs, action_dim), device=self.device, dtype=torch.float32)
        action_env[:, self.leg_joint_indices] = leg_action_env

        if self.arm_action_dim > 0:
            action_env[:, self.arm_joint_indices] = self.arm_default_action.repeat(num_envs, 1)

        return action_env

    # ------------------------------------------------------------------
    # Main inference entry point
    # ------------------------------------------------------------------
    def predicts(self, obs: dict, current_score: float) -> dict:
        proprio = obs["proprio"].to(self.device)

        if not self._initialized:
            self._init_from_obs(proprio)

        if current_score > 500:
            return {"action": [], "giveup": True}

        action_dim = self.action_dim
        policy_obs = self._extract_policy_obs(obs, action_dim)

        with torch.inference_mode():
            if self._use_raw_ckpt:
                action_train = self._actor_model({"policy": policy_obs}, obs_set="actor")
            else:
                action_train = self.policy(policy_obs)

        if not isinstance(action_train, torch.Tensor):
            action_train = torch.as_tensor(action_train, device=self.device, dtype=torch.float32)
        action_train = action_train.to(device=self.device, dtype=torch.float32)
        if action_train.ndim == 1:
            action_train = action_train.unsqueeze(0)

        action_env = self._map_policy_action_to_env_action(action_train, action_dim)
        action_env = action_env.cpu().numpy().tolist()
        return {"action": action_env, "giveup": False}
