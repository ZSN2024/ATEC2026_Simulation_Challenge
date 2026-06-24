import os
import torch
from typing import Any


class AlgSolution:
    """Solution for ATEC tasks with B2Piper robot.

    Task A (Off-road Navigation): Uses trained PPO locomotion policy.
    Task B (Garbage Collection): Uses heuristic search + push strategy.
    """

    ACTION_SCALE = 0.5

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
        self._task_detected: str | None = None
        self._use_raw_ckpt = False

        # ---- Load policy for locomotion (Task A) ----
        policy_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "policy.pt")
        if os.path.exists(policy_path):
            print(f"[solution] Loading JIT policy: {policy_path}")
            self.policy = torch.jit.load(policy_path, map_location=self.device)
            self.policy.eval()
            self._use_raw_ckpt = False
        elif (raw_ckpt := self._find_latest_checkpoint()) is not None:
            print(f"[solution] Loading raw checkpoint: {raw_ckpt}")
            self._load_raw_checkpoint(raw_ckpt)
            self._use_raw_ckpt = True
        else:
            print("[solution] WARNING: No policy found. Task A will not work.")
            self.policy = None

        # Velocity command for locomotion
        self.fixed_velocity_commands = torch.tensor(
            [0.5, 0.0, 0.0], device=self.device, dtype=torch.float32
        ).view(1, 3)

        # ---- Task B state ----
        self._tb_step = 0
        self._tb_phase = "search"       # search | push | carry
        self._tb_search_dir = 1         # +1 or -1 for lawnmower
        self._tb_search_row = 0         # current row index
        self._tb_target = torch.tensor([-3.0, -10.0], device=self.device)  # target circle center
        self._tb_arm_push_pos = torch.tensor(   # arm position for pushing objects on ground
            [0.0, 0.5, -1.0, 0.0, 1.5, 0.0, 0.035, -0.035],
            device=self.device,
        )
        self._tb_arm_carry_pos = torch.tensor(  # arm position for carrying
            [-0.3, 0.8, -1.2, 0.0, 1.2, 0.0, 0.035, -0.035],
            device=self.device,
        )

        # Deferred init
        self.action_dim: int = 0
        self.leg_action_dim: int = 0
        self.arm_action_dim: int = 0
        self.leg_joint_indices: list[int] = []
        self.arm_joint_indices: list[int] = []
        self.train_to_env_action_scale: torch.Tensor | None = None
        self.env_to_train_action_scale: torch.Tensor | None = None

    # ==================================================================
    # Policy loading
    # ==================================================================
    def _find_latest_checkpoint(self) -> str | None:
        ckpt_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                                 "logs", "rsl_rl", "unitree_b2_rough")
        candidates = []
        demo_dir = os.path.dirname(os.path.abspath(__file__))
        for f in os.listdir(demo_dir):
            if f.startswith("model_") and f.endswith(".pt"):
                candidates.append(os.path.join(demo_dir, f))
        if os.path.isdir(ckpt_dir):
            for run_dir in sorted(os.listdir(ckpt_dir), reverse=True):
                run_path = os.path.join(ckpt_dir, run_dir)
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
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        if "actor_state_dict" in ckpt:
            actor_sd = ckpt["actor_state_dict"]
        elif "model_state_dict" in ckpt:
            actor_sd = {}
            for k, v in ckpt["model_state_dict"].items():
                if k.startswith("actor."):
                    actor_sd["mlp." + k[len("actor."):]] = v
                elif k == "std":
                    actor_sd["distribution.std_param"] = v
        else:
            raise ValueError(f"Unknown checkpoint format: {list(ckpt.keys())}")
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

    # ==================================================================
    # Action spec
    # ==================================================================
    def get_action_spec(self) -> dict[str, dict[str, Any]] | None:
        return {}

    # ==================================================================
    # Initialization
    # ==================================================================
    def _init_from_obs(self, proprio: torch.Tensor) -> None:
        self.action_dim = (int(proprio.shape[-1]) - 12) // 3
        robot_info = self._ROBOT_DIM_TABLE.get(self.action_dim)
        if robot_info is None:
            robot_info = {"leg": self.action_dim, "arm": 0, "wheel": 0}
        self.leg_action_dim = robot_info["leg"]
        self.arm_action_dim = robot_info["arm"]
        self.leg_joint_indices = list(range(self.leg_action_dim))
        self.arm_joint_indices = list(range(
            self.leg_action_dim, self.leg_action_dim + self.arm_action_dim
        ))
        self._build_action_scales()

        # Detect task: Task A has terrain at x=-141, Task B has objects
        # Use exteroception (LiDAR) presence to distinguish
        # Task A: robot starts at x=-141 (far from origin)
        # Task B: robot starts at x=-10, y=-10
        # We infer from proprio base_lin_vel position (not directly available)
        # Instead: check if extero key has LiDAR data (Task A) or is empty
        print(f"[solution] Initialized: action_dim={self.action_dim}, "
              f"leg_dim={self.leg_action_dim}, arm_dim={self.arm_action_dim}")
        self._initialized = True

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

    # ==================================================================
    # Observation processing (for locomotion policy)
    # ==================================================================
    def _get_velocity_commands(self, proprio: torch.Tensor) -> torch.Tensor:
        num_envs = proprio.shape[0]
        cmd = self.fixed_velocity_commands.to(dtype=proprio.dtype, device=self.device)
        if num_envs > 1:
            cmd = cmd.repeat(num_envs, 1)
        return cmd

    def _extract_policy_obs(self, obs: dict, action_dim: int) -> torch.Tensor:
        proprio = obs["proprio"].to(self.device)
        idx = 0
        idx += 3  # base_lin_vel
        base_ang_vel = proprio[:, idx:idx + 3]; idx += 3
        idx += 3  # velocity_commands (overridden below)
        projected_gravity = proprio[:, idx:idx + 3]; idx += 3
        joint_pos_all = proprio[:, idx:idx + action_dim]; idx += action_dim
        joint_vel_all = proprio[:, idx:idx + action_dim]; idx += action_dim
        actions_all = proprio[:, idx:idx + action_dim]

        joint_pos_leg = joint_pos_all[:, self.leg_joint_indices]
        joint_vel_leg = joint_vel_all[:, self.leg_joint_indices]
        actions_env_leg = actions_all[:, self.leg_joint_indices]
        actions_train_leg = actions_env_leg * self.env_to_train_action_scale.to(dtype=proprio.dtype)
        velocity_commands = self._get_velocity_commands(proprio)

        return torch.cat([
            base_ang_vel * 0.25, projected_gravity, velocity_commands,
            joint_pos_leg, joint_vel_leg * 0.05, actions_train_leg,
        ], dim=-1)

    def _map_leg_action_to_env(self, action_train: torch.Tensor, action_dim: int) -> torch.Tensor:
        num_envs = action_train.shape[0]
        leg_action_env = action_train * self.train_to_env_action_scale.to(dtype=action_train.dtype)
        action_env = torch.zeros((num_envs, action_dim), device=self.device, dtype=torch.float32)
        action_env[:, self.leg_joint_indices] = leg_action_env
        return action_env

    # ==================================================================
    # Task B: Heuristic loco-manipulation controller
    # ==================================================================
    def _task_b_predicts(self, obs: dict, current_score: float) -> dict:
        """Heuristic controller for Task B garbage collection.

        Strategy:
          1. Walk in a lawnmower pattern to cover the object area
          2. Keep arm low to push objects
          3. Once score plateaus, move to target area
        """
        proprio = obs["proprio"].to(self.device)
        action_dim = self.action_dim
        self._tb_step += 1

        # --- Compute base velocity command (lawnmower pattern) ---
        # Object area: x in [-15, -5], y in [-15, -5]
        # Robot starts at (-10, -10), target at (-3, -10)
        # Simple strategy: walk in expanding rectangles centered on object area

        lin_vel_x = 0.3   # slow forward speed for pushing
        lin_vel_y = 0.0
        ang_vel_z = 0.0

        # Phase switching based on step count
        cycle = self._tb_step % 600  # ~12 second cycle at 0.02s step

        if cycle < 200:
            # Move right (positive y) to sweep
            lin_vel_x = 0.2
            lin_vel_y = 0.3
        elif cycle < 300:
            # Turn around
            lin_vel_x = 0.1
            ang_vel_z = 0.5
        elif cycle < 500:
            # Move left (negative y) to sweep back
            lin_vel_x = 0.2
            lin_vel_y = -0.3
        else:
            # Turn around
            lin_vel_x = 0.1
            ang_vel_z = -0.5

        # --- Compute leg action via locomotion policy ---
        cmd = torch.tensor([[lin_vel_x, lin_vel_y, ang_vel_z]],
                           device=self.device, dtype=torch.float32)
        # Override _get_velocity_commands temporarily
        saved_cmd = self.fixed_velocity_commands.clone()
        self.fixed_velocity_commands = cmd

        if self.policy is not None or self._use_raw_ckpt:
            policy_obs = self._extract_policy_obs(obs, action_dim)
            with torch.inference_mode():
                if self._use_raw_ckpt:
                    leg_action = self._actor_model({"policy": policy_obs}, obs_set="actor")
                else:
                    leg_action = self.policy(policy_obs)
            if not isinstance(leg_action, torch.Tensor):
                leg_action = torch.as_tensor(leg_action, device=self.device, dtype=torch.float32)
            leg_action = leg_action.to(device=self.device, dtype=torch.float32)
            if leg_action.ndim == 1:
                leg_action = leg_action.unsqueeze(0)
        else:
            # No policy: use zero leg action (stand)
            leg_action = torch.zeros((1, self.leg_action_dim), device=self.device)

        self.fixed_velocity_commands = saved_cmd

        # --- Compute arm action (push position) ---
        arm_action = self._tb_arm_push_pos.clone().view(1, -1)

        # --- Combine into full action ---
        action_env = self._map_leg_action_to_env(leg_action, action_dim)
        if self.arm_action_dim > 0:
            action_env[:, self.arm_joint_indices] = arm_action

        action_env = action_env.cpu().numpy().tolist()
        return {"action": action_env, "giveup": False}

    # ==================================================================
    # Task A: Locomotion-only controller
    # ==================================================================
    def _task_a_predicts(self, obs: dict, current_score: float) -> dict:
        """Locomotion-only controller for Task A."""
        proprio = obs["proprio"].to(self.device)
        action_dim = self.action_dim

        if current_score > 500:
            return {"action": [], "giveup": True}

        if self.policy is None and not self._use_raw_ckpt:
            # No policy: zero action
            action = [0.0] * action_dim
            return {"action": [action], "giveup": False}

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

        action_env = self._map_leg_action_to_env(action_train, action_dim)
        # Zero arm action for Task A
        action_env = action_env.cpu().numpy().tolist()
        return {"action": action_env, "giveup": False}

    # ==================================================================
    # Task D: Obstacle Traversal (pit + platform, push box)
    # ==================================================================
    def _task_d_predicts(self, obs: dict, current_score: float) -> dict:
        """Locomotion controller for Task D.

        Robot starts at (-3, 0), needs to cross a pit to reach x > 3.5.
        Box at (-3, 1.6) can be pushed for bonus.
        Strategy: walk forward, steer slightly right to push box toward target.
        """
        proprio = obs["proprio"].to(self.device)
        action_dim = self.action_dim

        if current_score > 500:
            return {"action": [], "giveup": True}

        if self.policy is None and not self._use_raw_ckpt:
            action = [0.0] * action_dim
            return {"action": [action], "giveup": False}

        # Forward + slight right to push box toward target zone
        self.fixed_velocity_commands = torch.tensor(
            [0.5, 0.1, 0.0], device=self.device, dtype=torch.float32
        ).view(1, 3)

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

        action_env = self._map_leg_action_to_env(action_train, action_dim)
        action_env = action_env.cpu().numpy().tolist()
        return {"action": action_env, "giveup": False}

    # ==================================================================
    # Main entry point
    # ==================================================================
    def predicts(self, obs: dict, current_score: float) -> dict:
        proprio = obs["proprio"].to(self.device)

        if not self._initialized:
            self._init_from_obs(proprio)

        # Auto-detect task based on exteroception (LiDAR height scan) variance:
        #   Task B: flat terrain → std ≈ 0
        #   Task D: pit + platform (small terrain 12×8m) → medium variance
        #   Task A: long rough terrain (300m) → highest variance
        if self._task_detected is None:
            if "extero" in obs and obs["extero"] is not None:
                extero = obs["extero"].to(self.device)
                extero_std = extero.std().item() if extero.numel() > 0 else 0.0
                extero_max = extero.max().item() if extero.numel() > 0 else 0.0
                if extero_std < 0.01:
                    self._task_detected = "B"
                    print("[solution] Detected Task B (Garbage Collection)")
                elif extero_max > 3.0:
                    # Task D has deep pit (~1m depth → large range)
                    self._task_detected = "D"
                    print("[solution] Detected Task D (Obstacle Traversal)")
                else:
                    self._task_detected = "A"
                    print("[solution] Detected Task A (Off-road Navigation)")
            else:
                self._task_detected = "A"

        if self._task_detected == "B":
            return self._task_b_predicts(obs, current_score)
        elif self._task_detected == "D":
            return self._task_d_predicts(obs, current_score)
        else:
            return self._task_a_predicts(obs, current_score)
