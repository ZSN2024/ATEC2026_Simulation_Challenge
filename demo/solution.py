"""Task B solution — PPO policy + FSM high-level commands + Cartesian IK.

Architecture:
  1. FSM.step(obs) → {velocity_command, ee_goal, ee_rpy, close_gripper, giveup}
  2. obs_adapter builds 869-dim policy observation from proprio + commands
  3. PPO policy → 16-dim action (leg + wheel)
  4. CartesianController.compute_base(ee_goal) → 6-dim arm joint positions
  5. action_adapter assembles full 24-dim action
"""

import torch
import numpy as np
from typing import Any

from atec_rl_lab.utils.cartesian_controller import CartesianController

from demo.constants import (
    ARM_DIM, GRIPPER_DIM, ARM_START,
    DEFAULT_ACTION_SPEC,
)
from demo.perception import YOLOObjectDetector
from demo.fsm import TaskBFSM
from demo.obs_adapter import adapt_obs, reset_history
from demo.policy_loader import load_policy, load_policy_meta
from demo.action_adapter import adapt_action


class AlgSolution:
    """Solution entry point for ATEC Task B."""

    # CartesianController config — matches original GraspController
    ARM_JOINT_NAMES = [
        "arm_joint1", "arm_joint2", "arm_joint3",
        "arm_joint4", "arm_joint5", "arm_joint6",
    ]
    EE_BODY_NAME = "gripper_base"

    def __init__(self):
        self.device = "cuda"
        self.env = None
        self.robot = None
        self.cart_ctrl = None
        self.detector = None
        self.fsm = None
        self.policy = None
        self.policy_meta = None
        self.default_arm_gripper = None  # (8,) default arm(6)+gripper(2) positions
        self.step_count = 0
        self._initialized = False

        # Load policy eagerly so errors surface early
        print("=" * 60)
        try:
            self.policy = load_policy(device=self.device)
            self.policy_meta = load_policy_meta()
            print(f"[Solution] Policy loaded  — obs_dim={self.policy_meta['policy_obs_dim']}, "
                  f"action_dim={self.policy_meta['action_dim']}")
        except FileNotFoundError as e:
            print(f"[Solution] Policy not found ({e}) — will skip PPO inference")
            self.policy = None
            self.policy_meta = None
        print("=" * 60)

    # ── env injection ───────────────────────────────────────

    def set_env(self, env):
        """Called by play_atec_task.py after env creation."""
        self.env = env
        self.robot = env.scene["robot"]
        self.device = str(env.device)

        print("=" * 60)
        print(f"[set_env] Robot articulation obtained successfully")
        print(f"  device   = {self.device}")
        print(f"  num_envs = {env.num_envs}")
        print(f"  joints   = {len(self.robot.joint_names)}")
        print("=" * 60)

        # Cartesian IK controller (position-only — maintains current EE orientation)
        self.cart_ctrl = CartesianController(
            robot=self.robot,
            ee_body_name=self.EE_BODY_NAME,
            arm_joint_names=self.ARM_JOINT_NAMES,
            num_envs=env.num_envs,
            device=self.device,
            command_type="position",
            max_joint_delta=0.05,
        )

        # Cache default arm + gripper positions for action scaling
        self.default_arm_gripper = self.robot.data.default_joint_pos[
            0, ARM_START:ARM_START + ARM_DIM + GRIPPER_DIM
        ].clone()

        # Perception
        self.detector = YOLOObjectDetector(self.robot, env, device=self.device)

        # FSM (only depends on detector now)
        self.fsm = TaskBFSM(
            detector=self.detector,
            device=self.device,
        )

        self._initialized = True

    def on_env_reset(self):
        """Called by play_atec_task.py after env.reset()."""
        if self.cart_ctrl is not None:
            self.cart_ctrl.reset()
        if self.fsm is not None:
            self.fsm.reset()
        reset_history()
        self.step_count = 0
        print("[on_env_reset] Controllers, FSM & obs history reset.")

    # ── action spec ─────────────────────────────────────────

    def get_action_spec(self) -> dict[str, dict[str, Any]] | None:
        return DEFAULT_ACTION_SPEC

    # ── predicts ────────────────────────────────────────────

    def predicts(self, obs, current_score):
        if not self._initialized:
            return {"action": [0.0] * 24, "giveup": False}

        self.step_count += 1

        # ── 1. FSM → high-level commands ─────────────────
        cmd = self.fsm.step(obs, self.step_count, current_score)

        if cmd["giveup"]:
            return {"action": [0.0] * 24, "giveup": True}

        # ── 2. Build policy observation ───────────────────
        vel_cmd = torch.from_numpy(cmd["velocity_command"]).float().unsqueeze(0).to(self.device)
        ee_goal_cmd = torch.from_numpy(cmd["ee_goal"]).float().unsqueeze(0).to(self.device)
        ee_rpy_cmd = torch.from_numpy(cmd["ee_rpy"]).float().unsqueeze(0).to(self.device)

        policy_obs = adapt_obs(
            obs,
            velocity_command=vel_cmd,
            ee_goal_command=ee_goal_cmd,
            ee_goal_orientation_command=ee_rpy_cmd,
            expected_policy_obs_dim=(
                self.policy_meta["policy_obs_dim"] if self.policy_meta else None
            ),
            policy_action_dim=(
                self.policy_meta["action_dim"] if self.policy_meta else 16
            ),
        )

        # ── 3. PPO policy inference → leg + wheel ─────────
        if self.policy is not None:
            with torch.no_grad():
                policy_action = self.policy(policy_obs)  # (1, 16)
        else:
            # Fallback: zeros if no policy loaded
            policy_action = torch.zeros(1, 16, device=self.device)

        # Clip to match training-time RslRlVecEnvWrapper(clip_actions=1.0).
        # Without this the policy output can drift outside [-1, 1], creating
        # a feedback loop through last_action in the observation.
        policy_action = policy_action.clamp(-1.0, 1.0)

        # ── 4. Cartesian IK → arm joint positions ─────────
        ee_goal_b = ee_goal_cmd.clone()  # (1, 3) in base frame
        arm_joint_pos = self.cart_ctrl.compute_base(ee_goal_b)  # (1, 6)

        # ── 5. Gripper positions ──────────────────────────
        gripper_val = 0.3 if cmd["close_gripper"] else 0.0
        gripper_pos = torch.full((1, GRIPPER_DIM), gripper_val, device=self.device)

        # ── 6. Assemble full 24D action ────────────────────
        action = adapt_action(
            policy_action=policy_action,
            arm_joint_pos=arm_joint_pos,
            gripper_pos=gripper_pos,
            default_arm_gripper=self.default_arm_gripper,
            official_action_dim=24,
            arm_scale=DEFAULT_ACTION_SPEC["arm"]["scale"],
        )

        # ── Debug ──────────────────────────────────────────
        if self.step_count <= 5 or self.step_count % 50 == 0:
            a = action.squeeze(0)
            print(f"[solution] step={self.step_count} state={self.fsm.state} "
                  f"vel_cmd={vel_cmd.squeeze(0).tolist()} "
                  f"wheel={a[12:16].tolist()} arm={a[16:20].tolist()}")

        return {"action": action.squeeze(0).cpu().tolist(), "giveup": False}
