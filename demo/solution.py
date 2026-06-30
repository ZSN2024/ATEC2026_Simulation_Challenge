import torch
from typing import Any

from demo.controllers import WheelController, GraspController
from demo.constants import (
    LEG_LOCK_POS, ARM_DIM, GRIPPER_DIM,
    WHEEL_START, ARM_START,
    DEFAULT_ACTION_SPEC,
)
from demo.perception import YOLOObjectDetector
from demo.fsm import TaskBFSM


class AlgSolution:

    def __init__(self):
        self.device = "cuda"
        self.env = None
        self.robot = None
        self.wheel_ctrl = None
        self.grasp_ctrl = None
        self.detector = None
        self.fsm = None
        self.default_jpos = None
        self.step_count = 0
        self._initialized = False

    # ── env injection ──────────────────────────────────────────

    def set_env(self, env):
        """Called by play_atec_task.py after env creation."""
        self.env = env
        self.robot = env.scene["robot"]
        self.device = str(env.device)

        print("=" * 60)
        print("[set_env] Robot articulation obtained successfully")
        print(f"  device   = {self.device}")
        print(f"  num_envs = {env.num_envs}")
        print(f"  joints   = {len(self.robot.joint_names)}")
        print("=" * 60)

        self.wheel_ctrl = WheelController(device=self.device)
        self.grasp_ctrl = GraspController(
            robot=self.robot, device=self.device,
            num_envs=env.num_envs, command_type="position",
        )
        self.detector = YOLOObjectDetector(self.robot, env, device=self.device)

        self.fsm = TaskBFSM(
            wheel_ctrl=self.wheel_ctrl,
            grasp_ctrl=self.grasp_ctrl,
            detector=self.detector,
            robot=self.robot,
            device=self.device,
        )

        self._initialized = True

    def on_env_reset(self):
        """Called by play_atec_task.py after env.reset()."""
        self.default_jpos = self.robot.data.default_joint_pos.clone()
        if self.grasp_ctrl is not None:
            self.grasp_ctrl.reset()
        if self.fsm is not None:
            self.fsm.reset()
        print("[on_env_reset] Controllers & FSM reset.")

    # ── action spec ────────────────────────────────────────────

    def get_action_spec(self) -> dict[str, dict[str, Any]] | None:
        return DEFAULT_ACTION_SPEC

    # ── predicts ───────────────────────────────────────────────

    def predicts(self, obs, current_score):
        if not self._initialized:
            return {"action": [0.0] * 24, "giveup": False}

        self.step_count += 1

        # FSM step returns the action tensor
        action = self.fsm.step(obs, self.step_count, current_score)

        # Debug: print first 3 non-zero actions
        if self.step_count <= 5 or (self.step_count <= 80 and self.step_count % 10 == 0):
            a = action.squeeze(0)
            print(f"[solution] step={self.step_count} state={self.fsm.state} "
                  f"wheel={a[12:16].tolist()} arm={a[16:20].tolist()}...")

        if self.fsm.state == "DONE":
            return {"action": action.squeeze(0).cpu().tolist(), "giveup": True}

        return {"action": action.squeeze(0).cpu().tolist(), "giveup": False}
