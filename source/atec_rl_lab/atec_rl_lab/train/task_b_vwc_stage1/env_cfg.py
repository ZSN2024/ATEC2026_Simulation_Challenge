from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm

from atec_rl_lab.tasks.task_b.env_cfg import TaskBEnvB2Cfg, TaskBEnvB2WCfg

from .commands import CommandsCfg
from .observations import ObservationsCfg
from .rewards import RewardsCfg
from . import events as stage1_events
from . import terminations as stage1_mdp
from .terrain import TASK_B_STAGE1_TERRAIN_CFG


class _TaskBVwcStage1EnvMixin:
    base_height_target: float | None = None

    def _apply_stage1_overrides(self):
        # Replicate the official Task B terrain so all cloned training envs stay on terrain.
        self.scene.terrain = TASK_B_STAGE1_TERRAIN_CFG
        self.sim.physics_material = self.scene.terrain.physics_material
        self.episode_length_s = 10.0
        self.commands = CommandsCfg()
        self.observations = ObservationsCfg()
        self.rewards = RewardsCfg()
        if self.base_height_target is None:
            self.rewards.base_height_l2.params["target_height"] = float(self.scene.robot.init_state.pos[2])
        else:
            self.rewards.base_height_l2.params["target_height"] = self.base_height_target

        joint_names = self.scene.robot.joint_names
        self.observations.policy.policy.params["asset_cfg"].joint_names = joint_names
        self.observations.critic.critic.params["asset_cfg"].joint_names = joint_names

        # Keep leg position and wheel velocity commands within the policy action range.
        self.actions.joint_leg.clip = {".*": (-1.0, 1.0)}
        self.actions.joint_wheel.clip = {".*": (-1.0, 1.0)}

        # Turn off task-specific rewards and extra perception during Stage1 training.
        self.observations.extero = None
        self.observations.image = None
        self.scene.head_camera = None
        self.scene.ee_camera = None
        self.scene.ee_dual_camera = None
        self.events.physics_material = None
        self.events.base_external_force_torque = None
        self.events.reset_robot_root = EventTerm(
            func=stage1_events.reset_root_state_vwc_stage1,
            mode="reset",
            params={
                "pose_range": {
                    "x": (-0.5, 0.5),
                    "y": (-0.5, 0.5),
                    "z": (0.0, 0.0),
                    "roll": (0.0, 0.0),
                    "pitch": (0.0, 0.0),
                    "yaw": (-1.57079632679, 1.57079632679),
                },
                "velocity_range": {
                    "x": (-0.1, 0.1),
                    "y": (-0.1, 0.1),
                    "z": (-0.1, 0.1),
                    "roll": (-0.1, 0.1),
                    "pitch": (-0.1, 0.1),
                    "yaw": (-0.1, 0.1),
                },
            },
        )
        self.actions.joint_arm = None
        if hasattr(self.terminations, "objects_in_circle_done"):
            self.terminations.objects_in_circle_done = None
        self.terminations.illegal_contact = None
        self.terminations.fall = DoneTerm(
            func=stage1_mdp.root_height_below_minimum,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "minimum_height": 0.1,
            },
            time_out=False,
        )
        self.terminations.bad_orientation = DoneTerm(
            func=stage1_mdp.roll_pitch_exceeded,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "roll_limit": 0.8,
                "pitch_limit": 0.8,
            },
            time_out=False,
        )

        for obj_idx in range(1, 19):
            attr_name = f"object_{obj_idx}"
            if hasattr(self.scene, attr_name):
                setattr(self.scene, attr_name, None)


@configclass
class TaskBVwcStage1EnvB2Cfg(_TaskBVwcStage1EnvMixin, TaskBEnvB2Cfg):
    """Stage1 Task B training env for Unitree B2 Piper."""

    def __post_init__(self):
        super().__post_init__()
        self._apply_stage1_overrides()


@configclass
class TaskBVwcStage1EnvB2WCfg(_TaskBVwcStage1EnvMixin, TaskBEnvB2WCfg):
    """Stage1 Task B training env for Unitree B2W Piper."""

    def __post_init__(self):
        super().__post_init__()
        self._apply_stage1_overrides()
