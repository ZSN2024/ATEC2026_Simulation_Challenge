import gymnasium as gym

from . import agents  # noqa: F401
from .env import TaskBVwcStage1Env


def _register_stage1_task(task_id: str, env_cfg_entry_point: str) -> None:
    gym.register(
        id=task_id,
        entry_point=f"{__name__}.env:TaskBVwcStage1Env",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": env_cfg_entry_point,
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:TaskBVwcStage1PPORunnerCfg",
        },
    )


_register_stage1_task(
    "ATEC-TaskB-B2wPiper-VWC-Stage1-v0",
    f"{__name__}.env_cfg:TaskBVwcStage1EnvB2WCfg",
)

_register_stage1_task(
    "ATEC-TaskB-B2Piper-VWC-Stage1-v0",
    f"{__name__}.env_cfg:TaskBVwcStage1EnvB2Cfg",
)
