"""Export Task B Stage1 checkpoint to demo/taskb_stage1/policy.pt."""

import argparse
import json
import os

import torch
from isaaclab.app import AppLauncher


SINGLE_POLICY_OBS_DIM = 79
HISTORY_LEN = 10
POLICY_OBS_DIM = SINGLE_POLICY_OBS_DIM * (HISTORY_LEN + 1)
ACTION_DIM = 16
OFFICIAL_ACTION_DIM = 24
POLICY_HIDDEN_DIMS = [128, 128]


parser = argparse.ArgumentParser(description="Export Task B Stage1 checkpoint to JIT policy.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to raw checkpoint .pt")
parser.add_argument("--output", type=str, default="demo/taskb_stage1/policy.pt", help="Output JIT path")
args_cli, _ = parser.parse_known_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


def _load_actor_state_dict(ckpt_path: str) -> dict[str, torch.Tensor]:
    ckpt = torch.load(ckpt_path, map_location="cuda", weights_only=False)
    if "actor_state_dict" in ckpt:
        return ckpt["actor_state_dict"]
    if "model_state_dict" in ckpt:
        actor_sd = {}
        for key, value in ckpt["model_state_dict"].items():
            if key.startswith("actor."):
                actor_sd["mlp." + key[len("actor."):]] = value
            elif key == "std":
                actor_sd["distribution.std_param"] = value
        return actor_sd
    raise ValueError(f"Unknown checkpoint format: {list(ckpt.keys())}")


def _write_policy_meta(meta_path: str):
    meta = {
        "policy_obs_order": [
            "current: base_ang_vel, projected_gravity, base_velocity_command, ee_goal_pos, ee_goal_rpy, joint_pos, joint_vel, last_action",
            "history[10]: previous current-policy observations, oldest to newest",
        ],
        "policy_obs_dim": POLICY_OBS_DIM,
        "single_policy_obs_dim": SINGLE_POLICY_OBS_DIM,
        "action_dim": ACTION_DIM,
        "official_action_dim": OFFICIAL_ACTION_DIM,
        "policy_action_order": ["leg", "wheel"],
        "joint_order_source": "official_task_b_proprio_joint_order",
        "policy_joint_count": 16,
        "official_joint_count": 24,
        "history_length": HISTORY_LEN,
        "base_command_dim": 3,
        "ee_goal_dim": 3,
        "ee_goal_orientation_dim": 3,
        "obs_scale": {
            "base_ang_vel": 1.0,
            "projected_gravity": 1.0,
            "base_velocity_command": 1.0,
            "ee_goal_pos": 1.0,
            "ee_goal_rpy": 1.0,
            "joint_pos": 1.0,
            "joint_vel": 1.0,
            "last_action": 1.0,
        },
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=True)


def main():
    ckpt_path = os.path.abspath(args_cli.checkpoint)
    output = os.path.abspath(args_cli.output)
    os.makedirs(os.path.dirname(output), exist_ok=True)

    actor_sd = _load_actor_state_dict(ckpt_path)

    from rsl_rl.models.mlp_model import MLPModel

    actor = MLPModel(
        obs={"policy": torch.zeros(1, POLICY_OBS_DIM, device="cuda")},
        obs_groups={"actor": ["policy"]},
        obs_set="actor",
        output_dim=ACTION_DIM,
        hidden_dims=POLICY_HIDDEN_DIMS,
        activation="elu",
        obs_normalization=False,
        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0},
    )
    actor.load_state_dict(actor_sd, strict=False)
    actor.eval()
    actor.to("cuda")

    class PolicyWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self._model = model

        def forward(self, obs):
            return self._model({"policy": obs})

    wrapper = PolicyWrapper(actor)
    example = torch.zeros(1, POLICY_OBS_DIM, device="cuda")
    traced = torch.jit.trace(wrapper, example)
    torch.jit.save(traced, output)

    meta_path = os.path.join(os.path.dirname(output), "policy_meta.json")
    _write_policy_meta(meta_path)

    print(f"Exported policy: {output}")
    print(f"Exported meta: {meta_path}")


if __name__ == "__main__":
    main()
    simulation_app.close()
