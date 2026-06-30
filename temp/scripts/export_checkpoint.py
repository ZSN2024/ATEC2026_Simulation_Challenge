"""Export an RSL-RL checkpoint to a JIT policy.pt for demo/solution.py.

Usage:
    python scripts/export_checkpoint.py <checkpoint_path> [output_path]

Example:
    python scripts/export_checkpoint.py \
        logs/rsl_rl/unitree_b2_rough/2026-06-24_09-09-37/model_200.pt \
        demo/policy_rough.pt
"""
import argparse
import os
import sys

import torch

# Parse args before Isaac Sim starts (avoids Hydra conflicts)
parser = argparse.ArgumentParser()
parser.add_argument("checkpoint", type=str, help="Path to RSL-RL checkpoint (.pt)")
parser.add_argument("output", type=str, nargs="?", default=None, help="Output JIT path")
args = parser.parse_args()

from isaaclab.app import AppLauncher

# Minimal Isaac Sim launch (no rendering needed for export)
AppLauncher.add_app_launcher_args(parser)
app_launcher = AppLauncher(parser.parse_args([]))
simulation_app = app_launcher.app

# Now import rsl-rl and export
import gymnasium as gym
from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils.hydra import hydra_task_config

import importlib.metadata as metadata
import atec_rl_lab.train  # noqa: F401


def main():
    ckpt_path = os.path.abspath(args.checkpoint)
    if not os.path.exists(ckpt_path):
        print(f"ERROR: Checkpoint not found: {ckpt_path}")
        return

    output = args.output or os.path.join(os.path.dirname(ckpt_path), "exported", "policy.pt")
    os.makedirs(os.path.dirname(output), exist_ok=True)

    print(f"Loading checkpoint: {ckpt_path}")

    # Load the checkpoint
    ckpt = torch.load(ckpt_path, map_location="cuda", weights_only=False)

    # Extract actor state dict
    if "actor_state_dict" in ckpt:
        actor_sd = ckpt["actor_state_dict"]
    elif "model_state_dict" in ckpt:
        # Old format: extract actor.* keys
        actor_sd = {}
        for k, v in ckpt["model_state_dict"].items():
            if k.startswith("actor."):
                actor_sd["mlp." + k[len("actor."):]] = v
            elif k == "std":
                actor_sd["distribution.std_param"] = v
    else:
        print("ERROR: Cannot find actor_state_dict or model_state_dict in checkpoint")
        print(f"Keys in checkpoint: {list(ckpt.keys())}")
        return

    # Reconstruct the actor model (same architecture as training)
    from rsl_rl.modules.actor_critic import MLPModel
    # MLPModel expects: obs, obs_groups, obs_set, output_dim, hidden_dims, activation, obs_normalization
    # We just need to create it, then load state dict

    # Build a minimal MLPModel
    # Input: 45 (policy obs) → Output: 12 (leg actions)
    actor = MLPModel(
        obs={"policy": torch.zeros(1, 45)},
        obs_groups={"actor": ["policy"]},
        obs_set="actor",
        output_dim=12,
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0},
    )

    actor.load_state_dict(actor_sd, strict=False)
    actor.eval()

    # Create a JIT-traceable wrapper (same interface as old policy.pt)
    class PolicyWrapper(torch.nn.Module):
        def __init__(self, actor):
            super().__init__()
            self.actor = actor

        def forward(self, obs):
            # obs shape: (1, 45) or (N, 45)
            # actor expects a dict, but old policy.pt takes a flat tensor
            return self.actor({"policy": obs}, obs_set="actor")

    wrapper = PolicyWrapper(actor)
    wrapper = torch.jit.script(wrapper)

    torch.jit.save(wrapper, output)
    print(f"Exported JIT policy to: {output}")
    print(f"File size: {os.path.getsize(output) / 1024:.1f} KB")


if __name__ == "__main__":
    main()
    simulation_app.close()
