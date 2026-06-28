"""Export raw RSL-RL checkpoint as JIT policy.pt for demo/solution.py.

Usage:
    python scripts/export_jit.py <checkpoint.pt> [output.pt]
"""
import argparse
import os
import torch

# ── Parse our args first ─────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("checkpoint", type=str, help="Path to raw checkpoint .pt")
parser.add_argument("output", type=str, nargs="?", default=None,
                    help="Output JIT path (default: next to checkpoint)")
# Parse only known args, pass rest to AppLauncher
args_cli, remaining = parser.parse_known_args()

from isaaclab.app import AppLauncher
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


def main():
    ckpt_path = os.path.abspath(args_cli.checkpoint)
    output = args_cli.output or os.path.join(
        os.path.dirname(ckpt_path), "exported", "policy.pt"
    )
    os.makedirs(os.path.dirname(output), exist_ok=True)

    print(f"Loading: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cuda", weights_only=False)

    # ── Extract actor state dict ────────────────────────────────────────
    if "actor_state_dict" in ckpt:
        actor_sd = ckpt["actor_state_dict"]
        iteration = ckpt.get("iter", "?")
    elif "model_state_dict" in ckpt:
        actor_sd = {}
        for k, v in ckpt["model_state_dict"].items():
            if k.startswith("actor."):
                actor_sd["mlp." + k[len("actor."):]] = v
            elif k == "std":
                actor_sd["distribution.std_param"] = v
        iteration = ckpt.get("iter", "?")
    else:
        print(f"ERROR: Unknown format. Keys: {list(ckpt.keys())}")
        return

    print(f"Iteration: {iteration}, params: {len(actor_sd)}")

    # ── Build actor model ────────────────────────────────────────────────
    from rsl_rl.models.mlp_model import MLPModel

    actor = MLPModel(
        obs={"policy": torch.zeros(1, 45, device="cuda")},
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
    actor.to("cuda")

    # ── Wrap for JIT ─────────────────────────────────────────────────────
    class PolicyWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self._model = model

        def forward(self, obs):
            return self._model({"policy": obs})

    wrapper = PolicyWrapper(actor)
    wrapper.eval()
    example = torch.zeros(1, 45, device="cuda")
    traced = torch.jit.trace(wrapper, example)
    torch.jit.save(traced, output)

    size_kb = os.path.getsize(output) / 1024
    print(f"Exported: {output} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
    simulation_app.close()
