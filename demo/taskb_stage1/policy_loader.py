import json
import os

import torch


def _taskb_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def load_policy(device: str = "cuda", path: str | None = None):
    if path is None:
        path = os.path.join(_taskb_dir(), "policy.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    policy = torch.jit.load(path, map_location=device)
    policy.eval()
    return policy


def load_policy_meta(path: str | None = None) -> dict:
    if path is None:
        path = os.path.join(_taskb_dir(), "policy_meta.json")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
