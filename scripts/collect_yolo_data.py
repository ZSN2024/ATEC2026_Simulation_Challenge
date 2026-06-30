"""
Simulation data collector for YOLO fine-tuning.

Captures RGB images from head and EE cameras along with
auto-labeled bounding boxes computed from simulation ground truth
object positions projected to the image plane.

Strategy:
    Robot enters INIT (arm→scan pose, 2s), then SCAN (rotate in place).
    Both head and EE images are saved with ground-truth 2D bboxes.

Usage:
    python scripts/collect_yolo_data.py --task ATEC-TaskB-B2wPiper --samples 1000 --enable_cameras

Output:
    data/yolo_dataset/
        images/{train,val}/frame_XXXXXX.jpg
        labels/{train,val}/frame_XXXXXX.txt
        dataset.yaml
"""

import argparse
import os
import sys
import json

from isaaclab.app import AppLauncher

_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

parser = argparse.ArgumentParser(description="Collect YOLO training data from simulation.")
parser.add_argument("--task", type=str, default="ATEC-TaskB-B2wPiper")
parser.add_argument("--samples", type=int, default=500, help="Total frames to collect (each=1 head + 1 ee).")
parser.add_argument("--output", type=str, default="data/yolo_dataset")
parser.add_argument("--save_interval", type=int, default=30, help="Save every N simulation steps.")
parser.add_argument("--reset_interval", type=int, default=300, help="Reset env every N steps for diversity.")
parser.add_argument("--cameras", type=str, default="head,ee", help="Comma-separated cameras: head,ee")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
import numpy as np  # noqa: E402
from scipy.spatial.transform import Rotation as R  # noqa: E402

import atec_rl_lab.tasks  # noqa: F401, E402
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from atec_rl_lab.tasks.task_base.action_base import apply_safe_action_spec  # noqa: E402
from demo.constants import (  # noqa: E402
    HEAD_CAM_W, HEAD_CAM_H, HEAD_CAM_FX, HEAD_CAM_FY, HEAD_CAM_CX, HEAD_CAM_CY,
    EE_CAM_FX, EE_CAM_FY, EE_CAM_CX, EE_CAM_CY,
    HEAD_CAM_POS_B, HEAD_CAM_PITCH_DOWN,
    DEFAULT_ACTION_SPEC,
)
from demo.controllers import WheelController, GraspController  # noqa: E402

# ── Object half-extents (meters, from YCB dataset) ──────────
OBJ_HALF_EXTENTS = {
    0: np.array([0.10, 0.05, 0.035], dtype=np.float32),   # sugar_box
    1: np.array([0.03, 0.03, 0.10],   dtype=np.float32),   # mustard_bottle
    2: np.array([0.10, 0.025, 0.025], dtype=np.float32),   # banana
}

CLASS_NAMES = ["sugar_box", "mustard_bottle", "banana"]


def object_class_id(obj_idx: int) -> int:
    if obj_idx <= 6:
        return 0
    elif obj_idx <= 12:
        return 1
    else:
        return 2


def compute_2d_bbox(obj_pos_w: np.ndarray, obj_quat_w: np.ndarray,
                    half_extents: np.ndarray,
                    cam_pos_w: np.ndarray, cam_quat_w: np.ndarray,
                    fx: float, fy: float, cx: float, cy: float,
                    img_w: int, img_h: int):
    """Project 3D object bbox to YOLO-format 2D bbox (normalized). Returns (xc,yc,w,h) or None."""
    corners_obj = np.array([
        [-half_extents[0], -half_extents[1], -half_extents[2]],
        [-half_extents[0], -half_extents[1],  half_extents[2]],
        [-half_extents[0],  half_extents[1], -half_extents[2]],
        [-half_extents[0],  half_extents[1],  half_extents[2]],
        [ half_extents[0], -half_extents[1], -half_extents[2]],
        [ half_extents[0], -half_extents[1],  half_extents[2]],
        [ half_extents[0],  half_extents[1], -half_extents[2]],
        [ half_extents[0],  half_extents[1],  half_extents[2]],
    ], dtype=np.float32)

    # Object → world
    R_ow = R.from_quat([obj_quat_w[1], obj_quat_w[2],
                         obj_quat_w[3], obj_quat_w[0]]).as_matrix()
    corners_w = obj_pos_w + (R_ow @ corners_obj.T).T

    # World → camera
    cam_R_w = R.from_quat([cam_quat_w[1], cam_quat_w[2],
                           cam_quat_w[3], cam_quat_w[0]]).as_matrix().T
    corners_cam = (cam_R_w @ (corners_w - cam_pos_w).T).T

    if np.any(corners_cam[:, 2] <= 0.01):
        return None

    u = cx + fx * corners_cam[:, 0] / corners_cam[:, 2]
    v = cy + fy * corners_cam[:, 1] / corners_cam[:, 2]

    valid = (u >= 0) & (u < img_w) & (v >= 0) & (v < img_h)
    if valid.sum() < 2:
        return None

    u1, u2 = u[valid].min(), u[valid].max()
    v1, v2 = v[valid].min(), v[valid].max()

    x_c = ((u1 + u2) / 2) / img_w
    y_c = ((v1 + v2) / 2) / img_h
    bw = (u2 - u1) / img_w
    bh = (v2 - v1) / img_h

    if bw <= 0.001 or bh <= 0.001 or bw > 1.0 or bh > 1.0:
        return None
    return (x_c, y_c, bw, bh)


def get_head_cam_pose_w(robot) -> tuple:
    device = robot.device
    body_pos_w = robot.data.root_pos_w[0, :3].cpu().numpy()
    body_quat_w = robot.data.root_quat_w[0, :].cpu().numpy()
    T_cb = np.array(HEAD_CAM_POS_B, dtype=np.float32)
    r_cb = R.from_euler("xyz", [0.0, HEAD_CAM_PITCH_DOWN, 0.0])
    R_cb = r_cb.as_matrix()
    r_wb = R.from_quat([body_quat_w[1], body_quat_w[2],
                         body_quat_w[3], body_quat_w[0]])
    R_wb = r_wb.as_matrix()
    cam_pos_w = body_pos_w + R_wb @ T_cb
    cam_R_w = R_wb @ R_cb
    cam_quat_w = R.from_matrix(cam_R_w).as_quat(scalar_first=True)
    return cam_pos_w, cam_quat_w


def get_ee_cam_pose_w(robot) -> tuple:
    device = robot.device
    gripper_ids, _ = robot.find_bodies("gripper_base")
    gripper_idx = int(gripper_ids[0])
    body_pos_w = robot.data.body_pos_w[0, gripper_idx, :3].cpu().numpy()
    body_quat_w = robot.data.body_quat_w[0, gripper_idx, :].cpu().numpy()
    T_cb = np.array([-0.05, 0.0, 0.06], dtype=np.float32)
    r_cb = R.from_euler("xyz", [0.0, 0.0, -np.pi / 2])
    R_cb = r_cb.as_matrix()
    r_wb = R.from_quat([body_quat_w[1], body_quat_w[2],
                         body_quat_w[3], body_quat_w[0]])
    R_wb = r_wb.as_matrix()
    cam_pos_w = body_pos_w + R_wb @ T_cb
    cam_R_w = R_wb @ R_cb
    cam_quat_w = R.from_matrix(cam_R_w).as_quat(scalar_first=True)
    return cam_pos_w, cam_quat_w


def build_action(robot, grasp_ctrl, wheel_vec, arm_offset, device):
    """
    Build 24D action tensor conforming to action spec.
    Legs: action=0 (maintain reset pose via PD, position mode scale=0.5).
    Arm: drive to scan pose (position mode scale=0.5).
    Wheel: velocity command divided by scale=5.0.
    """
    from demo.constants import WHEEL_START, WHEEL_DIM, ARM_START, ARM_DIM, GRIPPER_DIM

    action = torch.zeros(1, 24, device=device, dtype=torch.float32)

    # Arm
    default_arm = robot.data.default_joint_pos[0, ARM_START:ARM_START + ARM_DIM + GRIPPER_DIM]
    arm_scale = DEFAULT_ACTION_SPEC["arm"]["scale"]
    action[0, ARM_START:ARM_START + ARM_DIM + GRIPPER_DIM] = (arm_offset - default_arm) / arm_scale

    # Wheel (velocity)
    if wheel_vec is not None:
        wheel_scale = DEFAULT_ACTION_SPEC["wheel"]["scale"]
        action[0, WHEEL_START:WHEEL_START + WHEEL_DIM] = wheel_vec / wheel_scale

    return action


def arm_scan_pose(grasp_ctrl):
    """Arm pose with shoulder raised + elbow bent for EE camera view."""
    rest = grasp_ctrl.compute_rest().squeeze(0).clone()
    rest[1] += np.pi / 3
    rest[2] -= np.pi / 6
    return rest


class DataCollector:
    """Standalone data collector (no FSM/YOLO dependency)."""

    def __init__(self):
        self.env = None
        self.robot = None
        self.device = None
        self.wheel_ctrl = None
        self.grasp_ctrl = None
        self.step_count = 0
        self._initialized = False
        self._state = "INIT"
        self._state_timer = 0.0

        self._total_samples = 0
        self._max_samples = 0
        self._output_dir = ""
        self._save_interval = 5
        self._reset_interval = 300
        self._arm_offset = None
        self._cameras = ["head", "ee"]
        self._cam_stats = {"head": 0, "ee": 0}
        self._cam_present = set()

    def configure(self, output_dir, max_samples, save_interval, reset_interval, cameras_str):
        self._output_dir = os.path.abspath(output_dir)
        self._max_samples = max_samples
        self._save_interval = save_interval
        self._reset_interval = reset_interval
        self._cameras = [c.strip() for c in cameras_str.split(",") if c.strip()]
        for split in ["train", "val"]:
            for sub in ["images", "labels"]:
                os.makedirs(os.path.join(self._output_dir, sub, split), exist_ok=True)
        self._write_dataset_yaml()

        # Resume from existing files
        import glob as _glob
        existing = _glob.glob(os.path.join(self._output_dir, "images", "*", "frame_*.jpg"))
        if existing:
            max_idx = max(int(os.path.splitext(os.path.basename(f))[0].split("_", 1)[1]) for f in existing)
            self._total_samples = max_idx + 1
            print(f"[DataCollect] Found {len(existing)} existing frames, resuming from frame_{self._total_samples:06d}")
        else:
            self._total_samples = 0

    def _write_dataset_yaml(self):
        yaml_path = os.path.join(self._output_dir, "dataset.yaml")
        with open(yaml_path, "w") as f:
            f.write(f"""# YOLO dataset config
path: {self._output_dir}
train: images/train
val: images/val
nc: 3
names:
  0: sugar_box
  1: mustard_bottle
  2: banana
""")
        print(f"[DataCollect] {yaml_path}")

    def set_env(self, env):
        self.env = env
        self.robot = env.scene["robot"]
        self.device = str(env.device)
        self.wheel_ctrl = WheelController(device=self.device)
        self.grasp_ctrl = GraspController(
            robot=self.robot, device=self.device,
            num_envs=env.num_envs, command_type="position",
        )
        self._initialized = True
        print(f"[DataCollect] set_env done. joints={len(self.robot.joint_names)}")

    def get_action_spec(self):
        return DEFAULT_ACTION_SPEC

    def on_env_reset(self):
        if self.grasp_ctrl is not None:
            self.grasp_ctrl.reset()
        self.step_count = 0
        self._state = "INIT"
        self._state_timer = 0.0
        self._arm_offset = arm_scan_pose(self.grasp_ctrl)

    def predicts(self, obs, current_score):
        if not self._initialized:
            return {"action": [0.0] * 24, "giveup": False}

        self.step_count += 1
        self._state_timer += 0.02

        if self._total_samples >= self._max_samples:
            return {"action": [0.0] * 24, "giveup": True}

        # ── Reset periodically for diversity ──
        if self.step_count > 0 and self.step_count % self._reset_interval == 0:
            return {"action": [0.0] * 24, "giveup": True}

        # ── Save frames ──
        if self.step_count % self._save_interval == 0:
            self._save_frame(obs)

        # ── State machine ──
        if self._state == "INIT":
            # Stand still for 2s to let arm settle
            if self._state_timer > 2.0:
                self._state = "SCAN"
                self._state_timer = 0.0
                print(f"[DataCollect] INIT → SCAN (step={self.step_count})")
            return {"action": build_action(
                self.robot, self.grasp_ctrl, None,
                self._arm_offset, self.device,
            ).squeeze(0).cpu().tolist(), "giveup": False}

        elif self._state == "SCAN":
            # Rotate in place with arm in scan pose
            omega = 3.0  # rad/s
            wheel_cmd = self.wheel_ctrl.rotate(omega)
            return {"action": build_action(
                self.robot, self.grasp_ctrl, wheel_cmd,
                self._arm_offset, self.device,
            ).squeeze(0).cpu().tolist(), "giveup": False}

        return {"action": [0.0] * 24, "giveup": False}

    def _save_frame(self, obs):
        import cv2

        images = obs.get("image", {})

        # ── First-call diagnostic: which camera keys are present? ──
        if not self._cam_present:
            available = [k for k in images if k.endswith("_rgb")]
            print(f"[DataCollect] Available camera keys: {available}")
            self._cam_present = set(available)

        camera_specs = []
        if "head" in self._cameras and "head_rgb" in self._cam_present:
            camera_specs.append(("head", get_head_cam_pose_w, HEAD_CAM_FX, HEAD_CAM_FY, HEAD_CAM_CX, HEAD_CAM_CY))
        if "ee" in self._cameras and "ee_rgb" in self._cam_present:
            camera_specs.append(("ee", get_ee_cam_pose_w, EE_CAM_FX, EE_CAM_FY, EE_CAM_CX, EE_CAM_CY))

        # ── Per-camera label counters for this frame ──
        frame_label_counts = {}

        for cam_type, get_cam_pose, fx, fy, cx, cy in camera_specs:
            rgb_tensor = images.get(f"{cam_type}_rgb")
            if rgb_tensor is None:
                continue

            cam_pos_w, cam_quat_w = get_cam_pose(self.robot)

            # ── First-frame deep diagnostic ──
            if not hasattr(self, "_head_diag_printed"):
                self._head_diag_printed = True
                print(f"[DataCollect] [{cam_type}] cam_pos_w=({cam_pos_w[0]:.2f},{cam_pos_w[1]:.2f},{cam_pos_w[2]:.2f})")
                for obj_idx in range(1, 19):
                    try:
                        obj_pos_w = self.env.scene[f"object_{obj_idx}"].data.root_pos_w[0, :3].cpu().numpy()
                        dist = np.linalg.norm(obj_pos_w[:2] - cam_pos_w[:2])
                        print(f"  obj_{obj_idx:02d} pos_w=({obj_pos_w[0]:.1f},{obj_pos_w[1]:.1f},{obj_pos_w[2]:.2f}) "
                              f"dist_2d={dist:.1f}m")
                    except KeyError:
                        pass

            labels = []
            obj_visible = 0  # objects in front of camera (successfully projected)
            for obj_idx in range(1, 19):
                try:
                    obj = self.env.scene[f"object_{obj_idx}"]
                except KeyError:
                    continue
                obj_pos_w = obj.data.root_pos_w[0, :3].cpu().numpy()
                obj_quat_w = obj.data.root_quat_w[0, :].cpu().numpy()
                cls_id = object_class_id(obj_idx)
                bbox = compute_2d_bbox(
                    obj_pos_w, obj_quat_w, OBJ_HALF_EXTENTS[cls_id],
                    cam_pos_w, cam_quat_w, fx, fy, cx, cy,
                    HEAD_CAM_W, HEAD_CAM_H,
                )
                if bbox is not None:
                    obj_visible += 1
                    labels.append(f"{cls_id} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}")

            frame_label_counts[cam_type] = len(labels)

            if not labels:
                continue

            # Convert & save
            rgb_np = rgb_tensor.squeeze().cpu().numpy().copy()
            if rgb_np.max() <= 1.5 and rgb_np.dtype in (np.float32, np.float64):
                rgb_np = (rgb_np * 255).clip(0, 255).astype(np.uint8)
            else:
                rgb_np = rgb_np.clip(0, 255).astype(np.uint8)

            split = "train" if self._total_samples < self._max_samples * 0.9 else "val"
            fname = f"frame_{self._total_samples:06d}"
            img_path = os.path.join(self._output_dir, "images", split, f"{fname}.jpg")
            lbl_path = os.path.join(self._output_dir, "labels", split, f"{fname}.txt")

            cv2.imwrite(img_path, cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR))
            with open(lbl_path, "w") as f:
                f.write("\n".join(labels))

            self._cam_stats[cam_type] += 1
            self._total_samples += 1
            if self._total_samples % 50 == 0:
                hc = frame_label_counts.get("head", 0)
                ec = frame_label_counts.get("ee", 0)
                print(f"[DataCollect] {self._total_samples}/{self._max_samples} frames "
                      f"(head={self._cam_stats['head']} ee={self._cam_stats['ee']}) "
                      f"this_frame: head={hc} objs, ee={ec} objs")


def main():
    output_dir = os.path.join(_PROJ_ROOT, args_cli.output)

    collector = DataCollector()
    collector.configure(
        output_dir=output_dir,
        max_samples=args_cli.samples,
        save_interval=args_cli.save_interval,
        reset_interval=args_cli.reset_interval,
        cameras_str=args_cli.cameras,
    )

    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=1,
        use_fabric=not getattr(args_cli, "disable_fabric", False),
    )
    env_cfg = apply_safe_action_spec(env_cfg, json.dumps(collector.get_action_spec()))

    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    collector.set_env(env.unwrapped)
    obs, _ = env.reset()
    collector.on_env_reset()

    step = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            resp = collector.predicts(obs, 0.0)
            if resp["giveup"]:
                break
            actions = torch.tensor(resp["action"], dtype=torch.float32,
                                   device=args_cli.device).view(1, -1)
            obs, reward, terminated, truncated, info = env.step(actions)
            if terminated.item() or truncated.item():
                obs, _ = env.reset()
                collector.on_env_reset()
            step += 1

    env.close()
    print(f"\n[DataCollect] Done. {collector._total_samples} frames "
          f"(head={collector._cam_stats['head']} ee={collector._cam_stats['ee']}) → {output_dir}")
    simulation_app.close()


if __name__ == "__main__":
    main()
