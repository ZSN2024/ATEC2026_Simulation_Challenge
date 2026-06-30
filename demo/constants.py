import torch

# ── Task B scene ──────────────────────────────────────────────
TOTAL_OBJECTS = 18          # 6 sugar + 6 mustard + 6 banana
MAX_SCORE = 36
BIN_CENTER = (-3.0, -10.0)
BIN_RADIUS = 1.0
TERRAIN_SIZE = (20, 20)

# ── B2wPiper: 24 DoF ──────────────────────────────────────────
TOTAL_JOINTS = 24
LEG_DIM    = 12              # 0 … 11
WHEEL_DIM  = 4               # 12 … 15
ARM_DIM    = 6               # 16 … 21
GRIPPER_DIM = 2              # 22 … 23

LEG_START   = 0
WHEEL_START = 12
ARM_START   = 16
GRIPPER_START = 22

# Joint names (from source/.../assets/robots/b2w.py)
LEG_JOINT_NAMES = [
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
]
WHEEL_JOINT_NAMES = [
    "FR_foot_joint", "FL_foot_joint", "RR_foot_joint", "RL_foot_joint",
]
ARM_JOINT_NAMES = [
    "arm_joint1", "arm_joint2", "arm_joint3",
    "arm_joint4", "arm_joint5", "arm_joint6",
]
GRIPPER_JOINT_NAMES = ["arm_joint7", "arm_joint8"]

ALL_JOINT_NAMES = (
    LEG_JOINT_NAMES + WHEEL_JOINT_NAMES + ARM_JOINT_NAMES + GRIPPER_JOINT_NAMES
)

# Default standing leg positions (from B2 init_state)
LEG_LOCK_POS = torch.tensor([
    -0.1,  0.8, -1.5,   # FR: hip, thigh, calf
     0.1,  0.8, -1.5,   # FL
    -0.1,  1.0, -1.5,   # RR
     0.1,  1.0, -1.5,   # RL
], dtype=torch.float32)

# ── Camera ─────────────────────────────────────────────────────
# Head camera on base_link
HEAD_CAM_POS_B = (0.4216, 0.025, 0.06185)         # base-frame position
HEAD_CAM_PITCH_DOWN = 0.523599                      # π/6 rad (30°)
HEAD_CAM_FOCAL_LENGTH = 24.0  # mm
HEAD_CAM_APERTURE = 20.955   # mm
HEAD_CAM_W, HEAD_CAM_H = 640, 480
HEAD_CAM_FX = HEAD_CAM_FY = HEAD_CAM_W * HEAD_CAM_FOCAL_LENGTH / HEAD_CAM_APERTURE  # ≈733.2
HEAD_CAM_CX, HEAD_CAM_CY = HEAD_CAM_W / 2, HEAD_CAM_H / 2

# EE camera on gripper_base
EE_CAM_POS_B = (-0.05, 0.0, 0.06)
EE_CAM_FOCAL_LENGTH = 15.0
EE_CAM_W, EE_CAM_H = 640, 480
EE_CAM_FX = EE_CAM_FY = EE_CAM_W * EE_CAM_FOCAL_LENGTH / HEAD_CAM_APERTURE
EE_CAM_CX, EE_CAM_CY = EE_CAM_W / 2, EE_CAM_H / 2

# ── Action defaults ────────────────────────────────────────────
DEFAULT_ACTION_SPEC = {
    "leg":   {"mode": "position", "scale": 0.5},
    "wheel": {"mode": "velocity", "scale": 5.0},
    "arm":   {"mode": "position", "scale": 0.5},
}

# ── Detection ──────────────────────────────────────────────────
HEIGHT_THRESHOLD = 0.03     # min height above ground (meters)
MIN_CLUSTER_PTS = 30        # min pixels per object
SATURATION_THRESHOLD = 0.5 # max(R,G,B)-min(R,G,B) < this → gray/white/black → excluded
GRASP_RANGE = 1.0      # arm reachable distance (meters)
BIN_APPROACH_RANGE = 1.7   # considered "at bin" (meters)

# ── YOLO Detection ─────────────────────────────────────────────
import os as _os
YOLO_MODEL_PATH = _os.path.join(_os.path.dirname(__file__), "yolo_detector.pt")
YOLO_CONF_THRESH = 0.1
YOLO_CLASS_NAMES = ["sugar_box", "mustard_bottle", "banana"]

# ── Wheel control ──────────────────────────────────────────────
TRACK_WIDTH = 0.5           # wheel track width (meters, approximate)
WHEEL_RADIUS = 0.08         # B2W foot-wheel radius (meters)
MAX_LIN_VEL = 2.0          # m/s
MAX_ANG_VEL = 3.0           # rad/s
