import os
import cv2
import torch
import numpy as np
from scipy.spatial.transform import Rotation as R

from demo.constants import (
    HEAD_CAM_POS_B, HEAD_CAM_PITCH_DOWN,
    HEAD_CAM_FX, HEAD_CAM_FY, HEAD_CAM_CX, HEAD_CAM_CY,
    HEAD_CAM_W, HEAD_CAM_H,
    EE_CAM_FX, EE_CAM_FY, EE_CAM_CX, EE_CAM_CY,
    BIN_CENTER, BIN_RADIUS,
    YOLO_MODEL_PATH, YOLO_CONF_THRESH, YOLO_CLASS_NAMES,
)


def depth_to_camera_points(depth: torch.Tensor,
                           fx: float, fy: float,
                           cx: float, cy: float) -> torch.Tensor:
    """
    Convert depth image to 3D points in camera frame.

    depth: (H, W) float32, in meters
    Returns: (3, H*W) in camera frame (X-right, Y-down, Z-forward)
    """
    H, W = depth.shape
    v_idx, u_idx = torch.meshgrid(
        torch.arange(H, device=depth.device),
        torch.arange(W, device=depth.device),
        indexing="ij",
    )
    u = u_idx.float()
    v = v_idx.float()
    d = depth.flatten()
    X = (u.flatten() - cx) * d / fx
    Y = (v.flatten() - cy) * d / fy
    Z = d
    return torch.stack([X, Y, Z], dim=0)  # (3, N)


class YOLOObjectDetector:
    """
    YOLO-based object detector for Task B simulation objects.

    Pipeline:
      1. YOLO detection on RGB image → bounding boxes + class labels
      2. Sample depth at bbox center → 3D position
      3. Transform to world/base frame via camera sensor pose
      4. Apply Z-height / distance / bin filters
      5. Return object centroids in base & world frame

    Supports both fine-tuned models and COCO pre-trained fallback.
    """

    # Fine-tuned model classes
    CLASS_MAP = {
        0: "sugar_box",
        1: "mustard_bottle",
        2: "banana",
    }

    # COCO pre-trained fallback mapping (COCO class ID → our class)
    COCO_FALLBACK = {
        46: "banana",              # banana
        39: "mustard_bottle",      # bottle
        40: "mustard_bottle",      # wine glass → close enough
        73: "sugar_box",           # book → sugar box (rough)
        77: "sugar_box",           # cell phone → unlikely but possible
    }

    def __init__(self, robot, env, device="cuda", model_path=None):
        from ultralytics import YOLO

        self.robot = robot
        self.env = env
        self.device = device

        if model_path is None:
            model_path = YOLO_MODEL_PATH

        self.model_path = model_path
        self.model = None
        self._is_fine_tuned = False
        self._load_model()

        # Head camera intrinsics
        self.head_fx = HEAD_CAM_FX
        self.head_fy = HEAD_CAM_FY
        self.head_cx = HEAD_CAM_CX
        self.head_cy = HEAD_CAM_CY
        # EE camera intrinsics (different focal length: 15mm vs 24mm)
        self.ee_fx = EE_CAM_FX
        self.ee_fy = EE_CAM_FY
        self.ee_cx = EE_CAM_CX
        self.ee_cy = EE_CAM_CY
        self.fx = self.head_fx
        self.fy = self.head_fy
        self.cx = self.head_cx
        self.cy = self.head_cy

        self._debug_dir = os.environ.get("PERCEPTION_DEBUG_DIR", None)
        self._debug_frame = 0

    def _load_model(self):
        """Load YOLO model. Uses fine-tuned model if available, else COCO pre-trained."""
        from ultralytics import YOLO

        if os.path.isfile(self.model_path):
            self.model = YOLO(self.model_path)
            self._is_fine_tuned = True
            print(f"[YOLO] Loaded fine-tuned model: {self.model_path}")
        else:
            print(f"[YOLO] Model not found at {self.model_path}, using COCO pre-trained yolov8n.pt")
            self.model = YOLO("yolov8n.pt")
            self._is_fine_tuned = False

    def _save_debug_image(self, rgb_uint8, raw_boxes, tag, debug_dir):
        """Save detection image with bounding boxes drawn (max 10 per run)."""
        import cv2

        if not hasattr(self, "_save_count"):
            self._save_count = 0
        if self._save_count >= 10:
            return
        self._save_count += 1

        if debug_dir is None:
            debug_dir = os.environ.get("YOLO_DEBUG_DIR", None)
        if debug_dir is None:
            debug_dir = os.path.join(os.path.dirname(__file__), "debug_detections")

        os.makedirs(debug_dir, exist_ok=True)

        viz = rgb_uint8.copy()
        for box, conf, cls_id in raw_boxes:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(viz, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"cls{cls_id}:{conf:.2f}"
            cv2.putText(viz, label, (x1, max(y1 - 5, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        fname = f"det_{tag}_{self._detect_counter:04d}.jpg"
        fpath = os.path.join(debug_dir, fname)
        cv2.imwrite(fpath, cv2.cvtColor(viz, cv2.COLOR_RGB2BGR))
        if self._save_count == 1:
            print(f"[YOLO] Saving debug detections to {debug_dir}/")

    def _get_cam_pose_world(self, use_ee: bool = False):
        device = self.robot.device

        if use_ee:
            gripper_ids, _ = self.robot.find_bodies("gripper_base")
            gripper_idx = int(gripper_ids[0])
            body_pos_w = self.robot.data.body_pos_w[0, gripper_idx, :3]
            body_quat_w = self.robot.data.body_quat_w[0, gripper_idx, :]
            T_cb = torch.tensor([-0.05, 0.0, 0.06], dtype=torch.float32, device=device)
            r_cb = R.from_euler("xyz", [0.0, 0.0, -np.pi / 2])
        else:
            body_pos_w = self.robot.data.root_pos_w[0, :3]
            body_quat_w = self.robot.data.root_quat_w[0, :]
            T_cb = torch.tensor(HEAD_CAM_POS_B, dtype=torch.float32, device=device)
            r_cb = R.from_euler("xyz", [0.0, HEAD_CAM_PITCH_DOWN, 0.0])

        R_cb = torch.tensor(r_cb.as_matrix(), dtype=torch.float32, device=device)

        q = body_quat_w
        r_wb = R.from_quat([q[1].item(), q[2].item(), q[3].item(), q[0].item()])
        R_wb = torch.tensor(r_wb.as_matrix(), dtype=torch.float32, device=device)

        cam_pos_w = body_pos_w + R_wb @ T_cb
        cam_R_w = R_wb @ R_cb

        r_cam_w = R.from_matrix(cam_R_w.cpu().numpy())
        cam_quat_w = torch.tensor(
            list(r_cam_w.as_quat(scalar_first=True)),
            dtype=torch.float32, device=device,
        )
        return cam_pos_w, cam_quat_w

    def _cam_to_base(self, pts_w: torch.Tensor) -> torch.Tensor:
        """Transform points from world frame to base frame."""
        device = pts_w.device
        root_pos = self.robot.data.root_pos_w[0, :3]
        root_quat = self.robot.data.root_quat_w[0]
        q = root_quat
        r = R.from_quat([q[1].item(), q[2].item(), q[3].item(), q[0].item()])
        R_wb_inv = torch.from_numpy(r.as_matrix().T).to(device).float()
        T = root_pos.to(device)

        single = (pts_w.dim() == 1)
        if single:
            pts_w = pts_w.unsqueeze(1)
        result = R_wb_inv @ (pts_w - T.unsqueeze(1))
        if single:
            result = result.squeeze(1)
        return result

    def _depth_to_world(self, u: int, v: int, depth_val: float,
                        fx: float, fy: float, cx: float, cy: float,
                        use_ee: bool = False) -> torch.Tensor | None:
        """Convert a single pixel (u,v) with depth to world-frame 3D point."""
        if depth_val <= 0 or depth_val > 10.0:
            return None

        X = (u - cx) * depth_val / fx
        Y = (v - cy) * depth_val / fy
        Z = depth_val
        pts_cam = torch.tensor([X, Y, Z], dtype=torch.float32, device=self.device)

        device = self.device
        cam_pos, cam_quat = self._get_cam_pose_world(use_ee=use_ee)
        T = cam_pos.to(device)
        q = cam_quat
        r = R.from_quat([q[1].item(), q[2].item(), q[3].item(), q[0].item()])
        Rot = torch.from_numpy(r.as_matrix()).to(device).float()
        return Rot @ pts_cam + T

    def _sample_depth(self, depth_img: torch.Tensor, cx: int, cy: int,
                      kernel: int = 3) -> float:
        """Sample depth around (cx, cy) with a small kernel, return median of valid values."""
        H, W = depth_img.shape
        half = kernel // 2
        y0, y1 = max(0, cy - half), min(H, cy + half + 1)
        x0, x1 = max(0, cx - half), min(W, cx + half + 1)
        patch = depth_img[y0:y1, x0:x1]
        valid = patch[(patch > 0.05) & (patch < 10.0)]
        if valid.numel() == 0:
            return -1.0
        return float(valid.median())

    def detect(self, rgb_img, depth_img=None, min_depth=0.05,
               use_ee=False, debug_dir=None, conf_thresh=None, **kwargs):
        """See class docstring."""
        if conf_thresh is None:
            conf_thresh = YOLO_CONF_THRESH

        if use_ee:
            fx, fy, cx_int, cy_int = self.ee_fx, self.ee_fy, self.ee_cx, self.ee_cy
        else:
            fx, fy, cx_int, cy_int = self.head_fx, self.head_fy, self.head_cx, self.head_cy

        tag = "EE" if use_ee else "head"

        # ── Normalize RGB → numpy uint8 (H, W, 3) ──────────────
        rgb_img = rgb_img.squeeze()
        if rgb_img.dim() == 4:
            rgb_img = rgb_img[0]
        rgb_np = rgb_img.detach().cpu().numpy().copy()
        if rgb_np.max() <= 1.5 and rgb_np.dtype in (np.float32, np.float64):
            rgb_np = (rgb_np * 255).clip(0, 255).astype(np.uint8)
        elif rgb_np.dtype != np.uint8:
            rgb_np = rgb_np.clip(0, 255).astype(np.uint8)
        rgb_uint8 = np.ascontiguousarray(rgb_np[..., :3]) if rgb_np.shape[-1] >= 3 else rgb_np

        # ── Normalize depth ────────────────────────────────────
        if depth_img is None:
            return []
        depth_img = depth_img.squeeze()
        if depth_img.dim() == 3 and depth_img.shape[2] == 1:
            depth_img = depth_img[:, :, 0]

        # ── Tracking counters ───────────────────────────────────
        if not hasattr(self, "_detect_counter"):
            self._detect_counter = 0
        self._detect_counter += 1

        # ── First-call diagnostic ──────────────────────────────
        if not hasattr(self, "_diag_printed"):
            self._diag_printed = True
            print(f"[YOLO:{tag}] first detect: rgb shape={rgb_uint8.shape} dtype={rgb_uint8.dtype} "
                  f"range=[{rgb_uint8.min()},{rgb_uint8.max()}] "
                  f"depth shape={tuple(depth_img.shape)} "
                  f"conf_thresh={conf_thresh} fine_tuned={self._is_fine_tuned}")

        # ── Run YOLO ───────────────────────────────────────────
        results = self.model(rgb_uint8, conf=conf_thresh, verbose=False)

        # ── Collect raw detections ─────────────────────────────
        raw_boxes = []
        for result in results:
            if result.boxes is None:
                continue
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            cls_ids = result.boxes.cls.cpu().numpy().astype(int)
            for i in range(len(boxes)):
                raw_boxes.append((boxes[i], float(confs[i]), int(cls_ids[i])))

        # ── Save debug image when raw detections exist ─────────
        if len(raw_boxes) > 0:
            self._save_debug_image(rgb_uint8, raw_boxes, tag, debug_dir)

        # ── Filter & convert to 3D ────────────────────────────
        objects = []
        stats = {"raw": len(raw_boxes), "no_class": 0, "tiny": 0,
                 "no_depth": 0, "bad_z": 0, "bin": 0, "too_close": 0, "kept": 0}

        for box, conf, cls_id in raw_boxes:
            x1, y1, x2, y2 = box
            cu = int(round((x1 + x2) / 2))
            cv = int(round((y1 + y2) / 2))
            area = int((x2 - x1) * (y2 - y1))

            # Map class
            if self._is_fine_tuned:
                class_name = self.CLASS_MAP.get(cls_id)
            else:
                class_name = self.COCO_FALLBACK.get(cls_id)
            if class_name is None:
                stats["no_class"] += 1
                continue

            if area < 50:
                stats["tiny"] += 1
                continue

            # Sample depth at bbox center
            depth_val = self._sample_depth(depth_img, cu, cv)
            if depth_val < min_depth:
                stats["no_depth"] += 1
                continue

            # Convert to world frame
            pts_w = self._depth_to_world(cu, cv, depth_val, fx, fy, cx_int, cy_int, use_ee)
            if pts_w is None:
                stats["no_depth"] += 1
                continue

            z_w = float(pts_w[2])
            if z_w < 0.0 or z_w > 0.4:
                stats["bad_z"] += 1
                continue

            # Filter out bin area
            dx = float(pts_w[0]) - BIN_CENTER[0]
            dy = float(pts_w[1]) - BIN_CENTER[1]
            if np.sqrt(dx * dx + dy * dy) < BIN_RADIUS:
                stats["bin"] += 1
                continue

            # Transform to base frame
            centroid_b = self._cam_to_base(pts_w)
            dist_b = float(torch.norm(centroid_b))
            if dist_b < 1.0:
                stats["too_close"] += 1
                continue

            stats["kept"] += 1
            objects.append({
                "pos_b": centroid_b.cpu(),
                "pos_w": pts_w.cpu(),
                "center_uv": (cu, cv),
                "n_pts": area,
                "class_name": class_name,
                "conf": conf,
            })

            if not hasattr(self, "_detail_printed"):
                self._detail_printed = True
                print(f"  [YOLO:{tag}] KEPT: {class_name} conf={conf:.2f} "
                      f"box=[{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}] "
                      f"pos_w=({float(pts_w[0]):.2f},{float(pts_w[1]):.2f},{float(pts_w[2]):.2f}) "
                      f"dist={dist_b:.2f}m area={area} depth={depth_val:.2f}m")

        objects.sort(key=lambda o: float(torch.norm(o["pos_b"])))

        # ── Diagnostic summary ──────────────────────────────────
        verbose_early = self._detect_counter <= 5
        if verbose_early or self._detect_counter % 30 == 1:
            print(f"[YOLO:{tag}#{self._detect_counter}] raw={stats['raw']} "
                  f"no_class={stats['no_class']} tiny={stats['tiny']} "
                  f"no_depth={stats['no_depth']} bad_z={stats['bad_z']} "
                  f"bin={stats['bin']} too_close={stats['too_close']} → kept={stats['kept']}")
            if stats["raw"] > 0 and stats["kept"] == 0:
                print(f"  ⚠ All {stats['raw']} raw detections filtered out! Showing raw boxes:")
                for box, conf, cls_id in raw_boxes[:10]:
                    is_fine = self._is_fine_tuned
                    cls_name = self.CLASS_MAP.get(cls_id) if is_fine else self.COCO_FALLBACK.get(cls_id)
                    x1, y1, x2, y2 = box
                    cu, cv = int(round((x1 + x2) / 2)), int(round((y1 + y2) / 2))
                    depth_val = self._sample_depth(depth_img, cu, cv)
                    z_info = ""
                    if depth_val >= min_depth:
                        pts_w = self._depth_to_world(cu, cv, depth_val, fx, fy, cx_int, cy_int, use_ee)
                        if pts_w is not None:
                            z_info = f" z_w={float(pts_w[2]):.3f}"
                    print(f"    cls={cls_id}({cls_name}) conf={conf:.2f} box=[{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}] "
                          f"depth={depth_val:.3f}{z_info}")

        return objects
