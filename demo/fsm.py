"""Task B finite-state machine — outputs high-level commands for the PPO policy.

States: INIT → SCAN → NAVIGATE → GRASP → GO_TO_BIN → RELEASE → (RESET | DONE)

Each state returns a dict of *commands* instead of a joint action tensor.
The commands are consumed by solution.py which runs the PPO policy + IK.
"""

import numpy as np
import torch

from demo.constants import GRASP_RANGE, BIN_CENTER


class TaskBFSM:
    """High-level FSM that produces velocity / EE-goal / gripper commands."""

    # ── Predefined EE targets in base frame ─────────────────
    # Scan pose: EE forward, low enough for camera to see objects
    SCAN_EE_GOAL = np.array([0.45, 0.0, 0.22], dtype=np.float32)
    SCAN_EE_RPY = np.array([0.0, 0.8, 0.0], dtype=np.float32)  # pitch ~45° down

    # Pre-grasp pose (during navigate approach)
    PREGRASP_EE_GOAL = np.array([0.45, 0.0, 0.28], dtype=np.float32)
    PREGRASP_EE_RPY = np.array([0.0, 0.9, 0.0], dtype=np.float32)

    # Carry pose (hold object close to body during GO_TO_BIN)
    CARRY_EE_GOAL = np.array([0.15, 0.0, 0.40], dtype=np.float32)
    CARRY_EE_RPY = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    # Release pose above bin
    RELEASE_EE_GOAL = np.array([0.30, 0.0, 0.50], dtype=np.float32)
    RELEASE_EE_RPY = np.array([0.0, 1.2, 0.0], dtype=np.float32)

    # Rest pose
    REST_EE_GOAL = np.array([0.20, 0.0, 0.45], dtype=np.float32)
    REST_EE_RPY = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    def __init__(self, detector, device="cuda", debug_dir=None):
        self.detector = detector
        self.device = device
        self.debug_dir = debug_dir

        self.state = "INIT"
        self.state_timer = 0.0
        self.scan_start_pos = np.zeros(3, dtype=np.float32)
        self.target_obj = None
        self.detected_cache = []
        self.seen_positions = []

        # NAVIGATE → GRASP transition
        self._stop_timer = 0.0
        # NAVIGATE → RESET delay
        self._reset_delay = 0.0

        # GRASP visual servoing
        self._grasp_close_count = 0
        self._grasp_done = False
        self._grasp_obj_lost_count = 0

        # GRASP → GO_TO_BIN timer (hold after grasp)
        self._go_to_bin_timer = 0.0

        # GO_TO_BIN state
        self._at_bin_timer = 0.0
        self._release_timer = 0.0

        # DONE latch
        self._giveup = False

        # Odometry
        self.base_pos_est = np.zeros(3, dtype=np.float32)
        self.base_yaw_est = 0.0

    # ── Public API ──────────────────────────────────────────

    def reset(self):
        self.state = "INIT"
        self.state_timer = 0.0
        self.scan_start_pos = np.zeros(3, dtype=np.float32)
        self.target_obj = None
        self.detected_cache = []
        self.seen_positions = []
        self._stop_timer = 0.0
        self._reset_delay = 0.0
        self._grasp_close_count = 0
        self._grasp_done = False
        self._grasp_obj_lost_count = 0
        self._go_to_bin_timer = 0.0
        self._at_bin_timer = 0.0
        self._release_timer = 0.0
        self._giveup = False
        self.base_pos_est = np.zeros(3, dtype=np.float32)
        self.base_yaw_est = 0.0

    def step(self, obs, step_count, score):
        """Advance FSM and return high-level command dict.

        Returns:
            dict with keys:
                velocity_command  — np.ndarray (3,)  [vx, vy, omega] base frame
                ee_goal           — np.ndarray (3,)  EE target position base frame
                ee_rpy            — np.ndarray (3,)  EE target RPY base frame
                close_gripper     — bool
                giveup            — bool
        """
        dt = 0.02
        self.state_timer += dt

        proprio = obs["proprio"]
        images = obs.get("image", {})
        self._update_odometry(proprio, dt)

        if self.state == "INIT":
            return self._do_init()
        elif self.state == "SCAN":
            return self._do_scan(images, step_count)
        elif self.state == "NAVIGATE":
            return self._do_navigate(images, step_count)
        elif self.state == "GRASP":
            return self._do_grasp(images)
        elif self.state == "GO_TO_BIN":
            return self._do_go_to_bin()
        elif self.state == "RELEASE":
            return self._do_release()
        elif self.state == "DONE":
            return {
                "velocity_command": np.zeros(3, dtype=np.float32),
                "ee_goal": self.REST_EE_GOAL.copy(),
                "ee_rpy": self.REST_EE_RPY.copy(),
                "close_gripper": False,
                "giveup": True,
            }
        else:
            return self._zero_command()

    # ── Odometry ────────────────────────────────────────────

    def _update_odometry(self, proprio, dt: float):
        lin_vel = proprio[0, 0:3].cpu().numpy()
        ang_vel = proprio[0, 3:6].cpu().numpy()
        self.base_pos_est += lin_vel * dt
        self.base_yaw_est += float(ang_vel[2]) * dt

    # ── Velocity helpers ────────────────────────────────────

    @staticmethod
    def _compute_base_velocity(target_pos_b: np.ndarray) -> np.ndarray:
        """Convert relative target position to base velocity command [vx, vy, omega]."""
        dx, dy = float(target_pos_b[0]), float(target_pos_b[1])
        dist = np.sqrt(dx * dx + dy * dy)
        if dist < 0.30:
            return np.zeros(3, dtype=np.float32)
        angle = np.arctan2(dy, dx)
        vx = np.clip(dist * 0.8, 0.0, 1.5)
        omega = np.clip(angle * 2.5, -2.5, 2.5)
        return np.array([vx, 0.0, omega], dtype=np.float32)

    def _compute_bin_velocity(self) -> np.ndarray:
        """Compute base velocity toward the bin using odometry estimate."""
        dx = BIN_CENTER[0] - self.base_pos_est[0]
        dy = BIN_CENTER[1] - self.base_pos_est[1]
        # Rotate into base frame using estimated yaw
        c = np.cos(-self.base_yaw_est)
        s = np.sin(-self.base_yaw_est)
        dx_b = c * dx - s * dy
        dy_b = s * dx + c * dy
        dist = np.sqrt(dx_b * dx_b + dy_b * dy_b)
        if dist < 1.5:
            return np.zeros(3, dtype=np.float32)
        angle = np.arctan2(dy_b, dx_b)
        vx = np.clip(dist * 0.5, 0.0, 1.5)
        omega = np.clip(angle * 2.0, -2.0, 2.0)
        return np.array([vx, 0.0, omega], dtype=np.float32)

    # ── Zero command ────────────────────────────────────────

    def _zero_command(self):
        return {
            "velocity_command": np.zeros(3, dtype=np.float32),
            "ee_goal": self.SCAN_EE_GOAL.copy(),
            "ee_rpy": self.SCAN_EE_RPY.copy(),
            "close_gripper": False,
            "giveup": self._giveup,
        }

    # ── INIT ────────────────────────────────────────────────

    def _do_init(self):
        """Lock legs, move arm to scan pose, then transition to SCAN."""
        if self.state_timer > 2.0:
            self.state = "SCAN"
            self.state_timer = 0.0
            self.scan_start_pos = self.base_pos_est.copy()
            self.detected_cache = []
            print(f"[FSM] INIT → SCAN (start_pos=({self.scan_start_pos[0]:.1f},{self.scan_start_pos[1]:.1f}))")
        return {
            "velocity_command": np.zeros(3, dtype=np.float32),
            "ee_goal": self.SCAN_EE_GOAL.copy(),
            "ee_rpy": self.SCAN_EE_RPY.copy(),
            "close_gripper": False,
            "giveup": False,
        }

    # ── SCAN ────────────────────────────────────────────────

    def _do_scan(self, images, step_count):
        """Slow forward movement; run detection periodically."""
        v_fwd = -0.5  # m/s

        scan_dist = float(np.linalg.norm(self.base_pos_est[:2] - self.scan_start_pos[:2]))

        # Run detection every 15 steps (0.3 s)
        if step_count % 15 == 0:
            print(f"[FSM] SCAN: step={step_count}, dist={scan_dist:.1f}m")
            rgb_img = images.get("ee_rgb")
            if rgb_img is None:
                if not hasattr(self, "_no_cam_warned"):
                    self._no_cam_warned = True
                    print(f"[FSM] SCAN ⚠ images.keys()={list(images.keys()) if images else 'None'}"
                          f" — ee_rgb not available, are cameras enabled?")
            else:
                _dbg = self.debug_dir
                depth_img = images.get("ee_depth")
                objs = self.detector.detect(rgb_img, depth_img=depth_img, min_depth=0.3,
                                            use_ee=True, debug_dir=_dbg)
                if objs:
                    print(f"[FSM] SCAN: detected {len(objs)} object(s)")
                    for o in objs[:5]:
                        print(f"       pos_w=({o['pos_w'][0]:.2f},{o['pos_w'][1]:.2f},{o['pos_w'][2]:.2f}) "
                              f"dist={torch.norm(o['pos_b']):.2f}m n_pts={o['n_pts']}")
                    # Keep only new objects (not already attempted)
                    for o in objs:
                        key = (round(o["pos_w"][0].item(), 2),
                               round(o["pos_w"][1].item(), 2),
                               round(o["pos_w"][2].item(), 2))
                        if key not in self.seen_positions:
                            self.detected_cache.append(o)
                            self.seen_positions.append(key)

                    # Immediately navigate to the nearest detected object
                    if self.detected_cache:
                        self.detected_cache.sort(key=lambda o: torch.norm(o["pos_b"]).item())
                        self.target_obj = self.detected_cache[0]
                        self.state = "NAVIGATE"
                        self.state_timer = 0.0
                        print(f"[FSM] SCAN → NAVIGATE (target pos_b={self.target_obj['pos_b'].tolist()})")
                        return self._zero_command()

        # Complete scan after 6 m or timeout
        if scan_dist > 6.0:
            print(f"\n[FSM] SCAN complete after {scan_dist:.1f}m: {len(self.detected_cache)} unique objects")
            for i, o in enumerate(self.detected_cache):
                print(f"  [{i}] pos_w=({o['pos_w'][0]:.2f},{o['pos_w'][1]:.2f},{o['pos_w'][2]:.2f}) "
                      f"dist={torch.norm(o['pos_b']):.2f}m n_pts={o['n_pts']}")
            if self.detected_cache:
                self.detected_cache.sort(key=lambda o: torch.norm(o["pos_b"]).item())
                self.target_obj = self.detected_cache[0]
                self.state = "NAVIGATE"
                self.state_timer = 0.0
                print(f"[FSM] SCAN → NAVIGATE (target pos_b={self.target_obj['pos_b'].tolist()})")
            else:
                self.state = "DONE"
                self.state_timer = 0.0
                print("[FSM] SCAN → DONE (no objects found)")
            return self._zero_command()

        # Slow forward movement
        return {
            "velocity_command": np.array([v_fwd, 0.0, 0.0], dtype=np.float32),
            "ee_goal": self.SCAN_EE_GOAL.copy(),
            "ee_rpy": self.SCAN_EE_RPY.copy(),
            "close_gripper": False,
            "giveup": False,
        }

    # ── NAVIGATE ────────────────────────────────────────────

    def _do_navigate(self, images, step_count):
        """Drive toward target object; use visual servoing to update target."""

        # -- Waiting for robot to stop before GRASP --
        if self._stop_timer > 0:
            self._stop_timer += 0.02
            if self._stop_timer >= 0.5:
                self._stop_timer = 0.0
                self.state = "GRASP"
                self.state_timer = 0.0
                print("[FSM] NAVIGATE → GRASP (after 0.5s stop)")
            return self._zero_command()

        # -- Delay before RESET --
        if self._reset_delay > 0:
            self._reset_delay += 0.02
            if self._reset_delay >= 0.1:
                self._reset_delay = 0.0
                self.state = "RESET"
                self.state_timer = 0.0
                print("[FSM] NAVIGATE → RESET (after delay)")
            return self._zero_command()

        if self.target_obj is None:
            self.state = "SCAN"
            self.state_timer = 0.0
            return self._zero_command()

        # Re-detect every 25 steps
        if step_count % 25 == 0:
            rgb_img = images.get("ee_rgb")
            if rgb_img is not None:
                depth_img = images.get("ee_depth")
                objs = self.detector.detect(rgb_img, depth_img=depth_img, min_depth=0.3,
                                            use_ee=True)
                if objs:
                    cur_pos_w = self.target_obj["pos_w"].to(self.device)
                    objs.sort(key=lambda o: float(torch.norm(o["pos_w"].to(self.device)[:2] - cur_pos_w[:2])))
                    best = objs[0]
                    match_dist = float(torch.norm(best["pos_w"].to(self.device)[:2] - cur_pos_w[:2]))
                    if match_dist < 2.0:
                        self.target_obj = best
                    dist = float(torch.norm(self.target_obj["pos_b"]))
                    if dist < GRASP_RANGE:
                        print(f"[FSM] NAVIGATE → stop 0.5s before GRASP (dist={dist:.2f}m)")
                        self._stop_timer = 0.02
                        return self._zero_command()
                else:
                    print(f"[FSM] NAVIGATE → RESET (no objects visible)")
                    self._reset_delay = 0.02
                    return self._zero_command()

        target_pos_b = self.target_obj["pos_b"].to(self.device)
        dist = float(torch.norm(target_pos_b))

        if step_count % 50 == 0:
            print(f"[FSM] NAVIGATE step={step_count} dist={dist:.2f}m "
                  f"target_pos_b=({target_pos_b[0]:.2f},{target_pos_b[1]:.2f},{target_pos_b[2]:.2f})")

        if dist < GRASP_RANGE:
            print(f"[FSM] NAVIGATE → stop 0.5s before GRASP (dist={dist:.2f}m)")
            self._stop_timer = 0.02
            return self._zero_command()

        # Timeout
        if self.state_timer > 20.0:
            print(f"[FSM] NAVIGATE timeout → RESET")
            self._reset_delay = 0.02
            return self._zero_command()

        vel_cmd = self._compute_base_velocity(target_pos_b.cpu().numpy())
        return {
            "velocity_command": vel_cmd,
            "ee_goal": self.PREGRASP_EE_GOAL.copy(),
            "ee_rpy": self.PREGRASP_EE_RPY.copy(),
            "close_gripper": False,
            "giveup": False,
        }

    # ── GRASP ───────────────────────────────────────────────

    def _do_grasp(self, images):
        """Visual servoing: track object with EE camera, approach, close gripper."""

        if self._grasp_done:
            # Hold gripper closed, prepare to go to bin
            self._go_to_bin_timer += 0.02
            if self._go_to_bin_timer > 1.5:
                self.state = "GO_TO_BIN"
                self.state_timer = 0.0
                print("[FSM] GRASP → GO_TO_BIN")
                return {
                    "velocity_command": np.zeros(3, dtype=np.float32),
                    "ee_goal": self.CARRY_EE_GOAL.copy(),
                    "ee_rpy": self.CARRY_EE_RPY.copy(),
                    "close_gripper": True,
                    "giveup": False,
                }
            return {
                "velocity_command": np.zeros(3, dtype=np.float32),
                "ee_goal": self.CARRY_EE_GOAL.copy(),
                "ee_rpy": self.CARRY_EE_RPY.copy(),
                "close_gripper": True,
                "giveup": False,
            }

        # Timeout
        if self.state_timer > 12.0:
            print("[FSM] GRASP timeout → RESET")
            self.state = "RESET"
            self.state_timer = 0.0
            return self._zero_command()

        step_in_state = int(self.state_timer / 0.02)

        # Detection every 20 steps, or sooner if object lost
        if step_in_state % 20 == 0 or self._grasp_obj_lost_count > 3:
            found = False
            ee_rgb = images.get("ee_rgb")
            if ee_rgb is not None:
                ee_depth = images.get("ee_depth")
                objs = self.detector.detect(ee_rgb, depth_img=ee_depth, min_depth=0.12,
                                            use_ee=True)
                if objs:
                    objs.sort(key=lambda o: float(torch.norm(o["pos_b"])))
                    self.target_obj = objs[0]
                    self._grasp_obj_lost_count = 0
                    found = True
            if not found:
                self._grasp_obj_lost_count += 1

        # Distance to target
        dist = float(torch.norm(self.target_obj["pos_b"])) if self.target_obj is not None else 99.0

        if step_in_state % 25 == 0:
            print(f"[FSM] GRASP step={step_in_state} obj_dist={dist:.2f}m "
                  f"close={self._grasp_close_count} lost={self._grasp_obj_lost_count}")

        # Grasp state machine
        if dist < 0.25:
            self._grasp_close_count += 1
            if self._grasp_close_count >= 5:
                self._grasp_done = True
                self._go_to_bin_timer = 0.0
                print(f"[FSM] GRASP success (dist={dist:.2f}m)")
        elif dist < 0.40:
            self._grasp_close_count += 1
        else:
            self._grasp_close_count = max(0, self._grasp_close_count - 1)

        close = self._grasp_done or self._grasp_close_count >= 3
        if self.target_obj is not None:
            target_pos_b = self.target_obj["pos_b"].cpu().numpy().copy()
            xy_dist = np.linalg.norm(target_pos_b[:2])
            z_offset = 0.35 if xy_dist > 0.6 else max(0.05, (xy_dist - 0.25) * 1.0)
            ee_goal = target_pos_b.copy()
            ee_goal[2] += z_offset
        else:
            ee_goal = self.PREGRASP_EE_GOAL.copy()

        return {
            "velocity_command": np.zeros(3, dtype=np.float32),
            "ee_goal": ee_goal.astype(np.float32),
            "ee_rpy": np.array([0.0, 1.2, 0.0], dtype=np.float32),
            "close_gripper": close,
            "giveup": False,
        }

    # ── GO_TO_BIN ───────────────────────────────────────────

    def _do_go_to_bin(self):
        """Drive to the bin while holding the object."""
        dx = BIN_CENTER[0] - self.base_pos_est[0]
        dy = BIN_CENTER[1] - self.base_pos_est[1]
        dist_to_bin = np.sqrt(dx * dx + dy * dy)

        if dist_to_bin < 2.0:
            self._at_bin_timer += 0.02
            if self._at_bin_timer > 0.5:
                self.state = "RELEASE"
                self.state_timer = 0.0
                print(f"[FSM] GO_TO_BIN → RELEASE (dist={dist_to_bin:.1f}m)")
                return {
                    "velocity_command": np.zeros(3, dtype=np.float32),
                    "ee_goal": self.RELEASE_EE_GOAL.copy(),
                    "ee_rpy": self.RELEASE_EE_RPY.copy(),
                    "close_gripper": True,
                    "giveup": False,
                }
            return {
                "velocity_command": np.zeros(3, dtype=np.float32),
                "ee_goal": self.CARRY_EE_GOAL.copy(),
                "ee_rpy": self.CARRY_EE_RPY.copy(),
                "close_gripper": True,
                "giveup": False,
            }

        # Timeout (30 s for bin approach)
        if self.state_timer > 30.0:
            print("[FSM] GO_TO_BIN timeout")
            self.state = "DONE"
            self.state_timer = 0.0
            self._giveup = True

        vel_cmd = self._compute_bin_velocity()
        return {
            "velocity_command": vel_cmd,
            "ee_goal": self.CARRY_EE_GOAL.copy(),
            "ee_rpy": self.CARRY_EE_RPY.copy(),
            "close_gripper": True,
            "giveup": False,
        }

    # ── RELEASE ─────────────────────────────────────────────

    def _do_release(self):
        """Open gripper to release object, then go DONE."""
        self._release_timer += 0.02
        if self._release_timer > 1.5:
            self.state = "DONE"
            self.state_timer = 0.0
            print("[FSM] RELEASE → DONE")
            return {
                "velocity_command": np.zeros(3, dtype=np.float32),
                "ee_goal": self.REST_EE_GOAL.copy(),
                "ee_rpy": self.REST_EE_RPY.copy(),
                "close_gripper": False,
                "giveup": True,
            }
        return {
            "velocity_command": np.zeros(3, dtype=np.float32),
            "ee_goal": self.RELEASE_EE_GOAL.copy(),
            "ee_rpy": self.RELEASE_EE_RPY.copy(),
            "close_gripper": False,
            "giveup": False,
        }

    # ── RESET (internal, not an env reset) ──────────────────

    def _do_reset(self):
        """Return to SCAN to try the next object."""
        self.state = "SCAN"
        self.state_timer = 0.0
        self.scan_start_pos = self.base_pos_est.copy()
        self.target_obj = None
        self.detected_cache = []
        self._grasp_done = False
        self._grasp_close_count = 0
        self._grasp_obj_lost_count = 0
        self._go_to_bin_timer = 0.0
        print("[FSM] RESET → SCAN")
        return self._zero_command()
