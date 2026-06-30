import torch
import numpy as np

from demo.constants import GRASP_RANGE, BIN_APPROACH_RANGE, BIN_CENTER
from demo.controllers import LegPDController


class TaskBFSM:
    """
    Simple state machine for Task B.
    
    States: INIT → SCAN → NAVIGATE → GRASP → (future: GO_TO_BIN, RELEASE, RESET)
    """

    def __init__(self, wheel_ctrl, grasp_ctrl, detector, robot, device="cuda", debug_dir=None):
        self.wheel = wheel_ctrl
        self.grasp = grasp_ctrl
        self.detector = detector
        self.robot = robot
        self.device = device
        self.debug_dir = debug_dir  # for perception debug outputs

        self.state = "INIT"
        self.state_timer = 0.0      # seconds in current state
        self.scan_start_pos = np.zeros(3, dtype=np.float32)  # base position when SCAN started
        self.target_obj = None      # current target object dict
        self.detected_cache = []    # cached detections from SCAN
        self.seen_positions = []    # (x, z) keys of attempted objects
        self._stop_timer = 0.0      # delay timer for NAVIGATE→GRASP transition
        self._reset_delay = 0.0     # delay timer for NAVIGATE→RESET transition

        # Visual servoing state for GRASP
        self._grasp_close_count = 0  # consecutive frames object within grasp distance
        self._grasp_done = False     # latch: grasp complete → hold & transition
        self._grasp_obj_lost_count = 0  # consecutive steps EE camera lost the object

        self.bin_pos = np.array(BIN_CENTER, dtype=np.float32)  # (-3, -10)

        # Direct position controller for leg joints
        self.leg_pd = LegPDController(device=device)
        self._init_leg_offset()

        # Odometry
        self.base_pos_est = np.zeros(3, dtype=np.float32)
        self.base_yaw_est = 0.0

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
        self._init_leg_offset()
        self.base_pos_est = np.zeros(3, dtype=np.float32)
        self.base_yaw_est = 0.0

    def _init_leg_offset(self):
        offset = torch.tensor([
             0.0,  0.25, -0.5,   # FR: hip forward, thigh squat, calf squat
             0.0,  0.25, -0.5,   # FL: hip forward+splay, thigh squat, calf squat
            -0.0,  0.25, -0.5,   # RR: hip out, thigh deep squat, calf deep squat
             0.0,  0.25, -0.5,   # RL: hip out, thigh deep squat, calf deep squat
        ], device=self.device)
        self.leg_pd.set_offset(offset)

    def _update_odometry(self, proprio, dt: float):
        """Dead-reckoning odometry from base velocity."""
        lin_vel = proprio[0, 0:3].cpu().numpy()   # base_lin_vel
        ang_vel = proprio[0, 3:6].cpu().numpy()    # base_ang_vel
        self.base_pos_est += lin_vel * dt
        self.base_yaw_est += float(ang_vel[2]) * dt

    def step(self, obs, step_count, score):
        dt = 0.02  # simulation step size
        self.state_timer += dt

        proprio = obs["proprio"]
        images = obs.get("image", {})
        self._update_odometry(proprio, dt)

        # ── State dispatch ──────────────────────────────────
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
        elif self.state == "RESET":
            return self._do_reset()
        elif self.state == "DONE":
            return self._build_action(
                wheel=torch.zeros(4, device=self.device),
                arm=self.grasp.compute_rest().squeeze(0),
            )
        else:
            return self._build_action()

    # ── INIT ────────────────────────────────────────────────
    def _do_init(self):
        """Lock legs, move arm to scan pose, then transition to SCAN."""
        if self.state_timer > 2.0:
            self.state = "SCAN"
            self.state_timer = 0.0
            self.scan_start_pos = self.base_pos_est.copy()
            self.detected_cache = []
            print(f"[FSM] INIT → SCAN (start_pos=({self.scan_start_pos[0]:.1f},{self.scan_start_pos[1]:.1f}))")
        return self._build_action(
            wheel=torch.zeros(4, device=self.device),
            arm=self._arm_scan_pose(),
        )

    def _arm_scan_pose(self):
        """Arm pose for SCAN: joint1 rotated 180° for EE camera view."""
        rest = self.grasp.compute_rest().squeeze(0).clone()
        # rest[0] += np.pi
        rest[1] += np.pi/3
        rest[2] -= np.pi/6
        return rest

    # ── SCAN ────────────────────────────────────────────────
    def _do_scan(self, images, step_count):
        """Slow forward movement, run detection periodically."""
        from demo.constants import WHEEL_RADIUS

        v_fwd = -0.5  # m/s slow forward

        # Track distance traveled since scan start
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
                        return self._build_action(
                            wheel=torch.zeros(4, device=self.device),
                            arm=self.grasp.compute_rest().squeeze(0),
                        )

        # Complete scan after moving 6m or timeout
        if scan_dist < -6.0:
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
            return self._build_action(
                wheel=torch.zeros(4, device=self.device),
                arm=self._arm_scan_pose(),
            )

        # Slow forward movement
        vr = v_fwd / WHEEL_RADIUS
        vl = v_fwd / WHEEL_RADIUS
        w = torch.tensor([vr, vl, vr, vl], device=self.device, dtype=torch.float32)
        action = self._build_action(wheel=w, arm=self._arm_scan_pose())
        self._debug_action(action, "SCAN")
        if step_count == 55:
            print(f"[FSM debug] SCAN step 55: wheel_raw={w.tolist()}, dist={scan_dist:.1f}m")
        return action

    # ── NAVIGATE ────────────────────────────────────────────
    def _do_navigate(self, images, step_count):
        """Drive toward target object using visual servoing."""
        # Waiting 0.5s for robot to stop before GRASP
        if self._stop_timer > 0:
            self._stop_timer += 0.02
            if self._stop_timer >= 0.5:
                self._stop_timer = 0.0
                self.state = "GRASP"
                self.state_timer = 0.0
                print("[FSM] NAVIGATE → GRASP (after 0.5s stop)")
                return self._build_action(
                    wheel=torch.zeros(4, device=self.device),
                    arm=self._arm_scan_pose(),
                )
            return self._build_action(
                wheel=torch.zeros(4, device=self.device),
                arm=self._arm_scan_pose(),
            )

        # Delay 1s before RESET (from lost-object or timeout trigger)
        if self._reset_delay > 0:
            self._reset_delay += 0.02
            if self._reset_delay >= 0.1:
                self._reset_delay = 0.0
                self.state = "RESET"
                self.state_timer = 0.0
                print("[FSM] NAVIGATE → RESET (after 1s delay)")
            return self._build_action(
                wheel=torch.zeros(4, device=self.device),
                arm=self._arm_scan_pose(),
            )

        if self.target_obj is None:
            self.state = "SCAN"
            return self._build_action(
                wheel=torch.zeros(4, device=self.device),
                arm=self._arm_scan_pose(),
            )

        # Re-detect to track current target every 25 steps
        if step_count % 25 == 0:
            rgb_img = images.get("ee_rgb")
            if rgb_img is not None:
                depth_img = images.get("ee_depth")
                objs = self.detector.detect(rgb_img, depth_img=depth_img, min_depth=0.3,
                                            use_ee=True)
                if objs:
                    # Match to current target by world position, don't switch to other objects
                    cur_pos_w = self.target_obj["pos_w"].to(self.device)
                    objs.sort(key=lambda o: float(torch.norm(o["pos_w"].to(self.device)[:2] - cur_pos_w[:2])))
                    best = objs[0]
                    match_dist = float(torch.norm(best["pos_w"].to(self.device)[:2] - cur_pos_w[:2]))
                    if match_dist < 2.0:
                        self.target_obj = best
                    # else: target lost — keep old target, continue approaching last known position
                    dist = float(torch.norm(self.target_obj["pos_b"]))
                    if dist < GRASP_RANGE:
                        print(f"[FSM] NAVIGATE → stop 0.5s before GRASP (dist={dist:.2f}m)")
                        self._stop_timer = 0.02
                        return self._build_action(
                            wheel=torch.zeros(4, device=self.device),
                            arm=self._arm_scan_pose(),
                        )
                else:
                    print(f"[FSM] NAVIGATE → RESET (no objects visible, re-scan)")
                    self._reset_delay = 0.02
                    return self._build_action(
                        wheel=torch.zeros(4, device=self.device),
                        arm=self._arm_scan_pose(),
                    )

        target_pos_b = self.target_obj["pos_b"].to(self.device)
        dist = float(torch.norm(target_pos_b))

        if step_count % 50 == 0:
            print(f"[FSM] NAVIGATE step={step_count} dist={dist:.2f}m "
                  f"target_pos_b=({target_pos_b[0]:.2f},{target_pos_b[1]:.2f},{target_pos_b[2]:.2f})")

        if dist < GRASP_RANGE:
            print(f"[FSM] NAVIGATE → stop 0.5s before GRASP (dist={dist:.2f}m)")
            self._stop_timer = 0.02
            return self._build_action(
                wheel=torch.zeros(4, device=self.device),
                arm=self._arm_scan_pose(),
            )

        # Timeout
        if self.state_timer > 20.0:
            print(f"[FSM] NAVIGATE timeout → RESET (after 1s delay)")
            self._reset_delay = 0.02
            return self._build_action(
                wheel=torch.zeros(4, device=self.device),
                arm=self._arm_scan_pose(),
            )

        w = self.wheel.compute(target_pos_b)
        return self._build_action(wheel=w, arm=self._arm_scan_pose())

    # ── GRASP ───────────────────────────────────────────────
    def _do_grasp(self, images):
        """Visual servoing: track object via EE camera, approach with IK, close gripper."""
        # After successful grasp: hold gripper closed, then done
        if self._grasp_done:
            if self.state_timer > 1.5:
                self.state = "DONE"
                self.state_timer = 0.0
                print("[FSM] GRASP → DONE")
            return self._build_action(
                wheel=torch.zeros(4, device=self.device),
                arm=self.grasp.compute_grab(None, close_gripper=True).squeeze(0),
            )

        # Timeout safety
        if self.state_timer > 12.0:
            print("[FSM] GRASP timeout → RESET")
            self.state = "RESET"
            self.state_timer = 0.0
            return self._build_action(
                wheel=torch.zeros(4, device=self.device),
                arm=self.grasp.compute_rest().squeeze(0),
            )

        step_in_state = int(self.state_timer / 0.02)

        # ── Detection: every 2 steps, or sooner if object lost ──
        if step_in_state % 20 == 0 or self._grasp_obj_lost_count > 3:
            found = False

            # 1) Try EE camera (close-range params)
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
                if step_in_state == 60:
                    print(f"[FSM] GRASP step={step_in_state} object lost count={self._grasp_obj_lost_count}")

        # ── Distance to target ────────────────────────────────
        dist = float(torch.norm(self.target_obj["pos_b"])) if self.target_obj is not None else 99.0

        if step_in_state % 25 == 0:
            ee_pos_b = self.grasp.ee_pos_w()[0] - self.robot.data.root_pos_w[0, :3]
            from isaaclab.utils.math import quat_rotate_inverse
            ee_pos_b = quat_rotate_inverse(
                self.robot.data.root_quat_w[0:1], ee_pos_b.unsqueeze(0)
            ).squeeze(0)
            print(f"[FSM] GRASP step={step_in_state} obj_dist={dist:.2f}m "
                  f"ee_z={ee_pos_b[2].item():.2f}m "
                  f"close={self._grasp_close_count} lost={self._grasp_obj_lost_count}")

        # ── Grasp state machine ────────────────────────────────
        if dist < 0.25:
            self._grasp_close_count += 1
            if self._grasp_close_count >= 5:
                self._grasp_done = True
                self.state_timer = 0.0
                print(f"[FSM] GRASP success (dist={dist:.2f}m)")
        elif dist < 0.40:
            self._grasp_close_count += 1
        else:
            self._grasp_close_count = max(0, self._grasp_close_count - 1)

        # ── Command arm ────────────────────────────────────────
        close = self._grasp_done or self._grasp_close_count >= 3
        if self.target_obj is not None:
            target_pos_b = self.target_obj["pos_b"].to(self.device).unsqueeze(0).clone()
            # Approach from above: offset Z, descend as XY gets closer
            xy_dist = float(torch.norm(target_pos_b[0, :2]))
            z_offset = 0.35 if xy_dist > 0.6 else max(0.05, (xy_dist - 0.25) * 1.0)
            target_pos_b[0, 2] += z_offset
            arm_action = self.grasp.compute_grab(target_pos_b, close_gripper=close).squeeze(0)
        else:
            arm_action = self.grasp.compute_rest().squeeze(0)

        return self._build_action(
            wheel=torch.zeros(4, device=self.device),
            arm=arm_action,
        )

    # ── Stub states (for future phases) ─────────────────────
    def _do_go_to_bin(self):
        return self._build_action()

    def _do_release(self):
        return self._build_action()

    def _do_reset(self):
        self.state = "SCAN"
        self.state_timer = 0.0
        self.scan_start_pos = self.base_pos_est.copy()
        self.target_obj = None
        self.detected_cache = []
        return self._build_action(
            wheel=torch.zeros(4, device=self.device),
            arm=self.grasp.compute_rest().squeeze(0),
        )

    # ── Action builder ──────────────────────────────────────
    def _build_action(self, wheel=None, arm=None):
        """Build 24D action tensor (same logic as solution.py)."""
        from demo.constants import WHEEL_START, WHEEL_DIM, ARM_START, ARM_DIM, GRIPPER_DIM, LEG_DIM as _LEG_DIM, DEFAULT_ACTION_SPEC
        action = torch.zeros(1, 24, device=self.device, dtype=torch.float32)

        leg_scale = DEFAULT_ACTION_SPEC["leg"]["scale"]
        leg_offset = self.leg_pd.compute()
        action[0, :_LEG_DIM] = leg_offset / leg_scale

        if wheel is not None:
            action[0, WHEEL_START:WHEEL_START + WHEEL_DIM] = wheel / 5.0

        if arm is not None:
            default_arm = self.grasp.robot.data.default_joint_pos[
                0, ARM_START:ARM_START + ARM_DIM + GRIPPER_DIM
            ]
            action[0, ARM_START:ARM_START + ARM_DIM + GRIPPER_DIM] = (arm - default_arm) / 0.5

        return action

    # ── Debug helper ────────────────────────────────────────
    _first_action_printed = False

    def _debug_action(self, action, label=""):
        """Print first non-zero wheel action for debugging."""
        if self._first_action_printed:
            return
        w = action[0, 12:16]
        if w.abs().sum() > 0:
            print(f"[FSM debug] {label} action[12:16] = {w.tolist()} "
                  f"(full={action[0].tolist()[:5]}...)")
            self._first_action_printed = True
