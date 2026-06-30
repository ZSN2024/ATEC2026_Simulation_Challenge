## 方案总览：Stage1 PPO 策略替换底层控制器

### 新架构数据流

```
FSM.step(obs) → commands = {base_velocity(3), ee_goal(3), ee_rpy(3), close_gripper, giveup}

Solution.predicts(obs):
  1. 从 proprio 提取传感器数据 + FSM 命令 → 拼接 79 维单帧观测
  2. 维护 10 帧历史 → 拼接为 869 维策略观测
  3. PPO 策略推理 → 16 维动作 (12腿 + 4轮)
  4. Cartesian IK 从 ee_goal 计算 6 维臂关节位置
  5. 组装为 24 维全量动作: [leg(12)|wheel(4)|arm(6)|gripper(2)]
```

### 具体修改清单

#### 步骤 1　导出策略（非代码修改，需手动执行一次）
```
运行 temp/scripts/task_b_vwc_stage1/export_stage1.py，输入：
  --checkpoint temp/logs/rsl_rl/task_b_vwc_stage1/2026-06-30_19-07-12/model_7800.pt
  --output demo/policy.pt
生成 demo/policy.pt + demo/policy_meta.json
```

#### 步骤 2　拷贝文件（不改逻辑）
- `temp/demo/taskb_stage1/obs_adapter.py` → `demo/obs_adapter.py`（不需要改）
- `temp/demo/taskb_stage1/policy_loader.py` → `demo/policy_loader.py`（不需要改）

#### 步骤 3　重写 `demo/action_adapter.py`
- **删除**固定的 `ARM_TARGET_REL` 硬编码目标
- **新增** 接受动态 `ee_goal_b` 参数，使用 `CartesianController` 计算臂关节位置
- 输出格式保持不变：`[policy_action(16) | arm_ik(6) | gripper(2)]`

#### 步骤 4　删除独立的 command_scheduler
- FSM 内部直接生成命令，不再需要外部调度器

#### 步骤 5　重构 `demo/fsm.py`（核心改动）
- `step()` 返回字典 `{velocity_command, ee_goal, ee_rpy, close_gripper, giveup}` 而非动作张量
- 移除 `WheelController`、`LegPDController`、`GraspController` 依赖
- 各状态改为输出"高层命令"：

| 状态 | velocity_command | ee_goal |
|------|-----------------|---------|
| INIT | [0,0,0] | 扫描姿态 (IK 目标) |
| SCAN | [v_fwd, 0, 0] | 扫描姿态 |
| NAVIGATE | 根据 target_pos_b 算线速度+角速度 | 扫描姿态/预抓取 |
| GRASP | [0,0,0] | 目标物体位置 (IK) |
| GO_TO_BIN | 根据 bin 方向算速度 | 携带姿态 |
| RELEASE | [0,0,0] | 松手位置 (IK)，开夹爪 |

- 新增 `_compute_base_velocity(target_pos_b)` 把相对位置转为速度命令

#### 步骤 6　重构 `demo/solution.py`
- `__init__`：加载 `policy.pt` + `policy_meta.json`
- `on_env_reset`：调用 `reset_history()` 清空观测历史
- `predicts`：
  1. FSM 返回高层命令
  2. 调用 `adapt_obs()` 构建 869 维策略观测
  3. 策略推理得到 16 维动作
  4. 调用 `adapt_action()` 组装 24 维动作（含 IK 臂+夹爪）
  5. 返回给环境

#### 步骤 7　测试
```bash
python scripts/play_atec_task.py --task ATEC-TaskB-B2wPiper --enable_cameras --debug
```

---

### 关键设计决策

1. **观测历史管理**：`obs_adapter` 内部维护 10 帧历史 buffer，`on_env_reset` 时清空
2. **臂控制**：使用现有 `CartesianController`（`atec_rl_lab.utils.cartesian_controller`），跟原 `GraspController` 一致，可靠且经过验证
3. **夹爪控制**：位置模式，0=开，0.3=关
4. **动作格式兼容**：`prev_actions` 在标准环境中是 24 维，obs_adapter 只读前 16 维（对应 leg+wheel），后 8 维不影响策略

---

