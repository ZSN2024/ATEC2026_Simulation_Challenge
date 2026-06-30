# AGENTS.md — ATEC 2026 仿真挑战赛 开发指南

## 项目概述

基于 **NVIDIA Isaac Lab v2.3.2** 的机器人仿真环境，评估腿足机器人的移动操作（loco-manipulation）能力。包含 Task A/B/D/E 四个任务，使用 Gymnasium 标准化接口。

- **Python 3.10+**, **PyTorch 2.5.1**, **PhysX 5 GPU**
- 安装：`conda activate isaaclab` → `cd source/atec_rl_lab && pip install -e .`
- 模型资产需单独下载解压到 `atec_robot_model/`

---

## 核心关注：Task B — 垃圾拾取与投放

机器人需在 20×20m 区域内收集 18 个目标物体（6 糖盒 + 6 芥末瓶 + 6 香蕉），放入中央圆形回收桶（圆心 (-3, -10)，半径 1m）。

| 项目 | 详情 |
|------|------|
| 评分 | 每个物体拾取 +1（EE 接触），投放 +1（物体进圈），最高 36 分 |
| 物体初始位置 | 随机散布于 x∈[-15,-5], y∈[-15,-5] |
| 机器人起点 | (-10, -10) |
| 仿真时长 | 1200 秒 |
| 终止条件 | 所有物体进圈 / 摔倒 / 超时 |
| 支持机器人 | G1 / Tron1Piper / Tron2ALegged / Tron2AWheel / B2Piper / B2wPiper |

### 观察空间（三组）

```
obs["proprio"]  → Tensor [base_lin_vel(3), base_ang_vel(3), velocity_cmds(3), projected_gravity(3), joint_pos(N), joint_vel(N), prev_actions(N)]
obs["extero"]   → Tensor LiDAR height_scan (16线, ±20°垂直, ±180°水平, 10m范围)
obs["image"]    → dict {"head_rgb", "head_depth", "ee_rgb", "ee_depth"}  每图 480×640
```

### 动作空间（按关节组分别控制）

| 组 | 默认模式 | 默认缩放 | 适用机器人 |
|----|---------|----------|-----------|
| `joint_leg` | position | ×0.5 | 所有（G1 全关节归为 leg） |
| `joint_wheel` | velocity | ×5.0 | 轮式机器人 |
| `joint_arm` | position | ×0.5 | 带机械臂的机器人 |

可通过 `AlgSolution.get_action_spec()` 自定义 mode/scale/clip。

---

## 目录结构

```
ATEC2026_Simulation_Challenge/
├── readme.md              # 安装说明、环境矩阵
├── TaskB.md               # 任务B中文描述
├── example.md             # 训练/评估/提交示例
├── demo/
│   └── solution.py        # ★ 参赛唯一入口，实现 AlgSolution.predicts()
├── scripts/
│   ├── play_atec_task.py  # ★ 主评分/运行入口
│   ├── list_envs.py       # 列出所有已注册环境
│   ├── view_task_b.py     # 可视化任务B场景
│   └── rsl_rl/            # PPO 运动训练
├── source/atec_rl_lab/
│   ├── pyproject.toml     # pip 包元数据
│   └── atec_rl_lab/
│       ├── tasks/
│       │   ├── task_base/ # 基类：环境/动作/传感器/MDP/地形
│       │   ├── task_a/    # 越野导航
│       │   ├── task_b/    # ★ 垃圾拾取与投放
│       │   ├── task_d/    # 越障
│       │   └── task_e/    # 桌面操作（Piper only）
│       └── assets/        # 机器人/物体USD资产配置
└── atec_robot_model/      # 3D模型资产（需下载）
```

---

## 关键文件速查

### Task B 核心源码

| 文件 | 说明 |
|------|------|
| `source/.../tasks/task_b/__init__.py` | 注册 6 个 Gym 环境 (`ATEC-TaskB-{Robot}`) |
| `source/.../tasks/task_b/env_cfg.py` | 环境配置：场景、奖励、终止、18 物体随机放置 |
| `source/.../tasks/task_b/terrain.py` | 20×20m 平地 + 橙色圆形回收桶 (直径2m, 高0.5m) |
| `source/.../tasks/task_b/mdp/rewards.py` | `ObjectsInCircle` + `GraspedObjectsByEE` 拾取/投放奖励 |
| `source/.../tasks/task_b/mdp/terminations.py` | `ObjectsInCircleDone` — 所有物体进圈即结束 |

### 基类核心源码

| 文件 | 说明 |
|------|------|
| `source/.../tasks/task_base/envs_base_cfg.py` | 传感器（LiDAR/摄像头/接触）、动作/观察/终止基础配置 |
| `source/.../tasks/task_base/envs_base.py` | `BaseRLEnv` — 环境运行实体 |
| `source/.../tasks/task_base/action_base.py` | 动作规格安全验证/合并/应用 |

---

## 常用命令

```bash
# 环境检查
python scripts/list_envs.py

# 可视化场景
python scripts/view_task_b.py --enable_cameras

# 运行评分（仅仿真，无训练）
python scripts/play_atec_task.py --task ATEC-TaskB-B2Piper --enable_cameras
python scripts/play_atec_task.py --task ATEC-TaskB-B2Piper --enable_cameras --debug
python scripts/play_atec_task.py --task ATEC-TaskB-B2Piper --enable_cameras --video

# 可用的 Task B 环境名
# ATEC-TaskB-G1            ATEC-TaskB-Tron1Piper
# ATEC-TaskB-Tron2ALegged  ATEC-TaskB-Tron2AWheel
# ATEC-TaskB-B2Piper       ATEC-TaskB-B2wPiper
```
**注意** 仿真所用的python环境与agent所在环境不同，如需要执行命令，告诉用户即可
---

## 参赛者开发流程

### 1. 实现 solution.py

文件位置：`demo/solution.py`（**不可改名**）

```python
class AlgSolution:
    def get_action_spec(self) -> dict | None:
        """返回自定义动作规格，返回None使用默认配置"""
        return {
            "leg": {"mode": "position", "scale": 1.0, "clip": [-10.0, 10.0]},
            "arm": {"mode": "effort", "scale": 3.0, "clip": [-12.0, 12.0]},
        }

    def predicts(self, obs, current_score):
        """
        obs: dict with keys "proprio", "extero", "image"
        current_score: float 当前累计分数
        返回: {"action": list, "giveup": bool}
        """
        proprio = obs["proprio"]
        action_dim = (int(proprio.shape[-1]) - 12) // 3
        action = [...]  # 长度 = total_joint_dim
        return {"action": action, "giveup": False}
```

### 2. 加载策略

```python
# 在 __init__ 中
self.policy = torch.jit.load("policy.pt", map_location="cuda")
# 或使用本地路径：os.path.dirname(__file__) + "/policy.pt"
```

### 3. 本地测试

```bash
python scripts/play_atec_task.py --task ATEC-TaskB-B2Piper --enable_cameras --debug
```

---

## 架构关键点

### 配置继承链

```
ManagerBasedRLEnvCfg (Isaac Lab)
  └── BaseEnvCfg (envs_base_cfg.py)
       └── TaskBEnvCfg (task_b/env_cfg.py)
            ├── TaskBEnvB2Cfg     └── ...B2WCfg
            ├── TaskBEnvG1Cfg     └── ...Tron1Cfg
            └── TaskBEnvTron2ALeggedCfg / Tron2AWheelCfg
```

### 关键常量

```python
TARGET_CENTER = (-3.0, -10.0)   # 回收桶圆心
TARGET_MARKER_Z = 0.06          # 标记高度
# 物体类型：Sugar(i<6), Mustard(6≤i<12), Banana(i≥12)，共18个
```

### 奖励机制（`source/.../task_b/mdp/rewards.py`）

- **ObjectsInCircle**: 物体 XY 坐标在圆圈半径内且 Z∈[0,0.5] → +1/物体（一次性，不重复）
- **GraspedObjectsByEE**: EE距离物体 < grasp_dist_thresh (默认0.20m) → +1/物体（一次性）
- 两类奖励独立计算，一个物体理论上可贡献最多 2 分

### 终止条件

- `time_out`: 1200 秒到期
- `illegal_contact`: 非允许部位接触地面（如躯干/thigh）
- `fall`: 机器人根部高度 < 0
- `objects_in_circle_done`: 18 物体全进圈 → 提前结束

### 安全动作规格（`action_base.py`）

参赛者不可直接修改 `env_cfg`，仅能通过 `get_action_spec()` 返回字典，系统安全合并和应用：
- 缺失组 → 使用默认配置
- 机器人无对应组 → 忽略
- 关节名和顺序继承自任务/机器人原有配置

---

## 代码约定

- 使用 `@configclass` 装饰器定义 Isaac Lab 配置类
- 环境通过 `gymnasium.register()` 在 `__init__.py` 中注册
- 配置文件中的 `replace()` 方法用于覆写 `MISSING` 字段
- 地形使用 `trimesh` 构建，通过 `TerrainGeneratorCfg` / `TerrainImporterCfg` 加载
- 观察按组构造（proprio/extero/image），每组 `concatenate_terms` 决定是否拼接
- 仿真无并行训练支持 — 环境仅用于评估
