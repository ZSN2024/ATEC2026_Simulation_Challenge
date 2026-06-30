# TaskB Stage1 训练方案说明

本方案将 TaskB 拆成两层：Stage1 只负责轮足底盘的低层运动控制，输出 12 个腿关节动作和 4 个轮关节动作；上层任务逻辑负责给 Stage1 提供底盘速度命令和末端目标命令。当前提交版本只接入 Stage1 作为底层控制，不使用机械臂 IK，机械臂 8 维动作保持为零，后续可在同一接口中补充视觉目标和 IK 控制。

## 训练目的

Stage1 的目标不是直接完成 TaskB 全流程，而是先训练一个稳定的轮足底盘控制器，使机器人能够在 TaskB 场景中保持不倒、轮子接地、底盘可按速度命令前进和转向，并在机械臂目标变化时尽量保持姿态稳定。

训练时 policy 不直接控制机械臂。机械臂在完整方案中由 IK 控制，Stage1 只学习在机械臂扰动和末端目标命令存在时维持底盘稳定。这样可以降低策略输出维度，也避免 policy 同时学习底盘运动和机械臂解算两个不同问题。

## 课程学习

Stage1 使用由易到难的 velocity command curriculum。早期只训练低速前进和站稳，避免一开始就给全范围随机速度、倒车和大 yaw 命令导致策略坍塌。

课程大致分三阶段：

- Stage A：只给低速前进和零 yaw，让轮足结构先学会稳定支撑、轮子接地和基本前进。
- Stage B：逐步加入低速倒车和小 yaw，让策略开始学习转向时的姿态稳定。
- Stage C：放开更完整的前进、后退和 yaw 范围，用于接近最终推理时的底盘控制需求。

末端目标命令也按 VWC 思路设计为连续轨迹，而不是每步跳变的随机点。训练中使用 target、trajectory time、hold time，使机械臂目标变化更平滑，减少对底盘的突然扰动。

## 观测设计

Actor 只使用推理时可获得的信息，避免依赖评测环境中拿不到的仿真内部状态。单帧观测包含：

- base angular velocity
- projected gravity
- base velocity command
- ee goal position
- ee goal orientation command
- joint position
- scaled joint velocity
- last policy action

策略使用 10 帧历史观测，提升低层控制对速度、接触和姿态变化趋势的可观测性。当前 checkpoint 对应单帧 79 维，history 后 policy observation 为 869 维，policy action 为 16 维。

Critic 在训练阶段可以使用更多特权信息，例如 root state、实际 EE 状态、EE error、command phase、wheel contact 和 undesired contact 等，用来提升 value 估计质量；这些信息不进入 actor，因此不影响部署约束。

## Reward 设计

Reward 设计重点是“轮足底盘稳定”，不是复用腿足机器人的 gait reward。保留通用稳定项，同时加入轮足专用约束。

主要正向奖励：

- alive：鼓励 episode 存活。
- track_lin_vel_x：跟踪前向速度命令。
- track_ang_vel_z：跟踪 yaw 角速度命令。
- track_ee_position：在训练环境中约束末端目标跟踪能力。
- leg_posture：鼓励腿保持支撑构型，而不是学习步态摆腿。
- wheel_contact_required：鼓励四个轮子持续接地。

主要惩罚项：

- flat_orientation / roll-pitch penalty：抑制底盘倾斜。
- base_height：约束底盘高度，避免过高重心或趴地。
- lin_vel_z、ang_vel_xy：抑制垂向运动和横滚/俯仰角速度。
- side_slip：轮式底盘不应产生明显横向滑移。
- action_rate、joint_acc、joint_torque：约束动作平滑性和能耗。
- undesired_contacts：惩罚 base、hip、thigh 等非期望部位碰撞。
- stand_still_zero_cmd：零命令时要求稳定站住，不晃动、不空转。

没有直接使用 VWC 的 feet air time、feet height 等腿足步态奖励，因为 B2W 是轮足平台，轮子负责平面运动，腿主要负责支撑和姿态稳定。

## 当前提交策略

当前提交目录中的 `solution.py` 加载 `taskb_stage1_8400.pt`，根据官方 TaskB 的 proprio observation 构造 Stage1 观测，并输出 24 维动作：

- 前 16 维来自 Stage1 policy，用于腿和轮。
- 后 8 维机械臂动作固定为 0。

该版本用于验证 Stage1 底层接入和提交包结构，不是最终得分方案。后续要得分，需要在 `solution.py` 中加入视觉目标搜索、物体接近逻辑，以及可部署的机械臂 IK 或近似触碰控制。
