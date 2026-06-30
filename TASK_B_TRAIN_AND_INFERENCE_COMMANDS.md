# Task B Stage1 训练与推理命令

本文档对应当前 `B2wPiper` 的 Task B Stage1 训练链路，使用的任务是：

```text
ATEC-TaskB-B2wPiper-VWC-Stage1-v0
```

官方 Task B 环境 ID 是：

```text
ATEC-TaskB-B2wPiper
```

## 1. 环境准备

远端执行目录：

```bash
cd /home/atec/ATEC2026_Simulation_Challenge
```

激活环境：

```bash
source /home/atec/anaconda3/etc/profile.d/conda.sh
conda activate ATEC_use
export PYTHONPATH=/home/atec/ATEC2026_Simulation_Challenge/source:/home/atec/ATEC2026_Simulation_Challenge/scripts:$PYTHONPATH
```

## 2. 开始训练

最常用训练命令：

```bash
python scripts/task_b_vwc_stage1/train_stage1.py \
  --task ATEC-TaskB-B2wPiper-VWC-Stage1-v0 \
  --num_envs 4096 \
  --max_iterations 12000 \
  --headless
```

说明：

- `--max_iterations 12000` 是当前建议的第一轮正式训练轮数。
- `--num_envs` 不写时用配置默认值；如果显存紧张可降到 `2048` 或 `1024`。
- 当前 PPO 配置里 `save_interval = 200`，所以会保存 `model_200.pt`、`model_400.pt`、`model_600.pt` 这类 checkpoint。

带视频训练：

```bash
python scripts/task_b_vwc_stage1/train_stage1.py \
  --task ATEC-TaskB-B2wPiper-VWC-Stage1-v0 \
  --num_envs 1024 \
  --max_iterations 3000 \
  --video \
  --video_length 300 \
  --video_interval 2000
```

## 3. 继续训练

从指定 run 和 checkpoint 继续：

```bash
python scripts/task_b_vwc_stage1/train_stage1.py \
  --task ATEC-TaskB-B2wPiper-VWC-Stage1-v0 \
  --num_envs 4096 \
  --resume \
  --load_run 2026-06-30_10-57-44 \
  --checkpoint model_7000.pt \
  --max_iterations 8000 \
  --headless
```

说明：

- `--load_run` 写 run 目录名，不带完整路径。
- `--checkpoint` 可以写 `model_1000.pt` 这种文件名。
- `--max_iterations` 是继续训练后的总迭代数目标，不是“再加多少轮”。

## 4. 查看训练日志和 checkpoint

日志根目录：

```bash
logs/rsl_rl/task_b_vwc_stage1
```

查看最新 run：

```bash
ls -lt logs/rsl_rl/task_b_vwc_stage1 | head
```

查看某个 run 下的 checkpoint：

```bash
ls -lh logs/rsl_rl/task_b_vwc_stage1/2026-06-29_18-34-28/model_*.pt
```

## 5. 可视化推理

加载指定 checkpoint 可视化：

```bash
python scripts/task_b_vwc_stage1/play_stage1.py \
  --task ATEC-TaskB-B2wPiper-VWC-Stage1-v0 \
  --checkpoint  /home/atec/ATEC2026_Simulation_Challenge/logs/rsl_rl/task_b_vwc_stage1/2026-06-30_19-07-12/model_9400.pt \
  --num_envs 30 \
  --print_metrics \
  --print_interval 50 \
  --show_command_markers
```

说明：

- `--checkpoint` 在 `play_stage1.py` 里可以直接写完整路径。
- `--print_metrics` 会打印高度、EE 误差、base velocity、base command 等实时指标。
- `--show_command_markers` 会显示 EE goal marker。

实时运行：

```bash
python scripts/task_b_vwc_stage1/play_stage1.py \
  --task ATEC-TaskB-B2wPiper-VWC-Stage1-v0 \
  --checkpoint logs/rsl_rl/task_b_vwc_stage1/2026-06-29_18-34-28/model_1000.pt \
  --num_envs 1 \
  --real-time \
  --print_metrics \
  --show_command_markers
```

录制推理视频：

```bash
python scripts/task_b_vwc_stage1/play_stage1.py \
  --task ATEC-TaskB-B2wPiper-VWC-Stage1-v0 \
  --checkpoint logs/rsl_rl/task_b_vwc_stage1/2026-06-29_18-34-28/model_1000.pt \
  --num_envs 1 \
  --video \
  --video_length 500
```

## 6. 导出 Stage1 policy

把原始训练 checkpoint 导出为 demo 使用的 `policy.pt`：

```bash
python scripts/task_b_vwc_stage1/export_stage1.py \
  --checkpoint logs/rsl_rl/task_b_vwc_stage1/2026-06-29_18-34-28/model_1000.pt \
  --output demo/taskb_stage1/policy.pt
```

导出后还会写出：

```text
demo/taskb_stage1/policy_meta.json
```

## 7. 官方 Task B 环境 smoke 测试

用官方 Task B B2W 环境测试 demo adapter：

```bash
python scripts/task_b_vwc_stage1/smoke_demo_stage1.py \
  --task ATEC-TaskB-B2wPiper \
  --num_envs 1 \
  --steps 128
```

这个命令验证的是：

- `demo/taskb_stage1/solution_stage1.py`
- `demo/taskb_stage1/obs_adapter.py`
- `demo/taskb_stage1/policy.pt`

是否能在官方 Task B 环境里正常跑通。

## 8. 辅助调试命令

查看 Stage1 命令采样：

```bash
python scripts/task_b_vwc_stage1/probe_commands.py --num_envs 4 --headless
```

查看轮子动作响应：

```bash
python scripts/task_b_vwc_stage1/probe_wheel_action_response.py \
  --task ATEC-TaskB-B2wPiper-VWC-Stage1-v0 \
  --num_envs 1 \
  --steps 20 \
  --wheel_action 0.5 \
  --headless
```

查看 reward 输出：

```bash
python scripts/task_b_vwc_stage1/probe_rewards.py \
  --task ATEC-TaskB-B2wPiper-VWC-Stage1-v0 \
  --num_envs 8 \
  --steps 32 \
  --headless
```

## 9. 查看官方 Task B B2W 环境

查看官方 Task B B2W，不加载 Stage1 policy：

```bash
python scripts/view_task_b.py --robot b2w --num_envs 1
```

这个命令只是查看官方 Task B 环境配置，不是 Stage1 训练环境。

## 10. 当前建议

当前建议分三档：

1. `3000` 轮：小规模 smoke，确认训练链路稳定。
2. `12000` 轮：第一轮正式训练，已经覆盖 Stage A，并跑入 Stage B。
3. `30000+` 轮：在第一轮结果正常后，再做更完整训练。

按此前远端实际速度粗估：

- `1000` 轮约 `18` 分钟
- `3000` 轮约 `50-60` 分钟
- `12000` 轮约 `3.5-4` 小时
- `30000` 轮约 `9-10` 小时

