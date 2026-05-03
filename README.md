<h1 align="center"><a href="https://github.com/LovingPastry/LearningHumanoidRunning">LearningHumanoidRunning</a></h1>

<p align="center">
  <img src="https://github.com/user-attachments/assets/0cbca7ab-ade9-4e77-9b5c-9d2fafead47f" alt="climb_down" />
</p>

这是一个使用强化学习训练人形机器人奔跑的项目，基于 https://github.com/rohanpsingh/LearningHumanoidWalking 修改而来，主要新增了以下能力：
1. 仅使用双腿进行奔跑。
2. 使用双手辅助保持平衡。
3. 同时使用双手和双腿进行奔跑。

项目新增了 14 个手臂关节，使观测维度从 37 提升到 65，并加入了一些与手臂运动相关的奖励函数，以支持带机械臂动作的人形机器人训练。

## 代码结构
下面是仓库的大致结构，如果你想接入自己的机器人，可以先从这里了解：
```
LearningHumanoidWalking/
├── envs/                <-- 动作与观测空间、PD 增益、仿真步长、控制降采样、初始化等
├── tasks/               <-- 奖励函数、终止条件等
├── rl/                  <-- PPO、actor/critic 网络、观测归一化等
├── models/              <-- MuJoCo 模型文件：XML、mesh、texture
├── trained/             <-- JVRC 的预训练模型
└── scripts/             <-- 工具脚本等
```

## 环境要求
- Python 版本：`3.7.11`
- [Pytorch](https://pytorch.org/)
- 通过 `pip` 安装：
  - `mujoco==2.2.0`
  - [mujoco-python-viewer](https://github.com/rohanpsingh/mujoco-python-viewer)
  - `ray==1.9.2`
  - `transforms3d`
  - `matplotlib`
  - `scipy`

### 安装示例命令
```bash
conda create -n humanoid python==3.7.11
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
pip install -r requirements.txt
```

## 使用说明

支持的环境名称如下：

| 任务说明 | 环境名称 |
|---------------------------------|------------------|
| 基础行走任务 | `jvrc_walk` |
| 台阶任务（使用落脚点） | `jvrc_step` |
| 带手臂的行走任务 | `jvrc_arm` |
| 仅使用双腿的奔跑任务 | `jvrc_run` |
| 使用双腿和双手的奔跑任务 | `jvrc_run_arm` |

### **训练**

#### 单环境训练

最基本的训练命令如下：

```bash
python run_experiment.py train \
  --env <name_of_environment> \
  --logdir <path_to_exp_dir> \
  --num_procs <num_of_cpu_procs>
```

示例：

```bash
python run_experiment.py train \
  --env jvrc_run \
  --logdir ./experiments/jvrc_run \
  --num_procs 12
```

#### 单环境续训

`run_experiment.py` 支持通过 `--continued` 从已有 checkpoint 继续训练。

恢复逻辑如下：

- 如果 `--continued` 指向实验目录，程序会优先查找 `training_state.pt`
- 如果找到了 `training_state.pt`，会从最近一次保存的训练状态继续
- 如果没有 `training_state.pt`，则退回到 `actor.pt` 和 `critic.pt`
- `actor.pt` 和 `critic.pt` 是历史最佳 reward 对应的模型，不一定是最近一次迭代保存的模型
- 也可以显式指定某个快照，例如 `training_state_999.pt` 或 `actor_999.pt`

从目录中的最新训练状态继续：

```bash
python run_experiment.py train \
  --env jvrc_run \
  --logdir ./experiments/jvrc_run \
  --num_procs 12 \
  --continued ./experiments/jvrc_run
```

从指定迭代的训练状态继续：

```bash
python run_experiment.py train \
  --env jvrc_run \
  --logdir ./experiments/jvrc_run \
  --num_procs 12 \
  --continued ./experiments/jvrc_run/training_state_999.pt
```

从历史最佳模型继续训练：

```bash
python run_experiment.py train \
  --env jvrc_run \
  --logdir ./experiments/jvrc_run_finetune \
  --num_procs 12 \
  --continued ./experiments/jvrc_run/actor.pt
```

建议：

- 想严格续上原来的训练进度，优先使用 `training_state.pt`
- 想从历史最佳模型再训练一段，使用 `actor.pt`，并建议换一个新的 `logdir`

#### 五环境批量训练脚本 [train_five_envs.sh](scripts/train_five_envs.sh)

脚本会依次训练这 5 个环境：

- `jvrc_run`
- `jvrc_run_arm`
- `jvrc_walk`
- `jvrc_step`
- `jvrc_arm`

基础用法：

```bash
bash scripts/train_five_envs.sh <base_logdir> <num_procs> [extra_train_args...]
```

示例：

```bash
bash scripts/train_five_envs.sh ./experiments/five_envs 12
```

脚本支持的恢复参数：

- `--resume`：每个环境从 `<base_logdir>/<env>` 继续训练
- `--continued_root <dir>`：每个环境从 `<dir>/<env>` 继续训练

注意：

- 不要直接把 `--continued` 传给 `train_five_envs.sh`
- 因为 5 个环境各自需要不同的 checkpoint 路径，脚本会自动按环境名拼接

在原目录中恢复五个环境的训练：

```bash
bash scripts/train_five_envs.sh ./experiments/five_envs 12 \
  --resume
```

从另一个根目录恢复五个环境的训练：

```bash
bash scripts/train_five_envs.sh ./experiments/five_envs_v2 12 \
  --continued_root ./experiments/five_envs
```

#### 使用 wandb 记录实验

如果 `wandb` 提示视频编码依赖 `moviepy`，请先安装：

```bash
pip install moviepy
```

单环境训练时使用 wandb：

```bash
export WANDB_API_KEY="<Enter API Key>"
wandb login
export MUJOCO_GL=egl
python run_experiment.py train \
  --env jvrc_run \
  --logdir ./experiments/jvrc_run \
  --num_procs 12 \
  --use_wandb \
  --wandb_project LearningHumanoidRunning \
  --wandb_video_freq 1 \
  --wandb_video_len 600 \
  --wandb_video_fps 30 \
  --wandb_video_on_start
```

单环境续训时使用 wandb
```bash
python run_experiment.py train \
  --env jvrc_run \
  --logdir ./experiments/jvrc_run_wandb \
  --num_procs 12 \
  --continued ./experiments/jvrc_run \
  --use_wandb \
  --wandb_project LearningHumanoidRunning \
  --wandb_video_freq 1 \
  --wandb_video_len 600 \
  --wandb_video_fps 30
```


五环境脚本使用 wandb：

```bash
export WANDB_API_KEY="<Enter API Key>"
wandb login
export MUJOCO_GL=egl
bash scripts/train_five_envs.sh ./experiments/five_envs 12 \
  --use_wandb \
  --wandb_project LearningHumanoidRunning \
  --wandb_video_freq 1 \
  --wandb_video_len 600 \
  --wandb_video_fps 30
```

如果你希望在第一次训练迭代开始之前，先做一次评估并上传视频，可以额外加上：

```bash
--wandb_video_on_start
```

### **运行已训练策略**

每个环境通常需要单独的播放脚本。  
例如，`jvrc_step` 环境可以使用 `debug_stepper.py`：
```bash
PYTHONPATH=.:$PYTHONPATH python scripts/debug_stepper.py --path <path_to_exp_dir>
```

#### **你应该看到的效果：**

<h1 align="center">
https://github.com/user-attachments/assets/08628f41-29f4-463e-947a-f9cd4d0b210c
</a></h1>
