# LearningHumanoidRunning

![climb_down](https://github.com/user-attachments/assets/0cbca7ab-ade9-4e77-9b5c-9d2fafead47f)



Training humanoid robots to run using reinforcement learning, modified from the work at https://github.com/rohanpsingh/LearningHumanoidWalking, with additions including: 
1. Running using only legs.
2. Using hands for balance.
3. Running with both hands and legs simultaneously.

Added 14 new arm joints, increasing the observation dimension from 37 to 65, and added some reward functions related to arm movements to support the training of robotic arm operations.

## Code structure:
A rough outline for the repository that might be useful for adding your own robot:
```
LearningHumanoidWalking/
├── envs/                <-- Actions and observation space, PD gains, simulation step, control decimation, init, ...
├── tasks/               <-- Reward function, termination conditions, and more...
├── rl/                  <-- Code for PPO, actor/critic networks, observation normalization process...
├── models/              <-- MuJoCo model files: XMLs/meshes/textures
├── trained/             <-- Contains pretrained model for JVRC
└── scripts/             <-- Utility scripts, etc.
```

## Requirements:
- Python version: 3.7.11  
- [Pytorch](https://pytorch.org/)
- pip install:
  - mujoco==2.2.0
  - [mujoco-python-viewer](https://github.com/rohanpsingh/mujoco-python-viewer)
  - ray==1.9.2
  - transforms3d
  - matplotlib
  - scipy
  - wandb

## Usage:

Environment names supported:  

| Task Description                | Environment name |
|---------------------------------|------------------|
| Basic Walking Task              | 'jvrc_walk'      |
| Stepping Task (using footsteps) | 'jvrc_step'      |
| Walking Task (using arm)        | 'jvrc_arm'       |
| run Task (only using leg)       | 'jvrc_run'       |
| run Task (using leg and arm)    | 'jvrc_run_arm'       |


#### **To train:** 

```
$ python3 run_experiment.py train --logdir <path_to_exp_dir> --num_procs <num_of_cpu_procs> --env <name_of_environment>
```  

#### **To train one experiment with Weights & Biases (wandb):**

```
$ python3 run_experiment.py train \
  --env jvrc_run \
  --logdir ./experiments/jvrc_run \
  --num_procs 12 \
  --use_wandb \
  --wandb_project LearningHumanoidRunning \
  --wandb_run_name jvrc_run_baseline \
  --wandb_video_freq 1
```

Before first use, make sure wandb is installed and logged in:
```
$ python3 -m pip install wandb
$ python3 -m wandb login
```

#### **To play:**

We need to write a script specific to each environment.    
For example, `debug_stepper.py` can be used with the `jvrc_step` environment.  
```
$ python3 scripts/debug_stepper.py --path <path_to_exp_dir>
```

## Hyperparameter tuning (PPO)

All training hyperparameters are exposed in `run_experiment.py`. You can change them from CLI.

Example:
```
$ python3 run_experiment.py train \
  --env jvrc_run \
  --logdir ./experiments/jvrc_run_tune \
  --num_procs 12 \
  --lr 3e-4 \
  --clip 0.15 \
  --epochs 5 \
  --minibatch_size 128 \
  --entropy_coeff 0.005 \
  --gamma 0.99 \
  --lam 0.95 \
  --max_grad_norm 0.05
```

Commonly tuned parameters:
- `--lr`: learning rate (typical range `1e-5` to `3e-4`)
- `--clip`: PPO clip ratio (typical range `0.1` to `0.3`)
- `--epochs`: PPO update epochs per iteration (typical range `3` to `10`)
- `--minibatch_size`: minibatch size (typical values `64`, `128`, `256`)
- `--entropy_coeff`: exploration strength (`0.0` to `0.01`)
- `--gamma`, `--lam`: return/GAE bias-variance tradeoff
- `--std_dev`: initial policy std exponent (larger means more exploration)
- `--max_traj_len`: horizon per rollout
- `--mirror_coeff`: symmetry loss weight

Recommended tuning order:
1. Tune `--lr` and `--clip` first.
2. Tune `--epochs` and `--minibatch_size` for update stability.
3. Tune `--entropy_coeff` / `--std_dev` for exploration.
4. Tune `--gamma` and `--lam` only if return variance remains high.

#### **What you should see:**





https://github.com/user-attachments/assets/08628f41-29f4-463e-947a-f9cd4d0b210c
