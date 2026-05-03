import os
import sys
import argparse
import re
import ray
from functools import partial

import numpy as np
import torch
import pickle

from envs.jvrc import JvrcArmEnv
from rl.algos.ppo import PPO
from rl.policies.actor import Gaussian_FF_Actor
from rl.policies.critic import FF_V
from rl.envs.normalize import get_normalization_params
from rl.envs.wrappers import SymmetricEnv


def parse_iteration_from_checkpoint(path):
    match = re.search(r"_(\d+)\.pt$", os.path.basename(path))
    if match is None:
        return None
    return int(match.group(1))


def resolve_resume_paths(continued):
    actor_path = None
    critic_path = None
    training_state_path = None

    if os.path.isdir(continued):
        actor_path = os.path.join(continued, "actor.pt")
        critic_path = os.path.join(continued, "critic.pt")
        candidate_state = os.path.join(continued, "training_state.pt")
        if os.path.isfile(candidate_state):
            training_state_path = candidate_state
    elif os.path.isfile(continued):
        basename = os.path.basename(continued)
        dirname = os.path.dirname(continued)
        if basename.startswith("training_state") and basename.endswith(".pt"):
            training_state_path = continued
        elif basename.startswith("actor") and basename.endswith(".pt"):
            actor_path = continued
            critic_path = os.path.join(dirname, basename.replace("actor", "critic", 1))
            suffix = basename[len("actor") : -3]
            candidate_state = os.path.join(dirname, "training_state" + suffix + ".pt")
            if os.path.isfile(candidate_state):
                training_state_path = candidate_state
        elif basename.startswith("critic") and basename.endswith(".pt"):
            critic_path = continued
            actor_path = os.path.join(dirname, basename.replace("critic", "actor", 1))
            suffix = basename[len("critic") : -3]
            candidate_state = os.path.join(dirname, "training_state" + suffix + ".pt")
            if os.path.isfile(candidate_state):
                training_state_path = candidate_state

    if training_state_path is not None:
        training_state = torch.load(training_state_path, map_location="cpu")
        actor_path = training_state.get("actor_path", actor_path)
        critic_path = training_state.get("critic_path", critic_path)
        resume_iteration = training_state.get("iteration")
    else:
        training_state = None
        resume_iteration = (
            parse_iteration_from_checkpoint(actor_path) if actor_path else None
        )

    if actor_path is None or critic_path is None:
        raise ValueError(
            "Could not resolve actor/critic checkpoint paths from --continued"
        )
    if not os.path.isfile(actor_path):
        raise FileNotFoundError("Actor checkpoint not found: {}".format(actor_path))
    if not os.path.isfile(critic_path):
        raise FileNotFoundError("Critic checkpoint not found: {}".format(critic_path))

    return {
        "actor_path": actor_path,
        "critic_path": critic_path,
        "training_state_path": training_state_path,
        "training_state": training_state,
        "resume_iteration": resume_iteration,
    }


def import_env(env_name_str):
    if env_name_str == "jvrc_walk":
        from envs.jvrc import JvrcWalkEnv as Env
    elif env_name_str == "jvrc_step":
        from envs.jvrc import JvrcStepEnv as Env
    elif env_name_str == "jvrc_run":
        from envs.jvrc import JvrcRunEnv as Env
    elif env_name_str == "jvrc_arm":
        from envs.jvrc import JvrcArmEnv as Env
    elif env_name_str == "jvrc_run_arm":
        from envs.jvrc import JvrcRunArmEnv as Env
    else:
        raise Exception("Check env name!")
    return Env


def run_experiment(args):
    # import the correct environment
    Env = import_env(args.env)

    # wrapper function for creating parallelized envs
    env_fn = partial(Env)
    if not args.no_mirror:
        try:
            print("Wrapping in SymmetricEnv.")
            env_fn = partial(
                SymmetricEnv,
                env_fn,
                mirrored_obs=env_fn().robot.mirrored_obs,
                mirrored_act=env_fn().robot.mirrored_acts,
                clock_inds=env_fn().robot.clock_inds,
            )
        except AttributeError as e:
            print("Warning! Cannot use SymmetricEnv.", e)
    obs_dim = env_fn().observation_space.shape[0]
    action_dim = env_fn().action_space.shape[0]

    # Set up Parallelism
    os.environ["OMP_NUM_THREADS"] = "1"
    if not ray.is_initialized():
        ray.init(num_cpus=args.num_procs)

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    resume_info = None
    if args.continued:
        resume_info = resolve_resume_paths(args.continued)
        policy = torch.load(resume_info["actor_path"], map_location="cpu")
        critic = torch.load(resume_info["critic_path"], map_location="cpu")
    else:
        policy = Gaussian_FF_Actor(
            obs_dim, action_dim, fixed_std=np.exp(args.std_dev), bounded=False
        )
        critic = FF_V(obs_dim)
        with torch.no_grad():
            policy.obs_mean, policy.obs_std = map(
                torch.Tensor,
                get_normalization_params(
                    iter=args.input_norm_steps,
                    noise_std=1,
                    policy=policy,
                    env_fn=env_fn,
                    procs=args.num_procs,
                ),
            )
        critic.obs_mean = policy.obs_mean
        critic.obs_std = policy.obs_std

    policy.train()
    critic.train()

    # dump hyperparameters
    os.makedirs(args.logdir, exist_ok=True)
    pkl_path = os.path.join(args.logdir, "experiment.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(args, f)

    algo = PPO(args=vars(args), save_path=args.logdir)
    algo.train(
        env_fn,
        policy,
        critic,
        args.n_itr,
        anneal_rate=args.anneal,
        resume_state=resume_info["training_state"] if resume_info else None,
        resume_iteration=resume_info["resume_iteration"] if resume_info else None,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    if sys.argv[1] != "train":
        raise Exception("Invalid usage.")

    sys.argv.remove(sys.argv[1])
    parser.add_argument(
        "--env", required=True, type=str
    )  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument(
        "--seed", default=0, type=int
    )  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument(
        "--logdir", type=str, default="./logs_dir/"
    )  # Where to log diagnostics to
    parser.add_argument("--input_norm_steps", type=int, default=100000)
    parser.add_argument(
        "--n_itr",
        type=int,
        default=2000,
        help="Number of iterations of the learning algorithm",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4, help="Adam learning rate"
    )  # Xie
    parser.add_argument(
        "--eps", type=float, default=1e-5, help="Adam epsilon (for numerical stability)"
    )
    parser.add_argument(
        "--lam",
        type=float,
        default=0.95,
        help="Generalized advantage estimate discount",
    )
    parser.add_argument("--gamma", type=float, default=0.99, help="MDP discount")
    parser.add_argument(
        "--anneal", default=1.0, action="store_true", help="anneal rate for stddev"
    )
    parser.add_argument(
        "--std_dev", type=int, default=-1.5, help="exponent of exploration std_dev"
    )
    parser.add_argument(
        "--entropy_coeff",
        type=float,
        default=0.0,
        help="Coefficient for entropy regularization",
    )
    parser.add_argument(
        "--clip",
        type=float,
        default=0.2,
        help="Clipping parameter for PPO surrogate loss",
    )
    parser.add_argument(
        "--minibatch_size", type=int, default=64, help="Batch size for PPO updates"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of optimization epochs per PPO update",
    )  # Xie
    parser.add_argument(
        "--use_gae",
        type=bool,
        default=True,
        help="Whether or not to calculate returns using Generalized Advantage Estimation",
    )
    parser.add_argument(
        "--num_procs", type=int, default=12, help="Number of threads to train on"
    )
    parser.add_argument(
        "--max_grad_norm", type=float, default=0.05, help="Value to clip gradients at."
    )
    parser.add_argument(
        "--max_traj_len", type=int, default=400, help="Max episode horizon"
    )
    parser.add_argument(
        "--no_mirror", required=False, action="store_true", help="to use SymmetricEnv"
    )
    parser.add_argument(
        "--mirror_coeff",
        required=False,
        default=0.4,
        type=float,
        help="weight for mirror loss",
    )
    parser.add_argument(
        "--eval_freq",
        required=False,
        default=100,
        type=int,
        help="Frequency of performing evaluation",
    )
    parser.add_argument(
        "--continued",
        required=False,
        default=None,
        type=str,
        help="path to pretrained weights",
    )
    parser.add_argument(
        "--use_wandb", action="store_true", help="Enable Weights & Biases logging"
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="LearningHumanoidRunning",
        help="wandb project name",
    )
    parser.add_argument(
        "--wandb_entity", type=str, default=None, help="wandb entity/team (optional)"
    )
    parser.add_argument(
        "--wandb_run_name", type=str, default=None, help="wandb run name (optional)"
    )
    parser.add_argument(
        "--wandb_video_freq",
        type=int,
        default=0,
        help="Log eval video every N evals (0 disables video)",
    )
    parser.add_argument(
        "--wandb_video_len",
        type=int,
        default=600,
        help="Max timesteps per logged eval video",
    )
    parser.add_argument(
        "--wandb_video_fps", type=int, default=30, help="FPS for logged eval video"
    )
    parser.add_argument(
        "--wandb_video_on_start",
        action="store_true",
        help="Run an evaluation and try to log a video before the first training iteration",
    )
    args = parser.parse_args()

    run_experiment(args)
