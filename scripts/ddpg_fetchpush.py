# DDPG (Deep Deterministic Policy Gradient) adapted for FetchPushFlat-v0
#
# Based on CleanRL's ddpg_continuous_action.py
# Original: https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ddpg_continuous_action.py
# Docs: https://docs.cleanrl.dev/rl-algorithms/ddpg/#ddpg_continuous_actionpy
#
# Modifications for this assessment:
#   - Pre-configured for FetchPushFlat-v0 environment
#   - Added --reward-type argument for reward function selection
#   - Added --her flag for Hindsight Experience Replay integration
#   - Model saving enabled by default
#
# Usage:
#   # Baseline (without HER):
#   python scripts/ddpg_fetchpush.py --reward-type sparse --total-timesteps 250000
#
#   # With HER (after implementing her_replay_buffer.py):
#   python scripts/ddpg_fetchpush.py --reward-type sparse --her --total-timesteps 250000
#
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter

from cleanrl_utils.buffers import ReplayBuffer

# Register FetchPush custom environment
from fetch_push_env import FetchPushFlatWrapper, register_fetch_push_envs
register_fetch_push_envs()


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "rl-fetch-push"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = True
    """whether to save model into the `runs/{run_name}` folder"""

    # Environment arguments
    env_id: str = "FetchPushFlat-v0"
    """the environment id"""
    reward_type: str = "sparse"
    """reward type for FetchPush (sparse, dense_basic, or your custom types)"""

    # HER arguments
    her: bool = False
    """enable Hindsight Experience Replay"""
    gradient_steps: int = 1
    """number of gradient updates per env step (increase for HER, e.g. 4-40)"""

    # Algorithm specific arguments
    total_timesteps: int = 250000
    """total timesteps of the experiments"""
    learning_rate: float = 1e-3
    """the learning rate of the optimizer"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.95
    """the discount factor gamma"""
    tau: float = 0.05
    """target smoothing coefficient"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    exploration_noise: float = 0.2
    """the scale of exploration noise (std of Gaussian noise added to actions)"""
    random_eps: float = 0.3
    """probability of taking a completely random action (epsilon-greedy exploration)"""
    learning_starts: int = 5000
    """timestep to start learning"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""


def make_env(env_id, seed, idx, capture_video, run_name, env_kwargs=None):
    def thunk():
        kw = env_kwargs or {}
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array", **kw)
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id, **kw)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env

    return thunk


# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod() + np.prod(env.single_action_space.shape), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mu = nn.Linear(256, np.prod(env.single_action_space.shape))
        # action rescaling
        self.register_buffer(
            "action_scale", torch.tensor((env.action_space.high - env.action_space.low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((env.action_space.high + env.action_space.low) / 2.0, dtype=torch.float32)
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc_mu(x))
        return x * self.action_scale + self.action_bias


if __name__ == "__main__":
    args = tyro.cli(Args)
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    env_kwargs = {"reward_type": args.reward_type}
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed, 0, args.capture_video, run_name, env_kwargs=env_kwargs)],
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    actor = Actor(envs).to(device)
    qf1 = QNetwork(envs).to(device)
    qf1_target = QNetwork(envs).to(device)
    target_actor = Actor(envs).to(device)
    target_actor.load_state_dict(actor.state_dict())
    qf1_target.load_state_dict(qf1.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()), lr=args.learning_rate)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.learning_rate)

    envs.single_observation_space.dtype = np.float32

    # =========================================================================
    # Replay buffer setup
    #
    # If --her is enabled, use HERReplayBuffer instead of the standard
    # ReplayBuffer. You need to:
    #   1. Import HERReplayBuffer from her_replay_buffer
    #   2. Instantiate it with the correct parameters
    #   3. Collect full episodes and call her_buffer.store_episode()
    #   4. Sample from her_buffer.sample() during training
    #
    # If --her is NOT enabled, the standard ReplayBuffer is used as-is.
    # =========================================================================
    if args.her:
        from her_replay_buffer import HERReplayBuffer
        rb = HERReplayBuffer(
            buffer_size=args.buffer_size,
            obs_dim=np.array(envs.single_observation_space.shape).prod(),
            action_dim=np.prod(envs.single_action_space.shape),
            goal_dim=3, # Goal represented as 3D position. Flattened obs.
            compute_reward_fn=FetchPushFlatWrapper.compute_reward_static,
            reward_type=args.reward_type,
            device=device,
        )

    else:
        rb = ReplayBuffer(
            args.buffer_size,
            envs.single_observation_space,
            envs.single_action_space,
            device,
            handle_timeout_termination=False,
        )

    start_time = time.time()

    # =========================================================================
    # Episode collection for HER
    #
    # When using HER, you need to collect complete episodes before storing
    # them in the buffer. The episode_buffer below accumulates transitions
    # for the current episode. On episode end, store_episode() is called.
    #
    # When NOT using HER, transitions are added to the replay buffer
    # immediately (standard approach).
    # =========================================================================
    if args.her:
        episode_buffer = {
            "obs": [], "action": [], "next_obs": [],
            "reward": [], "done": [],
        }

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            # Epsilon-greedy: with probability random_eps, take a random action
            if np.random.random() < args.random_eps:
                actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
            else:
                with torch.no_grad():
                    actions = actor(torch.Tensor(obs).to(device))
                    actions += torch.normal(0, actor.action_scale * args.exploration_noise)
                    actions = actions.cpu().numpy().clip(envs.single_action_space.low, envs.single_action_space.high)

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            fi = infos["final_info"]
            if isinstance(fi, dict) and "episode" in fi:
                # gymnasium >= 1.0 SAME_STEP mode: fi is a vectorized dict
                print(f"global_step={global_step}, episodic_return={fi['episode']['r'][0]:.2f}")
                writer.add_scalar("charts/episodic_return", fi["episode"]["r"][0], global_step)
                writer.add_scalar("charts/episodic_length", fi["episode"]["l"][0], global_step)
            elif isinstance(fi, (list, np.ndarray)):
                # gymnasium < 1.0: fi is a list of per-env info dicts
                for info in fi:
                    if info is not None and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                        break

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_obs`
        # (gymnasium >= 1.0 uses "final_obs" key with SAME_STEP autoreset mode)
        real_next_obs = next_obs.copy()
        for final_obs_key in ("final_obs", "final_observation"):
            if final_obs_key in infos:
                for idx, trunc in enumerate(truncations):
                    if trunc and infos[final_obs_key][idx] is not None:
                        real_next_obs[idx] = infos[final_obs_key][idx]
                break

        if args.her:
            # HER: accumulate transitions in episode buffer
            episode_buffer["obs"].append(obs[0])
            episode_buffer["action"].append(actions[0])
            episode_buffer["next_obs"].append(real_next_obs[0])
            episode_buffer["reward"].append(rewards[0])
            episode_buffer["done"].append(terminations[0])

            # On episode end (termination or truncation), call:
            if terminations[0] or truncations[0]:
                rb.store_episode({
                    "obs": np.array(episode_buffer["obs"]),
                   "action": np.array(episode_buffer["action"]),
                   "next_obs": np.array(episode_buffer["next_obs"]),
                   "reward": np.array(episode_buffer["reward"]),
                   "done": np.array(episode_buffer["done"]),
               })
                # Reset episode_buffer for the next episode.
                episode_buffer = {
                    "obs": [], "action": [], "next_obs": [], "reward": [], "done": [],
                    }
        else:
            rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
          for _grad_step in range(args.gradient_steps):
            data = rb.sample(args.batch_size)
            with torch.no_grad():
                next_state_actions = target_actor(data.next_observations)
                qf1_next_target = qf1_target(data.next_observations, next_state_actions)
                next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (qf1_next_target).view(-1)

            qf1_a_values = qf1(data.observations, data.actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)

            # optimize the model
            q_optimizer.zero_grad()
            qf1_loss.backward()
            q_optimizer.step()

            if global_step % args.policy_frequency == 0:
                actor_loss = -qf1(data.observations, actor(data.observations)).mean()
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

            # update the target network
            for param, target_param in zip(actor.parameters(), target_actor.parameters()):
                target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
            for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            if global_step % 100 == 0:
                writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save((actor.state_dict(), qf1.state_dict()), model_path)
        print(f"model saved to {model_path}")

    envs.close()
    writer.close()
