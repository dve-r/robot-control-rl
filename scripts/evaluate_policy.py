#!/usr/bin/env python3
"""
Evaluation script for trained RL policies on FetchPushFlat-v0.

Loads a trained CleanRL model, runs evaluation episodes, computes metrics,
and saves results in the required JSON format.

Usage:
    python scripts/evaluate_policy.py \
        --model-path runs/<run-name>/sac_fetchpush.cleanrl_model \
        --env-id FetchPushFlat-v0 \
        --reward-type sparse \
        --algorithm SAC \
        --n-episodes 100 \
        --output results/sac_her_results.json \
        --record-video \
        --video-dir videos/

NOTE: This is a helper script. Candidates may evaluate differently.
What matters is the output JSON format.
"""

import argparse
import json
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Register custom env
from scripts.fetch_push_env import register_fetch_push_envs
register_fetch_push_envs()

# SAC Constants
LOG_STD_MAX = 2
LOG_STD_MIN = -5

class SACActor(nn.Module):
    def __init__(self, env):
        super().__init__()
        obs_space = env.single_observation_space if hasattr(env, "single_observation_space") else env.observation_space
        act_space = env.single_action_space if hasattr(env, "single_action_space") else env.action_space
        
        self.fc1 = nn.Linear(np.array(obs_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, np.prod(act_space.shape))
        self.fc_logstd = nn.Linear(256, np.prod(act_space.shape))
        
        self.register_buffer("action_scale", torch.tensor((act_space.high - act_space.low) / 2.0, dtype=torch.float32))
        self.register_buffer("action_bias", torch.tensor((act_space.high + act_space.low) / 2.0, dtype=torch.float32))

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean


class DDPGActor(nn.Module):
    def __init__(self, env):
        super().__init__()
        obs_space = env.single_observation_space if hasattr(env, "single_observation_space") else env.observation_space
        act_space = env.single_action_space if hasattr(env, "single_action_space") else env.action_space
        
        self.fc1 = nn.Linear(np.array(obs_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mu = nn.Linear(256, np.prod(act_space.shape))
        
        self.register_buffer("action_scale", torch.tensor((act_space.high - act_space.low) / 2.0, dtype=torch.float32))
        self.register_buffer("action_bias", torch.tensor((act_space.high + act_space.low) / 2.0, dtype=torch.float32))

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc_mu(x))
        return x * self.action_scale + self.action_bias


def load_cleanrl_model(model_path: str, env: gym.Env, algorithm: str):
    """Loads the CleanRL state_dict into the correct Actor based on algorithm."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Pick the correct architecture
    if algorithm.upper() == "SAC":
        actor = SACActor(env).to(device)
    elif algorithm.upper() == "DDPG":
        actor = DDPGActor(env).to(device)
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}. Must be SAC or DDPG.")
    
    # 2. Load the file
    checkpoint = torch.load(model_path, map_location=device)
    
    # 3. Smart extraction: Handle Tuples, Lists, and Dictionaries
    if isinstance(checkpoint, (tuple, list)):
        actor_state_dict = checkpoint[0] 
    elif isinstance(checkpoint, dict):
        if "actor" in checkpoint:
            actor_state_dict = checkpoint["actor"]
        elif "model" in checkpoint:
            actor_state_dict = checkpoint["model"]
        else:
            actor_state_dict = checkpoint
    else:
        actor_state_dict = checkpoint

    # 3.5 Fix shape mismatches for action scale/bias (DDPG [1, 4] vs [4] issue)
    for key in ["action_scale", "action_bias"]:
        if key in actor_state_dict and len(actor_state_dict[key].shape) > 1:
            actor_state_dict[key] = actor_state_dict[key].squeeze(0)
    
    # 4. Load into our network
    actor.load_state_dict(actor_state_dict)
    actor.eval()
    
    return actor

def evaluate(
    model,
    env_id: str,
    n_episodes: int = 100,
    record_video: bool = False,
    video_dir: str = "videos/",
    seed: int = 42,
    **env_kwargs,
):
    """Run evaluation episodes and collect metrics."""
    if record_video:
        Path(video_dir).mkdir(parents=True, exist_ok=True)
        env = gym.make(env_id, render_mode="rgb_array", **env_kwargs)
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=video_dir,
            episode_trigger=lambda ep: ep < 10,  # Record first 10 episodes
        )
    else:
        env = gym.make(env_id, **env_kwargs)

    successes = []
    episode_returns = []
    episode_lengths = []
    episode_energies = []

    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        done = False
        ep_return = 0.0
        ep_length = 0
        ep_energy = 0.0

        while not done:
            # Get action from model
            with torch.no_grad():
                # 1. Dynamically check which device the model is sitting on
                device = next(model.parameters()).device
                
                # 2. Push the observation to that same device
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
                
                # 3. Get the action and push it back to the CPU for numpy/gym
                if hasattr(model, "get_action"):
                    action, _, _ = model.get_action(obs_tensor)
                    action = action.squeeze(0).cpu().numpy()
                elif hasattr(model, "actor"):
                    action = model.actor(obs_tensor).squeeze(0).cpu().numpy()
                else:
                    # Fallback: try calling the model directly
                    action = model(obs_tensor).squeeze(0).cpu().numpy()

            obs, reward, terminated, truncated, info = env.step(action)
            ep_return += reward
            ep_length += 1
            ep_energy += np.sum(action[:3] ** 2)  # L2 norm of end-effector actions
            done = terminated or truncated

        successes.append(float(info.get("is_success", False)))
        episode_returns.append(ep_return)
        episode_lengths.append(ep_length)
        episode_energies.append(ep_energy / max(ep_length, 1))

    env.close()

    return {
        "success_rate": float(np.mean(successes)),
        "mean_episode_return": float(np.mean(episode_returns)),
        "std_episode_return": float(np.std(episode_returns)),
        "mean_episode_length": float(np.mean(episode_lengths)),
        "mean_energy": float(np.mean(episode_energies)),
        "std_energy": float(np.std(episode_energies)),
        "n_episodes": n_episodes,
        "per_episode_success": successes,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate RL policy on FetchPushFlat")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--env-id", type=str, default="FetchPushFlat-v0")
    parser.add_argument("--reward-type", type=str, default="sparse")
    parser.add_argument("--n-episodes", type=int, default=100)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--record-video", action="store_true")
    parser.add_argument("--video-dir", type=str, default="videos/")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--algorithm", type=str, default="SAC")
    parser.add_argument("--total-timesteps", type=int, default=0)
    parser.add_argument("--training-wall-time", type=float, default=0.0)
    parser.add_argument("--hardware", type=str, default="unknown")
    # Domain randomization eval params
    parser.add_argument("--object-mass-multiplier", type=float, default=1.0)
    parser.add_argument("--friction-multiplier", type=float, default=1.0)
    parser.add_argument("--object-size-multiplier", type=float, default=1.0)
    args = parser.parse_args()

    env_kwargs = {
        "reward_type": args.reward_type,
        "object_mass_multiplier": args.object_mass_multiplier,
        "friction_multiplier": args.friction_multiplier,
        "object_size_multiplier": args.object_size_multiplier,
    }

    print(f"Loading model from: {args.model_path}")
    dummy_env = gym.make(args.env_id, **env_kwargs)
    model = load_cleanrl_model(args.model_path, dummy_env, args.algorithm)
    dummy_env.close()

    print(f"Evaluating for {args.n_episodes} episodes...")
    start = time.time()
    metrics = evaluate(
        model,
        args.env_id,
        n_episodes=args.n_episodes,
        record_video=args.record_video,
        video_dir=args.video_dir,
        seed=args.seed,
        **env_kwargs,
    )
    eval_time = time.time() - start

    results = {
        "experiment": f"reward_{args.reward_type}_{args.algorithm.lower()}",
        "algorithm": args.algorithm,
        "reward_type": args.reward_type,
        "env_id": args.env_id,
        "success_rate": metrics["success_rate"],
        "mean_episode_return": metrics["mean_episode_return"],
        "mean_episode_length": metrics["mean_episode_length"],
        "mean_energy": metrics["mean_energy"],
        "total_timesteps": args.total_timesteps,
        "training_wall_time_minutes": args.training_wall_time,
        "n_eval_episodes": args.n_episodes,
        "domain_randomization": {
            "object_mass_multiplier": args.object_mass_multiplier,
            "friction_multiplier": args.friction_multiplier,
            "object_size_multiplier": args.object_size_multiplier,
        },
        "hardware": args.hardware,
        "seed": args.seed,
        "notes": "",
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")
    print(f"Success rate: {metrics['success_rate']:.1%}")
    print(f"Mean return: {metrics['mean_episode_return']:.2f}")
    print(f"Mean energy: {metrics['mean_energy']:.4f}")
    print(f"Eval time: {eval_time:.1f}s")


if __name__ == "__main__":
    main()
