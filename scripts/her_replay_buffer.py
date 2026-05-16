"""
Hindsight Experience Replay (HER) Buffer for Goal-Conditioned RL.

Reference: Andrychowicz et al., "Hindsight Experience Replay", NeurIPS 2017
https://arxiv.org/abs/1707.01495

This module provides a replay buffer that implements the HER relabeling
strategy. HER enables learning from failed episodes by retroactively
replacing the desired goal with goals that the agent actually achieved.

YOUR TASK: Implement the methods marked with TODO below.

The buffer stores complete episodes and, at sample time, creates virtual
transitions where the desired goal is replaced with an achieved goal from
a future timestep in the same episode. This dramatically increases the
number of "successful" transitions the agent sees, which is critical for
sparse-reward goal-conditioned tasks like FetchPush.

Integration:
    The HERReplayBuffer is a drop-in replacement for CleanRL's ReplayBuffer.
    Its `sample()` method returns `ReplayBufferSamples` (same NamedTuple),
    so the training loop does not need to change.

Usage:
    from her_replay_buffer import HERReplayBuffer

    her_buffer = HERReplayBuffer(
        buffer_size=1_000_000,
        obs_dim=31,
        action_dim=4,
        goal_dim=3,
        compute_reward_fn=FetchPushFlatWrapper.compute_reward_static,
        reward_type="sparse",
        n_sampled_goal=4,
        strategy="future",
        device="cpu",
    )

    # During rollout, collect full episodes:
    episode = {"obs": [...], "action": [...], "next_obs": [...],
               "reward": [...], "done": [...]}
    her_buffer.store_episode(episode)

    # During training, sample with HER relabeling:
    batch = her_buffer.sample(batch_size=256)
    # batch.observations, batch.actions, etc. — same as ReplayBufferSamples
"""

from __future__ import annotations

from typing import Callable, NamedTuple

import numpy as np
import torch


class ReplayBufferSamples(NamedTuple):
    """Must match CleanRL's ReplayBufferSamples for compatibility."""
    observations: torch.Tensor
    actions: torch.Tensor
    next_observations: torch.Tensor
    dones: torch.Tensor
    rewards: torch.Tensor


class HERReplayBuffer:
    """
    Replay buffer with Hindsight Experience Replay (HER) goal relabeling.

    Key concepts:
        - Episodes are stored as complete trajectories (not individual transitions)
        - At sample time, for each sampled transition, with probability
          k/(k+1) we replace the desired goal with an achieved goal from
          a future timestep in the same episode ("future" strategy)
        - The reward is recomputed using the new goal via compute_reward_fn

    Observation layout (FetchPushFlat-v0, 31-dim):
        [0:25]  robot observation
        [25:28] desired_goal
        [28:31] achieved_goal

    Args:
        buffer_size: Maximum number of transitions to store
        obs_dim: Dimension of the flattened observation (default: 31)
        action_dim: Dimension of the action space (default: 4)
        goal_dim: Dimension of the goal space (default: 3)
        compute_reward_fn: Function(achieved_goal, desired_goal, reward_type) -> float
            Used to recompute rewards after goal relabeling.
            See FetchPushFlatWrapper.compute_reward_static
        reward_type: Reward type string passed to compute_reward_fn
        n_sampled_goal: Number of HER virtual goals per real transition (k).
            With k=4, ~80% of sampled transitions are HER-relabeled.
        strategy: Goal sampling strategy. Only "future" is required.
            "future": sample goal from achieved goals at timesteps t+1..T
                      in the same episode
        device: PyTorch device for returned tensors ("cpu" or "cuda")
    """

    def __init__(
        self,
        buffer_size: int,
        obs_dim: int = 31,
        action_dim: int = 4,
        goal_dim: int = 3,
        compute_reward_fn: Callable | None = None,
        reward_type: str = "sparse",
        n_sampled_goal: int = 4,
        strategy: str = "future",
        device: str = "cpu",
    ):
        assert strategy in ("future",), f"Only 'future' strategy is supported, got '{strategy}'"
        assert compute_reward_fn is not None, "compute_reward_fn is required"

        self.buffer_size = buffer_size
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.goal_dim = goal_dim
        self.compute_reward_fn = compute_reward_fn
        self.reward_type = reward_type
        self.n_sampled_goal = n_sampled_goal  # k in the paper
        self.strategy = strategy
        self.device = device

        # Goal indices within the flattened observation vector
        # FetchPushFlat-v0 layout: [robot_obs(25), desired_goal(3), achieved_goal(3)]
        self.desired_goal_start = obs_dim - 2 * goal_dim  # 25
        self.desired_goal_end = obs_dim - goal_dim          # 28
        self.achieved_goal_start = obs_dim - goal_dim       # 28
        self.achieved_goal_end = obs_dim                    # 31

        self.episodes = []  # List to store complete episodes as dicts
        self.total_transitions = 0  # Total number of transitions across all episodes

    def store_episode(self, episode: dict) -> None:
        """
        Store a complete episode in the buffer.

        Args:
            episode: Dict with keys:
                "obs":      np.ndarray of shape (T, obs_dim) — observations
                "action":   np.ndarray of shape (T, action_dim) — actions taken
                "next_obs": np.ndarray of shape (T, obs_dim) — next observations
                "reward":   np.ndarray of shape (T,) — rewards received
                "done":     np.ndarray of shape (T,) — done flags

                where T is the episode length.

        This method should:
            1. Append the episode to the internal episode storage
            2. Update the total transition count
            3. If the buffer exceeds buffer_size, remove oldest episodes (FIFO)
        """

        self.episodes.append(episode) # Append the new episode to the list of stored episodes
        self.total_transitions += len(episode["obs"])  # Update the total transition count

        # First In First Out, if buffer_size is exceeded.
        while self.total_transitions > self.buffer_size:
            oldest_episode = self.episodes.pop(0)
            self.total_transitions -= len(oldest_episode["obs"])

    def _sample_her_goals(
        self,
        episode_idx: int,
        transition_idx: int,
        n_goals: int,
    ) -> np.ndarray:
        """
        Sample n_goals virtual goals using the "future" strategy.

        For the "future" strategy:
            - Sample goal indices uniformly from {transition_idx + 1, ..., T-1}
              where T is the episode length
            - Return the achieved_goal at each sampled timestep

        Args:
            episode_idx: Index of the episode in the buffer
            transition_idx: Index of the transition within the episode
            n_goals: Number of goals to sample

        Returns:
            np.ndarray of shape (n_goals, goal_dim): Sampled goal positions

        Note:
            The achieved_goal at timestep t is the object's position AFTER
            taking the action at timestep t. For HER "future" strategy,
            we want goals from future timesteps, so we sample from the
            *next_obs* achieved_goal (i.e., the achieved goal after the
            transition).
        """
 
        episode = self.episodes[episode_idx] # Episode from storage
        T = len(episode["obs"])  # Episode length
        
        if transition_idx + 1 < T:
            sampled_indices = np.random.randint(transition_idx + 1, T, size=n_goals)  # Sample future indices unifromly
        else:
            sampled_indices = np.array([transition_idx] * n_goals)  # No future transitions available, use current index
        
        future_next_obs = episode["next_obs"][sampled_indices]  # Get next_obs for sampled indices
        achieved_goal = future_next_obs[:, self.achieved_goal_start:self.achieved_goal_end]  # Extract achieved_goal from next_obs
        return achieved_goal  # Return sampled goals

    def _recompute_reward(
        self,
        achieved_goal: np.ndarray,
        desired_goal: np.ndarray,
    ) -> np.ndarray:
        """
        Recompute the reward for relabeled transitions.

        After replacing the desired goal, the reward must be recomputed
        because the original reward was based on the original goal.

        Args:
            achieved_goal: np.ndarray of shape (batch_size, goal_dim)
                The achieved goal (object position) at the next timestep
            desired_goal: np.ndarray of shape (batch_size, goal_dim)
                The new desired goal (from HER relabeling)

        Returns:
            np.ndarray of shape (batch_size,): Recomputed rewards
        """
        return self.compute_reward_fn(achieved_goal, desired_goal, self.reward_type)

    def sample(self, batch_size: int) -> ReplayBufferSamples:
        """
        Sample a batch of transitions with HER goal relabeling.

        The sampling procedure:
            1. Sample batch_size transitions uniformly from stored episodes
            2. For each transition, with probability k/(k+1):
                a. Sample a virtual goal using _sample_her_goals (1 goal)
                b. Replace the desired_goal in both obs and next_obs
                c. Recompute the reward using _recompute_reward
            3. Return as ReplayBufferSamples (PyTorch tensors on self.device)

        Args:
            batch_size: Number of transitions to sample

        Returns:
            ReplayBufferSamples with fields:
                observations:      (batch_size, obs_dim) tensor
                actions:           (batch_size, action_dim) tensor
                next_observations: (batch_size, obs_dim) tensor
                dones:             (batch_size, 1) tensor
                rewards:           (batch_size, 1) tensor
        """

        obs = np.zeros((batch_size, self.obs_dim), dtype=np.float32) # Output observations
        actions = np.zeros((batch_size, self.action_dim), dtype=np.float32) # Output actions
        next_obs = np.zeros((batch_size, self.obs_dim), dtype=np.float32) # Output next observations
        rewards = np.zeros((batch_size, 1), dtype=np.float32) # Output rewards
        dones = np.zeros((batch_size, 1), dtype=np.float32) # Output dones

        p_relabel = self.n_sampled_goal / (self.n_sampled_goal + 1)  # Probability of relabeling (k/(k+1))
        num_episodes = len(self.episodes)  # Number of stored episodes

        for i in range(batch_size):
            episode_idx = np.random.randint(num_episodes)  # Randomly select an episode
            episode = self.episodes[episode_idx]  # Get selected episode

            T = len(episode["obs"])  # Length of the episode
            transition_idx = np.random.randint(T)  # Randomly select a transition index

            # To avoid corrupting raw buffer memory
            obs_i = episode["obs"][transition_idx].copy()  # Copy observation
            action_i = episode["action"][transition_idx].copy()  # Copy action
            next_obs_i = episode["next_obs"][transition_idx].copy()  # Copy next observation
            reward_i = episode["reward"][transition_idx].copy()  # Copy reward
            done_i = episode["done"][transition_idx].copy()  # Copy done flag

            if np.random.rand() < p_relabel:  # With probability k/(k+1), apply HER relabeling
                # Sample a new goal using the "future" strategy
                future_goal = self._sample_her_goals(episode_idx, transition_idx, n_goals=1)[0]  # Get one new goal

                # Replace desired_goal in obs and next_obs
                obs_i[self.desired_goal_start:self.desired_goal_end] = future_goal  # Update desired_goal in obs
                next_obs_i[self.desired_goal_start:self.desired_goal_end] = future_goal  # Update desired_goal in next_obs

                # Recompute reward with the new desired goal
                achieved_goal_next = next_obs_i[self.achieved_goal_start:self.achieved_goal_end]  # Extract achieved_goal from next_obs
                raw_reward = self._recompute_reward(achieved_goal_next, future_goal)  # Recompute reward
                reward_i = float(raw_reward)  # Ensure reward is a scalar float. Got an error when using np.float32 directly in the reward tensor
            
            #Store in output arrays
            obs[i] = obs_i
            actions[i] = action_i 
            next_obs[i] = next_obs_i
            rewards[i] = reward_i
            dones[i] = done_i
    
        return ReplayBufferSamples(
            observations=torch.as_tensor(obs, device=self.device),
            actions=torch.as_tensor(actions, device=self.device),
            next_observations=torch.as_tensor(next_obs, device=self.device),
            dones=torch.as_tensor(dones, device=self.device),
            rewards=torch.as_tensor(rewards, device=self.device)
        )

    def __len__(self) -> int:
        """Return total number of transitions stored."""
        return self.total_transitions