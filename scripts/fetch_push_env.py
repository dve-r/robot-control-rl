"""
FetchPush Flat Wrapper — Flattened observation + custom rewards + domain randomization.

This module provides a Gymnasium wrapper around FetchPush-v4 that:
1. Flattens the Dict observation space into a single Box vector
2. Supports configurable reward functions (add your own!)
3. Supports domain randomization for robustness training

Usage:
    from scripts.fetch_push_env import register_fetch_push_envs
    register_fetch_push_envs()

    import gymnasium as gym
    env = gym.make("FetchPushFlat-v0", reward_type="dense_basic")
"""

import gymnasium as gym
import gymnasium_robotics
import numpy as np
from gymnasium import spaces

# Ensure gymnasium-robotics envs are registered
gym.register_envs(gymnasium_robotics)


class FetchPushFlatWrapper(gym.Wrapper):
    """
    Wrapper that flattens FetchPush-v4's Dict observation space into a single
    Box observation, adds configurable reward functions, and supports domain
    randomization.

    Observation layout (31-dim):
        [0:25]  - robot observation (joint positions, velocities, gripper state,
                  object position, object velocity, relative positions)
        [25:28] - desired goal (x, y, z)
        [28:31] - achieved goal (x, y, z) — current object position

    Action space (4-dim):
        [0:3]   - end-effector velocity (dx, dy, dz)
        [3]     - gripper command (not used for pushing, but available)
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 25}

    REWARD_TYPES = ["sparse", "dense_basic", "multi_component", "dense_energy" "approach_progress"]
    # ^^^ CANDIDATES: Add your custom reward types to this list when you
    # implement them. For example:
    #   REWARD_TYPES = ["sparse", "dense_basic", "multi_component", "potential_based"]

    def __init__(
        self,
        reward_type="sparse",
        render_mode=None,
        # Domain randomization parameters
        randomize=False,
        object_mass_multiplier=1.0,
        friction_multiplier=1.0,
        object_size_multiplier=1.0,
        mass_range=None,
        friction_range=None,
        size_range=None,
        success_threshold=0.05,
    ):
        # Create the base FetchPush environment
        make_kwargs = {"max_episode_steps": 50}
        if render_mode is not None:
            make_kwargs["render_mode"] = render_mode
        base_env = gym.make("FetchPush-v4", **make_kwargs)
        super().__init__(base_env)

        self.reward_type = reward_type
        self.success_threshold = success_threshold

        # Domain randomization config
        self.randomize = randomize
        self.object_mass_multiplier = object_mass_multiplier
        self.friction_multiplier = friction_multiplier
        self.object_size_multiplier = object_size_multiplier
        self.mass_range = mass_range or [1.0, 1.0]
        self.friction_range = friction_range or [1.0, 1.0]
        self.size_range = size_range or [1.0, 1.0]

        # Store nominal physics parameters (will be set after first reset)
        self._nominal_params_saved = False
        self._nominal_object_mass = None
        self._nominal_friction = None
        self._nominal_object_size = None

        # Flatten observation space: observation + desired_goal + achieved_goal
        obs_sample, _ = base_env.reset()
        obs_dim = (
            obs_sample["observation"].shape[0]
            + obs_sample["desired_goal"].shape[0]
            + obs_sample["achieved_goal"].shape[0]
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float64
        )

        # For reward computation: track previous state
        self._prev_obs = None
        self._prev_distance = None
        self._step_count = 0

    def _flatten_obs(self, obs_dict):
        """Flatten Dict observation into a single vector."""
        return np.concatenate([
            obs_dict["observation"],
            obs_dict["desired_goal"],
            obs_dict["achieved_goal"],
        ])

    def _save_nominal_params(self):
        """Save the default physics parameters from the MuJoCo model."""
        model = self.unwrapped.model
        # Object body index — FetchPush calls it "object0"
        try:
            obj_body_id = model.body("object0").id
            self._nominal_object_mass = model.body_mass[obj_body_id].copy()

            # Friction: geom associated with the object
            obj_geom_id = model.geom("object0").id
            self._nominal_friction = model.geom_friction[obj_geom_id].copy()

            # Object size
            self._nominal_object_size = model.geom_size[obj_geom_id].copy()
        except Exception:
            # Fallback if names don't match
            self._nominal_object_mass = None
            self._nominal_friction = None
            self._nominal_object_size = None

        self._nominal_params_saved = True

    def _apply_domain_randomization(self):
        """Randomize physics parameters at the start of each episode."""
        if not self._nominal_params_saved or self._nominal_object_mass is None:
            return

        model = self.unwrapped.model

        try:
            obj_body_id = model.body("object0").id
            obj_geom_id = model.geom("object0").id

            if self.randomize:
                mass_mult = np.random.uniform(*self.mass_range)
                friction_mult = np.random.uniform(*self.friction_range)
                size_mult = np.random.uniform(*self.size_range)
            else:
                mass_mult = self.object_mass_multiplier
                friction_mult = self.friction_multiplier
                size_mult = self.object_size_multiplier

            model.body_mass[obj_body_id] = self._nominal_object_mass * mass_mult
            model.geom_friction[obj_geom_id] = self._nominal_friction * friction_mult
            model.geom_size[obj_geom_id] = self._nominal_object_size * size_mult
        except Exception:
            pass  # Silently skip if parameter modification fails

    def reset(self, **kwargs):
        obs_dict, info = self.env.reset(**kwargs)

        if not self._nominal_params_saved:
            self._save_nominal_params()

        self._apply_domain_randomization()

        flat_obs = self._flatten_obs(obs_dict)
        self._prev_obs = obs_dict
        self._prev_distance = np.linalg.norm(
            obs_dict["achieved_goal"] - obs_dict["desired_goal"]
        )
        self._step_count = 0

        # Add goal info for evaluation
        info["is_success"] = False
        info["desired_goal"] = obs_dict["desired_goal"]
        info["achieved_goal"] = obs_dict["achieved_goal"]

        return flat_obs, info

    def step(self, action):
        obs_dict, _, terminated, truncated, info = self.env.step(action)
        self._step_count += 1

        flat_obs = self._flatten_obs(obs_dict)

        # Compute success
        distance = np.linalg.norm(
            obs_dict["achieved_goal"] - obs_dict["desired_goal"]
        )
        is_success = distance < self.success_threshold
        info["is_success"] = is_success
        info["distance_to_goal"] = distance
        info["desired_goal"] = obs_dict["desired_goal"]
        info["achieved_goal"] = obs_dict["achieved_goal"]

        # Compute reward based on selected type
        reward = self._compute_reward(obs_dict, action, distance, is_success)

        self._prev_obs = obs_dict
        self._prev_distance = distance

        return flat_obs, reward, terminated, truncated, info

    def _compute_reward(self, obs_dict, action, distance, is_success):
        """
        Compute reward based on the selected reward type.

        CANDIDATES: This is where you implement your custom reward functions.
        Add new elif branches for your reward types.

        Available information:
            obs_dict: Dict with keys 'observation', 'achieved_goal', 'desired_goal'
            action: The action taken (4-dim)
            distance: L2 distance from object to goal
            is_success: True if distance < threshold
            self._prev_distance: Distance at previous timestep
            self._prev_obs: Previous observation dict
            self._step_count: Current step in episode
        """

        if self.reward_type == "sparse":
            # Default sparse reward: 0 if success, -1 otherwise
            return 0.0 if is_success else -1.0

        elif self.reward_type == "dense_basic":
            # Basic dense reward: negative L2 distance to goal
            # Closer to goal = higher (less negative) reward
            return -distance

        # =====================================================================
        # CANDIDATES: Add your custom reward functions below.
        #
        # Available variables for reward computation:
        #   - distance:          L2 distance from object to goal
        #   - self._prev_distance: distance at previous timestep
        #   - is_success:        True if distance < threshold
        #   - action:            4-dim action vector [dx, dy, dz, gripper]
        #   - self._step_count:  current step in episode (0 to 49)
        #   - obs_dict["observation"]:    25-dim (includes gripper pos, object pos, etc.)
        #   - obs_dict["achieved_goal"]:  3-dim (current object position)
        #   - obs_dict["desired_goal"]:   3-dim (target position)
        #   - self._prev_obs:    previous observation dict
        #
        # Useful quantities you can compute:
        #   gripper_pos = obs_dict["observation"][:3]
        #   object_pos  = obs_dict["achieved_goal"]
        #   goal_pos    = obs_dict["desired_goal"]
        #   gripper_to_object = np.linalg.norm(gripper_pos - object_pos)
        #   object_to_goal_direction = (goal_pos - object_pos) / (distance + 1e-6)
        #   object_velocity = obs_dict["observation"][...]  # check indices
        #   action_energy = np.sum(action[:3] ** 2)
        #
        # =====================================================================

        elif self.reward_type == "dense_energy":
            # Objective 1 & 3: Dense distance with Energy efficiency
            dense_reward = -distance
            # Penalize the L2 norm of the action (scaled by 0.1)
            energy_penalty = -0.1 * np.sum(action[:3] ** 2)
            return dense_reward + energy_penalty
        
        elif self.reward_type == "approach_progress":
            # Objective 2 & 4: Approach + push + Progress bonus
            gripper_pos = obs_dict["observation"][:3]
            object_pos = obs_dict["achieved_goal"]
            
            # Approach the object
            gripper_to_object = np.linalg.norm(gripper_pos - object_pos)
            approach_reward = -gripper_to_object
            
            # Push the object to the goal
            push_reward = -distance
            
            # Progress bonus: reward for reducing object-goal distance compared to previous step
            progress = self._prev_distance - distance
            progress_reward = 10.0 * progress # Scaled to make progress highly rewarding
            
            return approach_reward + push_reward + progress_reward

        elif self.reward_type == "multi_component":
            # Multi-component reward with approach + push phases
            gripper_pos = obs_dict["observation"][:3]
            object_pos = obs_dict["achieved_goal"]
            goal_pos = obs_dict["desired_goal"]

            gripper_to_object = np.linalg.norm(gripper_pos - object_pos)

            # Phase 1: Approach — reward gripper getting close to object
            # This helps the agent learn to reach the object first
            approach_reward = -gripper_to_object

            # Phase 2: Push — reward object moving toward goal
            # Weight increases as gripper gets closer to object
            push_weight = max(0, 1.0 - gripper_to_object / 0.1)
            push_reward = -distance * (1.0 + 2.0 * push_weight)

            # Progress bonus: reward for reducing object-goal distance
            progress = self._prev_distance - distance
            progress_reward = 20.0 * progress

            # Time penalty: encourage faster completion
            time_penalty = -0.02

            # Energy penalty: discourage wasteful actions
            energy_penalty = -0.05 * np.sum(action[:3] ** 2)

            # Success bonus: large reward for task completion
            success_bonus = 50.0 if is_success else 0.0

            return (
                approach_reward
                + push_reward
                + progress_reward
                + time_penalty
                + energy_penalty
                + success_bonus
            )

        else:
            raise ValueError(
                f"Unknown reward_type: '{self.reward_type}'. "
                f"Available types: {self.REWARD_TYPES}"
            )


    # -----------------------------------------------------------------
    # Static helpers for HER (Hindsight Experience Replay)
    # -----------------------------------------------------------------

    @staticmethod
    def compute_reward_static(
        achieved_goal: np.ndarray,
        desired_goal: np.ndarray,
        reward_type: str = "sparse",
        threshold: float = 0.05,
    ) -> float:
        """
        Compute reward without needing an env instance.

        This is used by HER to recompute rewards after goal relabeling.
        When HER replaces the desired goal with a virtual goal, the reward
        must be recalculated based on the new goal.

        Args:
            achieved_goal: np.ndarray of shape (3,) — object position
            desired_goal: np.ndarray of shape (3,) — target position
            reward_type: one of "sparse", "dense_basic", or your custom types
            threshold: success distance threshold (default 0.05m)

        Returns:
            float: the reward value
        """
        distance = np.linalg.norm(achieved_goal - desired_goal, axis=-1)

        if reward_type == "sparse":
            return 0.0 if distance < threshold else -1.0
        elif reward_type == "dense_basic":
            return -distance
        elif reward_type == "dense_energy":
            # For HER relabeling, no have 'action' to calculate energy.
            # Fall back to the basic dense distance.
            return -distance
            
        elif reward_type == "approach_progress":
            # For HER relabeling, no have 'gripper_pos' or previous steps.
            # Fall back to the basic dense distance.
            return -distance

        else:
            raise ValueError(
                f"Unknown reward_type for static computation: '{reward_type}'. "
                f"Add your custom reward type here or use 'sparse' for HER."
            )

    @staticmethod
    def get_goal_from_obs(obs: np.ndarray, goal_dim: int = 3):
        """
        Extract achieved_goal and desired_goal from a flattened observation.

        Observation layout (31-dim):
            [0:25]  robot observation
            [25:28] desired_goal (x, y, z)
            [28:31] achieved_goal (x, y, z)

        Args:
            obs: flattened observation vector of shape (31,)
            goal_dim: dimension of each goal vector (default 3)

        Returns:
            tuple: (achieved_goal, desired_goal) each of shape (goal_dim,)
        """
        desired_goal = obs[-(2 * goal_dim):-goal_dim]
        achieved_goal = obs[-goal_dim:]
        return achieved_goal, desired_goal


def register_fetch_push_envs():
    """Register the custom FetchPush environments with Gymnasium."""
    gym.register(
        id="FetchPushFlat-v0",
        entry_point="scripts.fetch_push_env:FetchPushFlatWrapper",
        kwargs={"reward_type": "sparse"},
        max_episode_steps=50,
    )


# Allow running this file directly for testing
if __name__ == "__main__":
    register_fetch_push_envs()

    print("=" * 50)
    print("Testing FetchPushFlat-v0")
    print("=" * 50)

    for reward_type in ["sparse", "dense_basic"]:
        env = gym.make("FetchPushFlat-v0", reward_type=reward_type)
        obs, info = env.reset()
        print(f"\nReward type: {reward_type}")
        print(f"  Observation shape: {obs.shape}")
        print(f"  Action space: {env.action_space}")

        total_reward = 0
        successes = 0
        for ep in range(10):
            obs, info = env.reset()
            ep_reward = 0
            for _ in range(50):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                ep_reward += reward
                if terminated or truncated:
                    break
            total_reward += ep_reward
            if info.get("is_success", False):
                successes += 1

        print(f"  Mean return (10 random eps): {total_reward / 10:.2f}")
        print(f"  Successes: {successes}/10")
        env.close()

    # Test domain randomization
    print(f"\nTesting domain randomization:")
    env = gym.make(
        "FetchPushFlat-v0",
        reward_type="dense_basic",
        randomize=True,
        mass_range=[0.5, 2.0],
        friction_range=[0.5, 2.0],
        size_range = [0.8, 1.2],
    )
    obs, info = env.reset()
    print(f"  DR environment created successfully")
    print(f"  Observation shape: {obs.shape}")
    env.close()

    print("\nAll tests passed!")
