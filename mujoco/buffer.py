import os
import torch
import pickle
import numpy as np
from dotmap import DotMap  
from drop_fn import create_drop_function  


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_discounted_cumsum(rewards, discount_factor):
    """
    Computes discounted cumulative sum of rewards.

    Args:
        rewards (np.ndarray): Array of rewards.
        discount_factor (float): Discount rate (gamma).

    Returns:
        np.ndarray: Discounted cumulative sum array.
    """
    result = np.zeros_like(rewards)
    for t in range(len(rewards) - 1, -1, -1):
        result[t] = rewards[t] + (result[t + 1] * discount_factor if t + 1 < len(rewards) else 0)
    return result


class SequenceBuffer:
    def __init__(self, env_name, dataset, context_length, root_dir, gamma, drop_config, sample_mode='traj_length', seed=42):
        """
        Initializes the Sequence Buffer.

        Args:
            env_name (str): Environment name.
            dataset (str): Dataset identifier.
            context_length (int): Length of the context window.
            root_dir (str): Root directory for datasets.
            gamma (float): Discount factor.
            drop_config (dict): Configuration for the drop function.
            sample_mode (str): Sampling mode (default is 'traj_length').
            seed (int): Random seed.
        """
        dataset_file = os.path.join(root_dir, f"{env_name.lower()}-{dataset}.pkl")
        if not os.path.isfile(dataset_file):
            raise FileNotFoundError(f"Dataset not found: {dataset_file}")

        with open(dataset_file, "rb") as file:
            self.trajectories = pickle.load(file)

        self.context_length = context_length
        self.num_trajs = len(self.trajectories)
        self.gamma = gamma
        self.sample_mode = sample_mode
        self.random_generator = np.random.default_rng(seed)

        first_traj = self.trajectories[0]
        self.state_dim = first_traj["observations"].shape[1]
        self.action_dim = first_traj["actions"].shape[1]

        total_frames = sum(len(traj["observations"]) for traj in self.trajectories) + 1
        self.states = np.zeros((total_frames, self.state_dim), dtype=np.float32)
        self.actions = np.zeros((total_frames, self.action_dim), dtype=np.float32)
        self.rewards_to_go = np.zeros(total_frames, dtype=np.float32)

        self.traj_metadata = []
        buffer_pointer = 0

        for traj_id, traj in enumerate(self.trajectories):
            traj_len = len(traj["observations"])
            self.traj_metadata.append((buffer_pointer, traj_len))

            self.states[buffer_pointer:buffer_pointer + traj_len] = traj["observations"]
            self.actions[buffer_pointer:buffer_pointer + traj_len] = traj["actions"]
            self.rewards_to_go[buffer_pointer:buffer_pointer + traj_len] = compute_discounted_cumsum(traj["rewards"], gamma)

            buffer_pointer += traj_len


        self.state_mean = self.states.mean(axis=0)
        self.state_std = self.states.std(axis=0)
        self.drop_fn = create_drop_function(drop_config, total_frames, [m[0] for m in self.traj_metadata], self.random_generator)

        if sample_mode == "traj_length":
            traj_lengths = np.array([m[1] for m in self.traj_metadata])
            self.sampling_probs = traj_lengths / traj_lengths.sum()
        elif sample_mode == "uniform":
            self.sampling_probs = np.ones(self.num_trajs) / self.num_trajs
        else:
            raise ValueError(f"Invalid sampling mode: {sample_mode}")

    def sample(self, batch_size):
        """
        Samples a batch of sequences.

        Args:
            batch_size (int): Number of samples.

        Returns:
            Tuple[torch.Tensor]: Sampled states, actions, rewards-to-go, timesteps, dropsteps, and masks.
        """
        selected_indices = self.random_generator.choice(self.num_trajs, size=batch_size, p=self.sampling_probs)
        selected_start_points = [self.traj_metadata[idx][0] for idx in selected_indices]

        offsets = self.random_generator.integers(0, high=self.context_length, size=batch_size)
        sampled_sp = np.array(selected_start_points) + offsets[:, None]
        sampled_ep = sampled_sp + self.context_length

        masks = sampled_sp < sampled_ep[:, None]
        self.drop_fn.step()
        drop_steps = self.drop_fn.get_dropsteps(sampled_sp)
        obs_indices = sampled_sp - drop_steps
        states = torch.tensor(self.states[obs_indices], dtype=torch.float32, device=device)
        actions = torch.tensor(self.actions[sampled_sp], dtype=torch.float32, device=device)
        rewards_to_go = torch.tensor(self.rewards_to_go[obs_indices], dtype=torch.float32, device=device)
        timesteps = torch.tensor(offsets[:, None] + np.arange(self.context_length), dtype=torch.int32, device=device)
        dropsteps = torch.tensor(drop_steps, dtype=torch.int32, device=device)

        return states, actions, rewards_to_go, timesteps, dropsteps, masks
