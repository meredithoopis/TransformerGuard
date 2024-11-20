import numpy as np
import gymnasium as gym


def create_drop_function(drop_config, buffer_length, trajectory_start_points, random_generator):
    """
    Factory function to create a drop function based on the given configuration.

    Args:
        drop_config: Configuration for the drop function.
        buffer_length: Total size of the buffer.
        trajectory_start_points: Start points for each trajectory.
        random_generator: Random number generator.

    Returns:
        Drop function instance.
    """
    if drop_config.drop_fn == "const":
        return ConstDropFunction(buffer_length, drop_config.drop_p, drop_config.update_interval, trajectory_start_points, random_generator)
    elif drop_config.drop_fn == "linear":
        return LinearDropFunction(
            buffer_length,
            drop_config.start_p,
            drop_config.end_p,
            drop_config.ascend_steps,
            drop_config.update_interval,
            trajectory_start_points,
            random_generator,
        )
    else:
        raise ValueError(f"Unsupported drop function type: {drop_config.drop_fn}")


class ObservationDropWrapper(gym.Wrapper):
    """
    A Gym wrapper to randomly drop observations with a given probability.
    """
    def __init__(self, env, drop_probability, random_seed):
        super().__init__(env)
        self.drop_probability = drop_probability
        self.last_observation = None
        self.random_generator = np.random.default_rng(random_seed)

    def step(self, action):
        next_obs, reward, done, trunc, info = self.env.step(action)
        if self.random_generator.random() > self.drop_probability:
            self.last_observation = next_obs
            info["dropped"] = False
        else:
            info["dropped"] = True
        return self.last_observation, reward, done, trunc, info

    def reset(self, seed):
        self.last_observation, info = self.env.reset(seed=seed)
        return self.last_observation, info


class DropFunction:
    """
    Base class for managing the dropping of steps in a trajectory.
    """
    def __init__(self, buffer_size, update_frequency, trajectory_start_points, rng, drop_aware=True):
        self.buffer_size = buffer_size
        self.update_frequency = update_frequency
        self.trajectory_start_points = np.append(trajectory_start_points, buffer_size - 1)
        self.random_generator = rng
        self.step_count = 0
        self.drop_aware = drop_aware
        self.drop_mask = np.ones(buffer_size, dtype=bool)
        self.drop_steps = np.zeros(buffer_size, dtype=int)

    def get_dropped_steps(self, indices):
        return self.drop_steps[indices]

    def get_drop_mask(self, indices):
        return self.drop_mask[indices]

    def step(self):
        if self.step_count % self.update_frequency == 0 and self.drop_aware:
            self.update_drop_mask()
            self.calculate_drop_steps()
        self.step_count += 1

    def update_drop_mask(self):
        raise NotImplementedError("Drop mask update method must be implemented in subclasses.")

    def calculate_drop_steps(self):
        """
        Calculate the number of steps since the last valid frame.
        """
        cumulative_drops = np.cumsum(~self.drop_mask)
        valid_frame_differences = np.diff(np.concatenate(([0], cumulative_drops[self.drop_mask])))
        valid_frame_distances = np.ones(self.buffer_size, dtype=int)
        valid_frame_distances[self.drop_mask] = -valid_frame_differences
        self.drop_steps = np.cumsum(valid_frame_distances)
        self.drop_steps[-1] = 0


class ConstDropFunction(DropFunction):
    """
    Drop function with a constant probability for dropping observations.
    """
    def __init__(self, buffer_size, drop_probability, update_frequency, trajectory_start_points, rng, drop_aware=True):
        super().__init__(buffer_size, update_frequency, trajectory_start_points, rng, drop_aware)
        self.drop_probability = drop_probability

    def update_drop_mask(self):
        self.drop_mask = self.random_generator.random(self.buffer_size) > self.drop_probability
        self.drop_mask[self.trajectory_start_points] = True
        self.drop_mask[-1] = False


class LinearDropFunction(DropFunction):
    """
    Drop function with a linearly increasing probability for dropping observations.
    """
    def __init__(self, buffer_size, start_probability, end_probability, ascend_steps, update_frequency, trajectory_start_points, rng, drop_aware=True):
        super().__init__(buffer_size, update_frequency, trajectory_start_points, rng, drop_aware)
        self.start_probability = start_probability
        self.end_probability = end_probability
        self.ascend_steps = ascend_steps

    def update_drop_mask(self):
        progress_ratio = min(self.step_count / self.ascend_steps, 1)
        drop_probability = self.end_probability * progress_ratio + self.start_probability * (1 - progress_ratio)

        if progress_ratio in [0.25, 0.5, 0.75]:
            print(f"Progress: {progress_ratio:.2f}, Drop Probability: {drop_probability:.4f}")

        self.drop_mask = self.random_generator.random(self.buffer_size) > drop_probability
        self.drop_mask[self.trajectory_start_points] = True
        self.drop_mask[-1] = False
