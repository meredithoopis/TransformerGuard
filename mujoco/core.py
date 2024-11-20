import torch
import numpy as np
import gymnasium as gym
from dotmap import DotMap
from omegaconf import OmegaConf
from gymnasium.vector import SyncVectorEnv
from gymnasium.wrappers import RecordEpisodeStatistics
from hydra.utils import instantiate
from buffer import SequenceBuffer
from model import DecisionTransformer
from drop_fn import ObservationDropWrapper
import torch.nn.functional as F
import utils


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def calculate_performance_drop(env: gym.vector.Env, model, target_rtg, drop_rates: list, seed):
    performance_means = []
    for drop_rate in drop_rates:
        modified_env = ObservationDropWrapper(env, drop_rate, seed)
        mean_return, _ = evaluate_policy(modified_env, model, target_rtg)
        performance_means.append(mean_return)
    return performance_means

@torch.no_grad()
def evaluate_policy(env: gym.vector.Env, model: DecisionTransformer, target_rtg):
    """
    Evaluate a given policy using a vectorized environment.

    Args:
        env (gym.vector.Env): The evaluation environment.
        model (DecisionTransformer): The trained model to evaluate.
        target_rtg: The target return-to-go for evaluation.

    Returns:
        Tuple[float, float]: Mean and standard deviation of returns.
    """
    model.eval()  # Set model to evaluation mode

    num_envs = env.num_envs
    accumulated_rewards = np.zeros(num_envs)
    episode_returns = np.zeros(num_envs)
    is_done = np.zeros(num_envs, dtype=bool)

    state_dim = utils.get_space_shape(env.observation_space, is_vector_env=True)
    action_dim = utils.get_space_shape(env.action_space, is_vector_env=True)
    max_steps = model.max_timestep
    context_length = model.context_len
    timestep_range = torch.arange(max_steps, device=device)
    dropsteps = torch.zeros(max_steps, dtype=torch.long, device=device)

    obs, _ = env.reset(seed=[np.random.randint(0, 10000) for _ in range(num_envs)])
    states = torch.zeros((num_envs, max_steps, state_dim), dtype=torch.float32, device=device)
    actions = torch.zeros((num_envs, max_steps, action_dim), dtype=torch.float32, device=device)
    rtgs = torch.zeros((num_envs, max_steps, 1), dtype=torch.float32, device=device)

    reward_to_go, time_step, drop_step = target_rtg, 0, 0

    while not is_done.all():
        states[:, time_step] = torch.tensor(obs, device=device)
        rtgs[:, time_step] = reward_to_go - torch.tensor(episode_returns, device=device).unsqueeze(-1)
        dropsteps[time_step] = drop_step
        context_indices = torch.arange(max(0, time_step - context_length + 1), time_step + 1)

        _, predicted_actions, _ = model.forward(
            states[:, context_indices],
            actions[:, context_indices],
            rtgs[:, context_indices - dropsteps[context_indices].cpu()],
            timestep_range[None, context_indices],
            dropsteps[None, context_indices]
        )

        action = predicted_actions[:, -1].detach()
        actions[:, time_step] = action

        obs, rewards, dones, truncs, info = env.step(action.cpu().numpy())
        drop_step = dropsteps[time_step].item() + 1 if info.get('dropped', False) else 0
        episode_returns += rewards * ~is_done
        is_done = np.logical_or(is_done, np.logical_or(dones, truncs))
        time_step += 1

    return np.mean(episode_returns), np.std(episode_returns)

def train_model(cfg, seed, log_data, process_idx, logger, sync_barrier, data_root):
    """
    Trains a decision transformer model.

    Args:
        cfg: Configuration settings for training.
        seed: Random seed for reproducibility.
        log_data: Dictionary for logging metrics.
        process_idx: Index of the current process (for multiprocessing).
        logger: Logger for output messages.
        sync_barrier: Multiprocessing synchronization barrier.
        data_root: Path to the dataset directory.
    """
    use_multiprocessing = sync_barrier is not None
    utils.setup_logging("main_mp.log" if use_multiprocessing else "main.log")
    env_name = cfg.env.env_name
    evaluation_env = gym.vector.make(
        env_name + '-v4',
        render_mode="rgb_array",
        num_envs=cfg.train.eval_episodes,
        asynchronous=False,
        wrappers=RecordEpisodeStatistics
    )
    utils.seed_all(evaluation_env, seed)

    state_dim = utils.get_space_shape(evaluation_env.observation_space, is_vector_env=True)
    action_dim = utils.get_space_shape(evaluation_env.action_space, is_vector_env=True)

    drop_config = cfg.buffer.drop_cfg
    replay_buffer = instantiate(cfg.buffer, root_dir=data_root, drop_cfg=drop_config, seed=seed)
    model = instantiate(
        cfg.model,
        state_dim=state_dim,
        action_dim=action_dim,
        action_space=evaluation_env.envs[0].action_space,
        state_mean=replay_buffer.state_mean,
        state_std=replay_buffer.state_std,
        device=device
    )
    training_config = DotMap(OmegaConf.to_container(cfg.train, resolve=True))
    optimizer = torch.optim.AdamW(model.parameters(), lr=training_config.lr, weight_decay=training_config.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min((step + 1) / training_config.warmup_steps, 1))

    logger.info(f"Training with seed {seed} for {training_config.train_steps} steps using {env_name} dataset")

    local_log_data = {key: [] for key in log_data.keys()} if use_multiprocessing else log_data
    best_reward = -float('inf')

    # Main training loop
    for step in range(1, training_config.train_steps + training_config.finetune_steps + 1):
        # Sample batch from buffer
        states, actions, rtgs, timesteps, dropsteps, masks = replay_buffer.sample(training_config.batch_size)
        state_preds, action_preds, return_preds = model.forward(states, actions, rtgs, timesteps, dropsteps)

        # Compute and log action loss
        action_preds = action_preds[masks]
        action_loss = F.mse_loss(action_preds, actions[masks].detach(), reduction='mean')
        utils.log_metric(local_log_data, 'action_loss', action_loss.item())

        # Optimize model
        optimizer.zero_grad()
        action_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
        optimizer.step()
        scheduler.step()

        # Evaluate periodically
        if step % training_config.eval_interval == 0:
            eval_mean, eval_std = evaluate_policy(evaluation_env, model, training_config.rtg_target)
            logger.info(f"Step: {step}, Eval mean: {eval_mean:.2f}, Eval std: {eval_std:.2f}")

            if eval_mean > best_reward:
                best_reward = eval_mean
                model.save(f'best_model_seed_{seed}')
    
    logger.info(f"Finished training with seed {seed}")
    return best_reward
