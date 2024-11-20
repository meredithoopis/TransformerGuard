# Inspiration: https://github.com/kzl/decision-transformer/blob/master/gym/decision_transformer/models/decision_transformer.py
# https://github.com/kzl/decision-transformer
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalAttention(nn.Module):
    """
    Implements masked causal attention with multiple heads.
    """
    def __init__(self, hidden_dim, context_length, num_heads, dropout_prob):
        super().__init__()
        self.num_heads = num_heads
        self.context_length = context_length

        # Query, Key, Value transformations
        self.query_net = nn.Linear(hidden_dim, hidden_dim)
        self.key_net = nn.Linear(hidden_dim, hidden_dim)
        self.value_net = nn.Linear(hidden_dim, hidden_dim)

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)

        # Dropout layers
        self.attention_dropout = nn.Dropout(dropout_prob)
        self.projection_dropout = nn.Dropout(dropout_prob)

        # Mask to ensure causal (future-blind) attention
        causal_mask = torch.tril(torch.ones((context_length, context_length))).view(1, 1, context_length, context_length)
        self.register_buffer("causal_mask", causal_mask)

    def forward(self, x):
        B, T, C = x.size()  # Batch size, sequence length, hidden dimension
        H, D = self.num_heads, C // self.num_heads  # Number of heads, dimension per head

        # Transform input to queries, keys, and values
        queries = self.query_net(x).view(B, T, H, D).transpose(1, 2)
        keys = self.key_net(x).view(B, T, H, D).transpose(1, 2)
        values = self.value_net(x).view(B, T, H, D).transpose(1, 2)

        # Compute scaled dot-product attention
        attention_scores = torch.matmul(queries, keys.transpose(-2, -1)) / math.sqrt(D)
        attention_scores = attention_scores.masked_fill(self.causal_mask[..., :T, :T] == 0, float("-inf"))

        # Apply softmax and dropout to attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_output = self.attention_dropout(torch.matmul(attention_weights, values))

        # Combine heads and project to output
        attention_output = attention_output.transpose(1, 2).contiguous().view(B, T, C)
        return self.projection_dropout(self.output_proj(attention_output))


class TransformerBlock(nn.Module):
    """
    Transformer block consisting of masked causal attention and feedforward layers.
    """
    def __init__(self, hidden_dim, context_length, num_heads, dropout_prob):
        super().__init__()
        self.attention = CausalAttention(hidden_dim, context_length, num_heads, dropout_prob)
        self.feedforward = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.Dropout(dropout_prob),
        )
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        x = x + self.attention(x)  # Residual connection + attention
        x = self.layer_norm1(x)
        x = x + self.feedforward(x)  # Residual connection + feedforward
        x = self.layer_norm2(x)
        return x


class DecisionTransformer(nn.Module):
    """
    Decision Transformer model for reinforcement learning tasks.
    """
    def __init__(self, state_dim, action_dim, num_heads, num_blocks, hidden_dim, context_length, dropout_prob, 
                 action_space, state_mean, state_std, reward_scale, max_timestep, drop_aware, device):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.context_length = context_length

        # Stack of transformer blocks
        self.transformer_blocks = nn.Sequential(
            *[TransformerBlock(hidden_dim, 3 * context_length, num_heads, dropout_prob) for _ in range(num_blocks)]
        )

        # Embedding layers
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.timestep_embedding = nn.Embedding(max_timestep, hidden_dim)
        self.dropstep_embedding = nn.Embedding(max_timestep, hidden_dim)
        self.rtg_embedding = nn.Linear(1, hidden_dim)
        self.state_embedding = nn.Linear(state_dim, hidden_dim)
        self.action_embedding = nn.Linear(action_dim, hidden_dim)

        # Output prediction layers
        self.rtg_predictor = nn.Linear(hidden_dim, 1)
        self.state_predictor = nn.Linear(hidden_dim, state_dim)
        self.action_predictor = nn.Sequential(nn.Linear(hidden_dim, action_dim), nn.Tanh())

        # Action and state normalization
        self.action_space = action_space
        self.action_space.low = torch.tensor(self.action_space.low, dtype=torch.float32)
        self.action_space.high = torch.tensor(self.action_space.high, dtype=torch.float32)
        self.state_mean = torch.tensor(state_mean, dtype=torch.float32)
        self.state_std = torch.tensor(state_std, dtype=torch.float32)
        self.reward_scale = reward_scale
        self.max_timestep = max_timestep
        self.drop_aware = drop_aware
        self.to(device)

    def normalize_action(self, action):
        return (action + 1) * (self.action_space.high - self.action_space.low) / 2 + self.action_space.low

    def normalize_state(self, state):
        return (state - self.state_mean) / self.state_std

    def normalize_rtg(self, rtg):
        return rtg / self.reward_scale

    def __repr__(self):
        return "DecisionTransformer"

    def to(self, device):
        self.action_space.low = self.action_space.low.to(device)
        self.action_space.high = self.action_space.high.to(device)
        self.state_mean = self.state_mean.to(device)
        self.state_std = self.state_std.to(device)
        return super().to(device)

    def freeze_core_layers(self):
        for layer in [self.state_embedding, self.action_embedding, self.rtg_embedding, self.timestep_embedding, self.transformer_blocks, self.layer_norm]:
            for param in layer.parameters():
                param.requires_grad = False

    def forward(self, states, actions, rewards_to_go, timesteps, dropsteps):
        states = self.normalize_state(states)
        rewards_to_go = self.normalize_rtg(rewards_to_go)
        B, T, _ = states.shape

        # Create embeddings
        time_embeds = self.timestep_embedding(timesteps)
        state_embeds = self.state_embedding(states) + time_embeds
        action_embeds = self.action_embedding(actions) + time_embeds
        rtg_embeds = self.rtg_embedding(rewards_to_go) + time_embeds

        if self.drop_aware:
            drop_embeds = self.dropstep_embedding(dropsteps)
            state_embeds += drop_embeds
            rtg_embeds += drop_embeds

        # Reshape sequence: (r_0, s_0, a_0, r_1, s_1, a_1, ...)
        input_sequence = torch.stack((rtg_embeds, state_embeds, action_embeds), dim=2).reshape(B, 3 * T, self.hidden_dim)

        # Apply transformer blocks and layer normalization
        h = self.layer_norm(input_sequence)
        h = self.transformer_blocks(h)

        # Reshape output for predictions
        h = h.view(B, T, 3, self.hidden_dim).permute(0, 2, 1, 3)

        return self.state_predictor(h[:, 2]), self.normalize_action(self.action_predictor(h[:, 1])), self.rtg_predictor(h[:, 2])

    def save(self, filename):
        os.makedirs("models", exist_ok=True)
        torch.save(self.state_dict(), os.path.join("models", f"{filename}.pt"))

    def load(self, filename):
        self.load_state_dict(torch.load(os.path.join("models", f"{filename}.pt")))
