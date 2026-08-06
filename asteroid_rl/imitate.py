"""Behavior-cloning warm-start from the scripted landing policy.

Collects (obs, action) pairs by rolling out ``scripted_action``, then fits the
SB3 policy network with supervised MSE loss so PPO fine-tuning starts near a
landing-capable controller instead of random throttle.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.nn import functional as F

from asteroid_rl.env import AsteroidLandingEnv, LandingEnvConfig
from asteroid_rl.policies import scripted_action, scripted_orbit_action


def collect_scripted_transitions(
    config: LandingEnvConfig,
    *,
    episodes: int = 8,
    max_steps: int = 1000,
    orbit: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Roll out a scripted baseline and stack observations / actions.

    Args:
        config: Environment config (obs_mode, flat surface, etc.).
        episodes: Number of scripted episodes to collect.
        max_steps: Cap on steps per episode.
        orbit: If True, use ``scripted_orbit_action`` (4-D).

    Returns:
        Tuple ``(observations, actions)`` with shapes ``(N, obs_dim)`` and
        ``(N, act_dim)``.
    """
    env = AsteroidLandingEnv(config=config)
    action_fn = scripted_orbit_action if orbit else scripted_action
    obs_list: List[np.ndarray] = []
    act_list: List[np.ndarray] = []
    for _ in range(int(episodes)):
        obs, info = env.reset()
        for _step in range(int(max_steps)):
            action = action_fn(obs, info)
            obs_list.append(np.asarray(obs, dtype=np.float32).reshape(-1))
            act_list.append(np.asarray(action, dtype=np.float32).reshape(-1))
            obs, _reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
    env.close()
    if not obs_list:
        raise RuntimeError("No scripted transitions collected")
    return np.stack(obs_list, axis=0), np.stack(act_list, axis=0)


def behavior_clone_ppo(
    model,
    observations: np.ndarray,
    actions: np.ndarray,
    *,
    epochs: int = 30,
    batch_size: int = 64,
    lr: float = 3e-4,
) -> float:
    """Fit an SB3 PPO policy to expert actions with MSE on the mean action.

    Args:
        model: Loaded ``stable_baselines3.PPO`` instance.
        observations: Expert observations, shape ``(N, obs_dim)``.
        actions: Expert actions, shape ``(N, act_dim)``.
        epochs: Number of passes over the dataset.
        batch_size: Mini-batch size.
        lr: Adam learning rate.

    Returns:
        Final mean batch loss.
    """
    from stable_baselines3.common.utils import obs_as_tensor

    device = model.device
    policy = model.policy
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    x_np = np.asarray(observations, dtype=np.float32)
    y_np = np.asarray(actions, dtype=np.float32)
    n = int(x_np.shape[0])
    last_loss = float("inf")
    for _epoch in range(int(epochs)):
        perm = np.random.permutation(n)
        total = 0.0
        count = 0
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            batch_x = obs_as_tensor(x_np[idx], device)
            batch_y = torch.as_tensor(y_np[idx], dtype=torch.float32, device=device)
            dist = policy.get_distribution(batch_x)
            pred = dist.mode()
            loss = F.mse_loss(pred, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss.item())
            count += 1
        last_loss = total / max(count, 1)
    return last_loss


def warmstart_from_scripted(
    model,
    config: LandingEnvConfig,
    *,
    episodes: int = 8,
    epochs: int = 30,
    orbit: bool = False,
) -> float:
    """Collect scripted demos and behavior-clone the PPO policy.

    Args:
        model: PPO model to warm-start in place.
        config: Env config matching the model's observation space.
        episodes: Scripted episodes for the demo dataset.
        epochs: BC optimization epochs.
        orbit: If True, clone ``scripted_orbit_action``.

    Returns:
        Final BC loss.
    """
    obs, acts = collect_scripted_transitions(
        config, episodes=episodes, orbit=orbit, max_steps=2500 if orbit else 1000
    )
    loss = behavior_clone_ppo(model, obs, acts, epochs=epochs)
    print(f"Behavior-clone warm-start done: n={len(obs)} final_mse={loss:.6f}")
    return loss
