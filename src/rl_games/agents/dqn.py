import random
from collections import deque
from pathlib import Path
from typing import Self

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ── Neural network ────────────────────────────────────────────────────


class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── Replay buffer ────────────────────────────────────────────────────


class ReplayBuffer:
    def __init__(self, capacity: int = 100_000) -> None:
        self.buffer: deque[tuple] = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done) -> None:
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> list[tuple]:
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        return len(self.buffer)


# ── Agent ─────────────────────────────────────────────────────────────


class DQNAgent:
    def __init__(
        self,
        env_id: str,
        *,
        lr: float = 5e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        batch_size: int = 64,
        buffer_capacity: int = 100_000,
        target_update_freq: int = 10,
        hidden: int = 128,
    ) -> None:
        self.env_id = env_id
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.training_episodes = 0

        env = gym.make(env_id)
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = int(env.action_space.n)
        env.close()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_net = QNetwork(self.state_dim, self.action_dim, hidden).to(self.device)
        self.target_net = QNetwork(self.state_dim, self.action_dim, hidden).to(
            self.device
        )
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.buffer = ReplayBuffer(buffer_capacity)

    def select_action(self, state: np.ndarray, *, deterministic: bool = False) -> int:
        if not deterministic and random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        with torch.no_grad():
            t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            return int(self.q_net(t).argmax(dim=1).item())

    def predict(
        self, obs: np.ndarray, *, deterministic: bool = True
    ) -> tuple[int, None]:
        return self.select_action(obs, deterministic=deterministic), None

    def _learn(self) -> float:
        if len(self.buffer) < self.batch_size:
            return 0.0

        batch = self.buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states_t = torch.FloatTensor(np.array(states)).to(self.device)
        actions_t = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards_t = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states_t = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones_t = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        current_q = self.q_net(states_t).gather(1, actions_t)
        with torch.no_grad():
            next_q = self.target_net(next_states_t).max(dim=1, keepdim=True).values

        target_q = rewards_t + self.gamma * next_q * (1.0 - dones_t)
        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=10.0)
        self.optimizer.step()
        return loss.item()

    def train(self, total_episodes: int = 1000, log_interval: int = 50) -> list[float]:
        env = gym.make(self.env_id)
        rewards_history: list[float] = []
        best_avg_reward = -float("inf")
        solved_threshold = 260  # Meta del taller

        for episode in range(1, total_episodes + 1):
            obs, _ = env.reset()
            total_reward, done = 0.0, False

            while not done:
                action = self.select_action(obs)
                next_obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                self.buffer.push(obs, action, float(reward), next_obs, done)
                self._learn()
                obs, total_reward = next_obs, total_reward + reward

            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
            self.training_episodes += 1
            rewards_history.append(total_reward)

            if episode % self.target_update_freq == 0:
                self.target_net.load_state_dict(self.q_net.state_dict())

            if episode % log_interval == 0:
                avg = np.mean(rewards_history[-log_interval:])
                print(
                    f"Ep {episode}/{total_episodes} | Avg: {avg:.2f} | Eps: {self.epsilon:.4f}"
                )

                # Guardar el mejor modelo hasta ahora
                if avg > best_avg_reward:
                    best_avg_reward = avg
                    self.save(Path(f"saves/{self.env_id}_best.pth"))

                # Early Stopping: Si ya superamos el objetivo del profesor
                if avg >= solved_threshold:
                    print(
                        f"\n¡Éxito! Promedio {avg:.2f} >= {solved_threshold}. Parando entrenamiento."
                    )
                    self.save(Path(f"saves/{self.env_id}_final.pth"))
                    break
        env.close()
        return rewards_history

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "q_net_state": self.q_net.state_dict(),
                "epsilon": self.epsilon,
                "env_id": self.env_id,
                "lr": self.lr,
                "gamma": self.gamma,
                "batch_size": self.batch_size,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> Self:
        data = torch.load(path, weights_only=False)
        agent = cls(
            data["env_id"],
            lr=data["lr"],
            gamma=data["gamma"],
            batch_size=data["batch_size"],
        )
        agent.q_net.load_state_dict(data["q_net_state"])
        agent.epsilon = data["epsilon"]
        return agent
