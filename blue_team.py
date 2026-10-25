"""
Blue Team Defender
==================
Wraps a Stable-Baselines3 PPO agent trained on CyberDefenseEnv.
Provides load/predict interface used by Flask dashboard and train.py.
"""

import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

# Lazy import to avoid circular dependency
def _make_env():
    from env.cyber_defense_env import CyberDefenseEnv
    return CyberDefenseEnv()


ACTION_LABELS = {
    0: "Do Nothing",
    1: "Block IP",
    2: "Isolate Node",
    3: "Reset System",
    4: "Increase Monitoring",
}


class EpisodeLoggerCallback(BaseCallback):
    """
    Custom SB3 callback: logs per-episode stats to a list.
    Used by train.py to persist metrics to SQLite.
    """

    def __init__(self, log_list: list, verbose: int = 0):
        super().__init__(verbose)
        self.log_list = log_list
        self._ep_reward = 0.0
        self._ep_steps = 0
        self._episode = 0

    def _on_step(self) -> bool:
        self._ep_reward += self.locals["rewards"][0]
        self._ep_steps += 1

        # SB3 sets dones when episode ends
        if self.locals["dones"][0]:
            self._episode += 1
            self.log_list.append({
                "episode": self._episode,
                "steps": self._ep_steps,
                "total_reward": round(self._ep_reward, 3),
            })
            self._ep_reward = 0.0
            self._ep_steps = 0
        return True


class BlueTeamDefender:
    """
    Wraps PPO agent for inference and training.

    Usage:
        defender = BlueTeamDefender()
        defender.train(total_timesteps=50000)
        action = defender.predict(observation)
    """

    MODEL_PATH = os.path.join(os.path.dirname(__file__),
                              "..", "models", "ppo_defender")

    def __init__(self):
        self.model: PPO | None = None
        self.env = DummyVecEnv([_make_env])

    def train(self, total_timesteps: int = 100_000,
              episode_log: list | None = None) -> list:
        """
        Train the PPO agent.

        Args:
            total_timesteps: Number of env steps to train for.
            episode_log: Optional list that will be populated with
                         per-episode dicts by EpisodeLoggerCallback.

        Returns:
            episode_log list (populated in-place).
        """
        if episode_log is None:
            episode_log = []

        callback = EpisodeLoggerCallback(log_list=episode_log, verbose=0)

        self.model = PPO(
            policy="MlpPolicy",
            env=self.env,
            learning_rate=3e-4,
            n_steps=1024,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,          # encourage exploration
            verbose=1,
           
        )

        self.model.learn(total_timesteps=total_timesteps, callback=callback)
        self.save()
        return episode_log

    def save(self):
        """Persist model weights to disk."""
        os.makedirs(os.path.dirname(self.MODEL_PATH), exist_ok=True)
        self.model.save(self.MODEL_PATH)
        print(f"[BlueTeam] Model saved → {self.MODEL_PATH}")

    def load(self) -> bool:
        """Load model weights from disk. Returns True on success."""
        path = self.MODEL_PATH + ".zip"
        if not os.path.exists(path):
            return False
        self.model = PPO.load(self.MODEL_PATH, env=self.env)
        print(f"[BlueTeam] Model loaded ← {path}")
        return True

    def predict(self, observation: np.ndarray,
                deterministic: bool = True) -> tuple[int, str]:
        """
        Run inference on a single observation.

        Args:
            observation: 1-D numpy array of shape (5,).
            deterministic: Use greedy policy if True.

        Returns:
            (action_int, action_label)
        """
        if self.model is None:
            # Fallback: simple heuristic if no model loaded
            action = self._heuristic(observation)
        else:
            obs = observation.reshape(1, -1)
            action, _ = self.model.predict(obs, deterministic=deterministic)
            action = int(action[0])

        return action, ACTION_LABELS[action]

    @staticmethod
    def _heuristic(obs: np.ndarray) -> int:
        """Simple rule-based fallback when no trained model exists."""
        compromised, traffic, health, alerts, stage = obs
        if health < 0.3 or compromised > 7:
            return 3   # Reset System
        if compromised > 0:
            return 2   # Isolate Node
        if traffic > 0.6 or alerts > 10:
            return 1   # Block IP
        if traffic > 0.3:
            return 4   # Increase Monitoring
        return 0       # Do Nothing
