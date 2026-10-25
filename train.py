"""
train.py
========
Entry point for training the Blue Team PPO defender.

Usage:
    python train.py [--timesteps 100000] [--episodes 200]

What it does:
    1. Initialises SQLite database
    2. Trains PPO agent on CyberDefenseEnv
    3. Logs every episode to the DB
    4. Saves trained model to models/ppo_defender.zip
    5. Generates a training reward plot → logs/training_reward.png
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")   # headless backend
import matplotlib.pyplot as plt

# Add project root to path so imports resolve
sys.path.insert(0, os.path.dirname(__file__))

from database.db_manager import init_db, log_episode, compute_and_store_metrics
from defender.blue_team import BlueTeamDefender


def plot_rewards(episode_log: list[dict], out_path: str):
    """Generate and save reward curve plot."""
    episodes = [e["episode"] for e in episode_log]
    rewards  = [e["total_reward"] for e in episode_log]

    # Compute rolling average (window=10)
    window = min(10, len(rewards))
    rolling = [
        sum(rewards[max(0, i - window):i + 1]) / min(window, i + 1)
        for i in range(len(rewards))
    ]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(episodes, rewards, alpha=0.3, color="#4fc3f7", label="Episode Reward")
    ax.plot(episodes, rolling, color="#0288d1", linewidth=2, label=f"Rolling Avg (w={window})")
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_title("DefendAI – PPO Training Reward Curve", fontsize=14, fontweight="bold")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[Train] Reward plot saved → {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Train DefendAI PPO agent")
    parser.add_argument("--timesteps", type=int, default=100_000,
                        help="Total training timesteps (default: 100000)")
    args = parser.parse_args()

    print("=" * 60)
    print("  DefendAI – Blue Team Defender Training")
    print("=" * 60)

    # 1. Init DB
    init_db()

    # 2. Train
    episode_log: list[dict] = []
    defender = BlueTeamDefender()
    print(f"\n[Train] Starting PPO training for {args.timesteps:,} timesteps ...\n")
    episode_log = defender.train(total_timesteps=args.timesteps,
                                  episode_log=episode_log)

    # 3. Persist episodes to DB
    print(f"\n[Train] Logging {len(episode_log)} episodes to SQLite ...")
    for ep in episode_log:
        log_episode(
            episode=ep["episode"],
            steps=ep["steps"],
            total_reward=ep["total_reward"],
            outcome="training",
        )

    # 4. Compute KPI metrics
    metrics = compute_and_store_metrics()
    if metrics:
        print("\n[Train] Computed Metrics:")
        for k, v in metrics.items():
            print(f"        {k}: {v}")

    # 5. Plot
    os.makedirs("logs", exist_ok=True)
    plot_rewards(episode_log, "logs/training_reward.png")

    print("\n[Train] Done! Model saved to models/ppo_defender.zip")
    print("        Run `python app.py` to start the dashboard.\n")


if __name__ == "__main__":
    main()
