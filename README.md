# 🛡️ DefendAI – Autonomous Cyber Threat Simulation & Defense System

> **Academic Major Project** | Cybersecurity + Reinforcement Learning  
> Python 3.10+ | Stable-Baselines3 PPO | Custom OpenAI Gymnasium Environment | Flask Dashboard

---

## 📋 Project Overview

DefendAI simulates a virtualised network where a **Red Team AI (attacker)** continuously escalates
cyber attacks through 5 stages, while a **Blue Team RL agent (defender)** trained via Proximal
Policy Optimisation (PPO) learns to detect and mitigate threats in real-time.

### Attack Stages (Red Team)
| Stage | Name | Behaviour |
|-------|------|-----------|
| 0 | Reconnaissance | Port scans, OS fingerprinting |
| 1 | Phishing / Initial Access | Credential harvesting emails |
| 2 | Malware Propagation | Worm/ransomware spread via SMB |
| 3 | Privilege Escalation | Kernel exploit, sudo abuse |
| 4 | Lateral Movement | PsExec, RDP hijacking across subnet |

### Defender Actions (Blue Team PPO)
| Action | Label | Effect |
|--------|-------|--------|
| 0 | Do Nothing | No cost, misses real threats |
| 1 | Block IP | Reduces traffic + alerts |
| 2 | Isolate Node | Removes compromised nodes |
| 3 | Reset System | Full recovery, expensive |
| 4 | Increase Monitoring | Reduces attacker success probability |

---

## 🏗️ Architecture

```
Red Team (Rule-Based Attacker)
         │ attack events
         ▼
   CyberDefenseEnv (gymnasium.Env)
   ─────────────────────────────
   State: [compromised_nodes, traffic_score,
           system_health, active_alerts, attack_stage]
         │ observations
         ▼
Blue Team PPO Agent (Stable-Baselines3)
         │ actions + rewards
         ▼
   SQLite Database ──► Flask Dashboard
```

---

## 📁 Folder Structure

```
DefendAI/
├── env/
│   └── cyber_defense_env.py    # Custom Gym environment
├── attacker/
│   └── red_team.py             # Rule-based attacker
├── defender/
│   └── blue_team.py            # PPO agent wrapper
├── dashboard/
│   ├── templates/              # Jinja2 HTML pages
│   └── static/                 # CSS + Chart.js
├── database/
│   └── db_manager.py           # SQLite CRUD
├── models/                     # Saved model weights
├── logs/                       # Plots + SQLite DB
├── train.py                    # Training entry point
├── app.py                      # Flask web app
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone <repo-url>
cd DefendAI

# Create virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Train the Defender

```bash
python train.py --timesteps 100000
```

This will:
- Train the PPO defender for 100k timesteps (~2-5 min on CPU)
- Save model to `models/ppo_defender.zip`
- Log all episodes to `logs/defendai.db`
- Generate `logs/training_reward.png`

For a quick test run:
```bash
python train.py --timesteps 20000
```

### 3. Launch the Dashboard

```bash
python app.py
```

Open **http://localhost:5000** in your browser.

### 4. Dashboard Pages

| Page | URL | Description |
|------|-----|-------------|
| Home | `/` | KPI overview + recent episodes |
| Live Simulation | `/simulation` | Start/stop inference simulation |
| Metrics | `/metrics` | Charts + training reward plot |
| Attack Logs | `/logs` | Searchable attack event table |

---

## 🔬 Environment Details

### State Space
```python
observation_space = Box(
    low  = [0.0, 0.0, 0.0,  0.0, 0.0],
    high = [10., 1.0, 1.0, 20.0, 4.0],
    dtype = float32
)
# Indices: [compromised_nodes, traffic_score, system_health, active_alerts, attack_stage]
```

### Reward Function
```
+10  Successful mitigation (attack stopped/node recovered)
+15  Terminal win bonus (system fully restored)
 -2  Unnecessary disruption (action when no threat)
 -5  False positive (defensive action on clean system)
-10  Missed attack (do-nothing when threat present)
-20  Terminal loss penalty (system health → 0)
```

### Hyperparameters (PPO)
| Parameter | Value |
|-----------|-------|
| Learning rate | 3e-4 |
| n_steps | 1024 |
| batch_size | 64 |
| n_epochs | 10 |
| gamma | 0.99 |
| ent_coef | 0.01 |

---

## 📊 Key Metrics

| Metric | Definition |
|--------|-----------|
| Detection Rate | `mitigations / total_steps` |
| False Positive Rate | `false_positives / total_steps` |
| Avg Response Time | `1 / detection_rate` (synthetic) |
| System Uptime % | `mean(system_health) × 100` |

---

## 🔧 Extending the Project

**Custom attacker:** Modify `attacker/red_team.py` → `PAYLOADS` dict and escalation logic.

**Different RL algorithm:** In `defender/blue_team.py`, swap `PPO` for `DQN` or `A2C`:
```python
from stable_baselines3 import DQN
self.model = DQN("MlpPolicy", env=self.env, ...)
```

**Larger network:** Increase `max_nodes` in the environment and expand state space.

---

## 📚 Academic References

- Schulman et al. (2017). *Proximal Policy Optimization Algorithms.* arXiv:1707.06347
- Brockman et al. (2016). *OpenAI Gym.* arXiv:1606.01540
- Miehling et al. (2018). *A POMDP Approach to the Dynamic Defense of Large-Scale Cyber Networks.*
- Bland et al. (2020). *Machine Learning Cyberattack and Defense Strategies.*

---

## 📄 License
MIT License – for academic use.
