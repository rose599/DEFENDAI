"""
app.py
======
Flask web dashboard for DefendAI.

Routes:
    GET  /                    Home – system overview
    GET  /simulation          Live simulation control page
    POST /api/sim/start       Start background simulation thread
    POST /api/sim/stop        Stop simulation
    GET  /api/sim/status      Polling endpoint for live data
    GET  /metrics             Metrics dashboard
    GET  /logs                Attack logs table
    GET  /api/metrics         JSON metrics for Chart.js
    GET  /api/rewards         JSON reward history
    GET  /api/actions         JSON action distribution
    GET  /api/attack_logs     JSON attack log rows
"""

import os
import sys
import threading
import time

import numpy as np
from flask import Flask, jsonify, render_template, request

sys.path.insert(0, os.path.dirname(__file__))

from database.db_manager import (
    compute_and_store_metrics, get_action_distribution,
    get_attack_logs, get_latest_metrics, get_recent_episodes,
    get_recent_steps, get_reward_history, init_db,
    log_attack_event, log_episode, log_step,
)

app = Flask(
    __name__,
    template_folder="dashboard/templates",
    static_folder="dashboard/static",
)

# ── Global simulation state ────────────────────────────────────────────────────
_sim_thread: threading.Thread | None = None
_sim_running = False
_sim_lock = threading.Lock()
_live_state: dict = {}           # shared between sim thread and Flask
_episode_counter = 10_000        # start above training episodes to distinguish


def _run_simulation():
    """
    Background thread: runs inference episodes with trained (or heuristic)
    defender vs rule-based attacker, logging every step to SQLite.
    """
    global _sim_running, _live_state, _episode_counter

    from env.cyber_defense_env import CyberDefenseEnv
    from attacker.red_team import RedTeamAttacker
    from defender.blue_team import BlueTeamDefender

    env = CyberDefenseEnv(max_steps=150)
    attacker = RedTeamAttacker(escalation_rate=0.2, seed=99)
    defender = BlueTeamDefender()
    model_loaded = defender.load()
    if not model_loaded:
        print("[Sim] No trained model found – using heuristic defender.")

    while _sim_running:
        _episode_counter += 1
        ep_num = _episode_counter
        obs, _ = env.reset()
        attacker.reset()
        ep_reward = 0.0
        step = 0
        outcome = "truncated"

        while _sim_running:
            step += 1
            action, action_label = defender.predict(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward

            # Log step to DB
            log_step(
                episode=ep_num, step=step,
                action=action, action_label=action_label,
                reward=reward, state=info["state"],
                attack_stage_label=info["attack_stage_label"],
                false_positive=info.get("false_positive", False),
                mitigation_ok=info.get("mitigation_success", False),
            )

            # Generate attacker event for log table
            atk_event = attacker.choose_attack(defender_monitoring=0.0)
            log_attack_event(ep_num, step, atk_event)

            # Update shared live state (thread-safe via GIL for dict writes)
            with _sim_lock:
                _live_state = {
                    "episode": ep_num,
                    "step": step,
                    "action": action_label,
                    "reward": round(reward, 2),
                    "ep_reward": round(ep_reward, 2),
                    "compromised_nodes": info["state"][0],
                    "traffic_score": round(info["state"][1], 3),
                    "system_health": round(info["state"][2], 3),
                    "active_alerts": info["state"][3],
                    "attack_stage": info["attack_stage_label"],
                    "attack_payload": atk_event["payload"],
                    "attack_target": atk_event["target_node"],
                    "running": True,
                }

            if terminated or truncated:
                outcome = info.get("outcome", "truncated")
                break

            time.sleep(0.15)   # throttle to ~6 steps/sec for dashboard

        # Log episode summary
        log_episode(ep_num, step, round(ep_reward, 3), outcome)
        compute_and_store_metrics()

        # Brief pause between episodes
        time.sleep(0.5)

    with _sim_lock:
        _live_state["running"] = False


# ── Page routes ────────────────────────────────────────────────────────────────

@app.route("/")
def home():
    metrics = get_latest_metrics()
    episodes = get_recent_episodes(5)
    return render_template("index.html", metrics=metrics, episodes=episodes)


@app.route("/simulation")
def simulation():
    return render_template("simulation.html")


@app.route("/metrics")
def metrics_page():
    metrics = get_latest_metrics()
    return render_template("metrics.html", metrics=metrics)


@app.route("/logs")
def logs_page():
    attack_logs = get_attack_logs(50)
    return render_template("logs.html", logs=attack_logs)


# ── Simulation control API ─────────────────────────────────────────────────────

@app.route("/api/sim/start", methods=["POST"])
def sim_start():
    global _sim_thread, _sim_running
    if _sim_running:
        return jsonify({"status": "already_running"})
    _sim_running = True
    _sim_thread = threading.Thread(target=_run_simulation, daemon=True)
    _sim_thread.start()
    return jsonify({"status": "started"})


@app.route("/api/sim/stop", methods=["POST"])
def sim_stop():
    global _sim_running
    _sim_running = False
    return jsonify({"status": "stopped"})


@app.route("/api/sim/status")
def sim_status():
    with _sim_lock:
        state = dict(_live_state)
    state["running"] = _sim_running
    return jsonify(state)


# ── Data API ───────────────────────────────────────────────────────────────────

@app.route("/api/metrics")
def api_metrics():
    return jsonify(get_latest_metrics())


@app.route("/api/rewards")
def api_rewards():
    return jsonify(get_reward_history(100))


@app.route("/api/actions")
def api_actions():
    return jsonify(get_action_distribution())


@app.route("/api/attack_logs")
def api_attack_logs():
    return jsonify(get_attack_logs(100))


@app.route("/api/recent_steps")
def api_recent_steps():
    return jsonify(get_recent_steps(50))


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    init_db()
    print("\n DefendAI Dashboard → http://127.0.0.1:5000\n")
    app.run(debug=False, host="0.0.0.0", port=5000)
