"""
Database Manager
================
SQLite CRUD layer for DefendAI.

Tables:
    episodes      - per-episode training summary
    step_logs     - per-step action/reward records
    attack_events - attacker event log
    metrics       - computed KPI snapshots
"""

import os
import sqlite3
import time
from contextlib import contextmanager
from typing import Any

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "logs", "defendai.db")


@contextmanager
def _connect():
    """Thread-safe SQLite connection context manager."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row      # dict-like rows
    conn.execute("PRAGMA journal_mode=WAL")  # safe concurrent reads
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    """Create all tables if they don't exist."""
    with _connect() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS episodes (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            episode       INTEGER NOT NULL,
            steps         INTEGER,
            total_reward  REAL,
            outcome       TEXT,
            created_at    TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%S','now'))
        );

        CREATE TABLE IF NOT EXISTS step_logs (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            episode         INTEGER,
            step            INTEGER,
            action          INTEGER,
            action_label    TEXT,
            reward          REAL,
            compromised_nodes REAL,
            traffic_score   REAL,
            system_health   REAL,
            active_alerts   REAL,
            attack_stage    REAL,
            attack_stage_label TEXT,
            false_positive  INTEGER DEFAULT 0,
            mitigation_ok   INTEGER DEFAULT 0,
            created_at      TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%S','now'))
        );

        CREATE TABLE IF NOT EXISTS attack_events (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            episode     INTEGER,
            step        INTEGER,
            stage       INTEGER,
            stage_name  TEXT,
            attack_type TEXT,
            target_node TEXT,
            payload     TEXT,
            success     INTEGER DEFAULT 0,
            created_at  TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%S','now'))
        );

        CREATE TABLE IF NOT EXISTS metrics (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            detection_rate      REAL,
            false_positive_rate REAL,
            avg_response_time   REAL,
            system_uptime_pct   REAL,
            total_episodes      INTEGER,
            total_steps         INTEGER,
            computed_at         TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%S','now'))
        );
        """)
    print(f"[DB] Initialised → {DB_PATH}")


# ── Write helpers ──────────────────────────────────────────────────────────────

def log_episode(episode: int, steps: int, total_reward: float,
                outcome: str = "truncated"):
    """Insert one episode summary row."""
    with _connect() as conn:
        conn.execute(
            "INSERT INTO episodes (episode, steps, total_reward, outcome) "
            "VALUES (?,?,?,?)",
            (episode, steps, total_reward, outcome)
        )


def log_step(episode: int, step: int, action: int, action_label: str,
             reward: float, state: list[float], attack_stage_label: str,
             false_positive: bool = False, mitigation_ok: bool = False):
    """Insert one step record."""
    with _connect() as conn:
        conn.execute("""
            INSERT INTO step_logs
            (episode, step, action, action_label, reward,
             compromised_nodes, traffic_score, system_health,
             active_alerts, attack_stage, attack_stage_label,
             false_positive, mitigation_ok)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            episode, step, action, action_label, reward,
            state[0], state[1], state[2], state[3], state[4],
            attack_stage_label,
            int(false_positive), int(mitigation_ok)
        ))


def log_attack_event(episode: int, step: int, event: dict):
    """Insert one attack event from RedTeamAttacker."""
    with _connect() as conn:
        conn.execute("""
            INSERT INTO attack_events
            (episode, step, stage, stage_name, attack_type,
             target_node, payload, success)
            VALUES (?,?,?,?,?,?,?,?)
        """, (
            episode, step,
            event.get("stage", 0),
            event.get("stage_name", ""),
            event.get("stage_name", ""),
            event.get("target_node", ""),
            event.get("payload", ""),
            int(event.get("success", False)),
        ))


def compute_and_store_metrics():
    """Compute KPIs from step_logs and store snapshot."""
    with _connect() as conn:
        rows = conn.execute("SELECT * FROM step_logs").fetchall()
        if not rows:
            return None

        total = len(rows)
        mitigations = sum(1 for r in rows if r["mitigation_ok"])
        false_pos    = sum(1 for r in rows if r["false_positive"])
        health_vals  = [r["system_health"] for r in rows]

        detection_rate      = round(mitigations / max(1, total), 4)
        false_pos_rate      = round(false_pos   / max(1, total), 4)
        avg_resp_time       = round(1.0 / max(0.01, detection_rate), 2)  # synthetic
        uptime_pct          = round(float(sum(health_vals)) / max(1, len(health_vals)) * 100, 2)

        ep_count = conn.execute("SELECT COUNT(DISTINCT episode) FROM step_logs").fetchone()[0]

        conn.execute("""
            INSERT INTO metrics
            (detection_rate, false_positive_rate, avg_response_time,
             system_uptime_pct, total_episodes, total_steps)
            VALUES (?,?,?,?,?,?)
        """, (detection_rate, false_pos_rate, avg_resp_time,
              uptime_pct, ep_count, total))

        return {
            "detection_rate": detection_rate,
            "false_positive_rate": false_pos_rate,
            "avg_response_time": avg_resp_time,
            "system_uptime_pct": uptime_pct,
            "total_episodes": ep_count,
            "total_steps": total,
        }


# ── Read helpers ───────────────────────────────────────────────────────────────

def get_latest_metrics() -> dict:
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM metrics ORDER BY id DESC LIMIT 1"
        ).fetchone()
        return dict(row) if row else {}


def get_recent_episodes(n: int = 50) -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM episodes ORDER BY id DESC LIMIT ?", (n,)
        ).fetchall()
        return [dict(r) for r in rows]


def get_recent_steps(n: int = 200) -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM step_logs ORDER BY id DESC LIMIT ?", (n,)
        ).fetchall()
        return [dict(r) for r in reversed(rows)]


def get_attack_logs(n: int = 100) -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM attack_events ORDER BY id DESC LIMIT ?", (n,)
        ).fetchall()
        return [dict(r) for r in rows]


def get_reward_history(n: int = 100) -> list[dict]:
    """Return episode rewards for chart rendering."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT episode, total_reward FROM episodes ORDER BY episode LIMIT ?", (n,)
        ).fetchall()
        return [dict(r) for r in rows]


def get_action_distribution() -> dict:
    """Count of each action taken across all steps."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT action_label, COUNT(*) as cnt FROM step_logs GROUP BY action_label"
        ).fetchall()
        return {r["action_label"]: r["cnt"] for r in rows}
