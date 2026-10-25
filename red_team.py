"""
Red Team Attacker
=================
Rule-based attacker that escalates through attack stages.
Can be used standalone for testing or as reference for an RL-based attacker.

Attack Types & Corresponding Stages:
    Stage 0 - Reconnaissance   : port scans, OS fingerprinting
    Stage 1 - Phishing         : credential harvesting emails
    Stage 2 - Malware Spread   : worm / ransomware propagation
    Stage 3 - Privilege Escal. : exploit kernel / sudo mis-config
    Stage 4 - Lateral Movement : pivot across network segments
"""

import random
import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AttackEvent:
    """Represents a single attack event record."""
    timestamp: str
    stage: int
    stage_name: str
    attack_type: str
    target_node: str
    payload: str
    success: bool


# Attack payloads per stage (for realistic log generation)
PAYLOADS = {
    0: [
        "nmap -sV -O 192.168.1.0/24",
        "masscan -p80,443 10.0.0.0/8",
        "ping sweep 172.16.0.0/12",
    ],
    1: [
        "Spear-phishing email with malicious .docx attachment",
        "HTML credential harvester page deployed",
        "OAuth token theft via fake login portal",
    ],
    2: [
        "WannaCry variant propagation via SMB exploit",
        "Mimikatz credential dump",
        "Cobalt Strike beacon deployed on node",
    ],
    3: [
        "CVE-2023-0386 OverlayFS privilege escalation",
        "Sudo -l abuse on misconfigured service account",
        "DirtyCow kernel exploit executed",
    ],
    4: [
        "PsExec lateral move to DC01",
        "RDP session hijacking from compromised host",
        "WMI-based remote execution across subnet",
    ],
}

STAGE_NAMES = {
    0: "Reconnaissance",
    1: "Phishing / Initial Access",
    2: "Malware Propagation",
    3: "Privilege Escalation",
    4: "Lateral Movement",
}


class RedTeamAttacker:
    """
    Rule-based Red Team attacker.

    Simulates realistic attacker behaviour:
    - Starts at stage 0 (Recon)
    - Escalates stage based on success rate
    - Generates structured AttackEvent records for logging
    """

    def __init__(self, escalation_rate: float = 0.25, seed: Optional[int] = None):
        """
        Args:
            escalation_rate: Probability of escalating to next stage on success.
            seed: Random seed for reproducibility.
        """
        self.escalation_rate = escalation_rate
        self.current_stage = 0
        self.rng = random.Random(seed)
        self.events: list[AttackEvent] = []

    def choose_attack(self, defender_monitoring: float = 0.0) -> dict:
        """
        Choose the next attack action given current stage.

        Args:
            defender_monitoring: Monitoring bonus (0-0.3) that reduces success.

        Returns:
            Dict with attack metadata for the environment.
        """
        stage = self.current_stage
        payloads = PAYLOADS[stage]
        payload = self.rng.choice(payloads)

        # Adjusted success probability
        base_prob = 0.4 + stage * 0.1
        success_prob = max(0.05, base_prob - defender_monitoring)
        success = self.rng.random() < success_prob

        # Escalate stage probabilistically on success
        if success and self.rng.random() < self.escalation_rate and stage < 4:
            self.current_stage = min(4, stage + 1)

        node_id = f"NODE-{self.rng.randint(1, 10):02d}"

        event = AttackEvent(
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            stage=stage,
            stage_name=STAGE_NAMES[stage],
            attack_type=STAGE_NAMES[stage],
            target_node=node_id,
            payload=payload,
            success=success,
        )
        self.events.append(event)

        return {
            "stage": stage,
            "stage_name": STAGE_NAMES[stage],
            "payload": payload,
            "target_node": node_id,
            "success": success,
            "new_stage": self.current_stage,
        }

    def reset(self):
        """Reset attacker state for new episode."""
        self.current_stage = 0
        self.events.clear()

    def get_last_event(self) -> Optional[AttackEvent]:
        """Return the most recent attack event."""
        return self.events[-1] if self.events else None

    def summary(self) -> dict:
        """Return summary statistics for this attacker run."""
        if not self.events:
            return {}
        successes = sum(1 for e in self.events if e.success)
        return {
            "total_attacks": len(self.events),
            "successful_attacks": successes,
            "success_rate": round(successes / len(self.events), 3),
            "final_stage": self.current_stage,
            "stage_name": STAGE_NAMES[self.current_stage],
        }
