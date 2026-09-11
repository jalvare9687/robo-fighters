"""
AI Director — watches the student's performance and adjusts
the opponent's difficulty to keep the student learning.

This is the core of your research contribution.  The Director
operates on a slower timescale (per‑round) than the agent (per‑frame).

It implements a "thermostat" curriculum:
  - Student winning too easily  → raise difficulty
  - Student losing too much     → lower difficulty
  - Student at ~50% win rate    → hold steady (zone of proximal development)

The difficulty_level (0.0–1.0) maps to concrete opponent parameters
via _level_to_config().
"""

from collections import deque
from dataclasses import dataclass, field
from typing import List

import numpy as np

from config import (
    DIRECTOR_HISTORY,
    DIRECTOR_UP_THRESHOLD,
    DIRECTOR_DOWN_THRESHOLD,
    DIRECTOR_STEP,
    MAX_HP,
)

@dataclass
class RoundRecord:
    """One round of performance data."""
    student_won: bool
    student_hp: int
    opponent_hp: int
    elapsed_frames: int
    difficulty_level: float

class AIDirector:
    """
        Rule‑based adaptive curriculum controller.

        Call observe_round() after each round, then get_difficulty_config()
        to get the new opponent parameters for the next round.
        """
    def __init__(self, initial_level: float = 0.3):
        self.difficulty_level = initial_level
        self.history: deque = deque(maxlen=DIRECTOR_HISTORY)
        self.all_records: List[RoundRecord] = []  # full log for analysis

    def observe_round(self, remaining_hps:list, elapsed_frame: int):
        """
                Record the result of a round.

                Args:
                    remaining_hps: [student_hp, opponent_hp]
                    elapsed_frame: how many frames the round lasted
                """
        student_hp = remaining_hps[0]
        opponent_hp = remaining_hps[1]
        student_won = student_hp > opponent_hp

        record = RoundRecord (
            student_won=student_won,
            student_hp=student_hp,
            opponent_hp=opponent_hp,
            elapsed_frames=elapsed_frame,
            difficulty_level=self.difficulty_level
        )
        self.history.append(record)
        self.all_records.append(record)

    def get_difficulty_config(self) -> dict:
        if len(self.history) >= 5:
            self._adjust_difficulty()
        return self._level_to_config(self.difficulty_level)

    def _adjust_difficulty(self):
        """Core curriculum logic — thermostat controller."""
        recent_win_rate = sum(1 for r in self.history if r.student_won) / len(self.history)

        if recent_win_rate > DIRECTOR_UP_THRESHOLD:
            print("increasing difficulty level")
            self.difficulty_level = min(1.0, self.difficulty_level + DIRECTOR_STEP)
        elif recent_win_rate < DIRECTOR_DOWN_THRESHOLD:
            # Student is struggling → easier
            self.difficulty_level = max(0.0, self.difficulty_level - DIRECTOR_STEP)

    def _level_to_config(self, level:float) -> dict:
        return {
            "aggression": round(0.2 + 0.6 * level, 3),  # 0.2 – 0.8
            "combo_frequency": round(0.1 + 0.5 * level, 3),  # 0.1 – 0.6
            "special_frequency": round(0.05 + 0.35 * level, 3),  # 0.05 – 0.4
            "reaction_delay": int(12 - 10 * level),  # 12 – 2 frames
        }

    def get_metrics(self) -> dict:
        """Return current Director stats for logging."""
        if not self.history:
            return {
                "director/difficulty_level": self.difficulty_level,
                "director/win_rate": 0.0,
                "director/avg_student_hp": 0.0,
            }

        win_rate = sum(1 for r in self.history if r.student_won) / len(self.history)
        avg_hp = np.mean([r.student_hp for r in self.history])
        return {
            "director/difficulty_level": self.difficulty_level,
            "director/win_rate": win_rate,
            "director/avg_student_hp": avg_hp,
            "director/rounds_played": len(self.all_records),
        }

    def save_log(self, path: str):
        """Save full training history to CSV for later analysis."""
        import csv
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "round", "student_won", "student_hp", "opponent_hp",
                "elapsed_frames", "difficulty_level",
            ])
            for i, r in enumerate(self.all_records):
                writer.writerow([
                    i, r.student_won, r.student_hp, r.opponent_hp,
                    r.elapsed_frames, r.difficulty_level,
                ])