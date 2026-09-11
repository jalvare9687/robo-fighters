"""
evaluate.py — Test trained agents against unseen opponent configurations.

This answers your research question:
  "Does the Director‑trained agent generalize better than the static‑trained one?"

It loads both saved models and runs them against 5 opponent configs
that neither agent saw during training.  Results are printed as a table
and saved to CSV.

Usage:
  1. Start the Java game with --pyftg-mode
  2. python evaluate.py
"""

import asyncio
import os
import csv
import threading
import time
from queue import Queue

import numpy as np
from stable_baselines3 import PPO

from config import GAME_HOST, GAME_PORT, MAX_HP
from src.Environment import Environment
from src.ParameterizedOpponent import ParameterizedOpponent
from src.GymAI import GymAI

from pyftg.socket.aio.gateway import Gateway

TEST_CONFIGS = {
    "passive": {
        "aggression": 0.1, "combo_frequency": 0.0,
        "special_frequency": 0.0, "reaction_delay": 15,
    },
    "balanced": {
        "aggression": 0.5, "combo_frequency": 0.3,
        "special_frequency": 0.2, "reaction_delay": 8,
    },
    "aggressive": {
        "aggression": 0.9, "combo_frequency": 0.7,
        "special_frequency": 0.5, "reaction_delay": 2,
    },
    "combo_heavy": {
        "aggression": 0.3, "combo_frequency": 0.8,
        "special_frequency": 0.1, "reaction_delay": 5,
    },
    "special_heavy": {
        "aggression": 0.8, "combo_frequency": 0.1,
        "special_frequency": 0.8, "reaction_delay": 3,
    },
}

ROUNDS_PER_CONFIG = 30

def run_evaluation(model_path: str, model_label: str, config_name: str, config: dict):
    """Run one model against one opponent config for ROUNDS_PER_CONFIG rounds."""
    opponent = ParameterizedOpponent(config=config.copy())

    obs_queue = Queue()
    action_queue = Queue()
    round_result_queue = Queue()

    gymAI = GymAI(
        obs_queue=obs_queue,
        action_queue=action_queue,
        round_result_queue=round_result_queue,
        agent_name=f"Eval_{model_label}",
    )

    env = Environment(
        host=GAME_HOST, port=GAME_PORT,
        player_number=True,
        character="GARNET", opponent_character="GARNET",
    )

    env.obs_queue = obs_queue
    env.action_queue = action_queue
    env.round_result_queue = round_result_queue
    env.connected = True

    def run_gateway():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        gateway = Gateway(host=GAME_HOST, port=GAME_PORT)
        gateway.register_ai(f"Eval_{model_label}", gymAI)
        gateway.register_ai("ParameterizedOpponent", opponent)
        loop.run_until_complete(
            gateway.run_game(
                characters=["GARNET", "GARNET"],
                agents=[f"Eval_{model_label}", "ParameterizedOpponent"],
                game_number=ROUNDS_PER_CONFIG,
            )
        )

    game_thread = threading.Thread(target=run_gateway, daemon=True)
    game_thread.start()
    time.sleep(2)

    model = PPO.load(model_path)

    wins = 0
    total_hp = 0
    rounds_played = 0

    obs,_ = env.reset()
    while rounds_played < ROUNDS_PER_CONFIG:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)

        if done:
            if env.last_round_result:
                r = env.last_round_result
                if r.remaining_hps[0] > r.remaining_hps[1]:
                    wins += 1
                total_hp += r.remaining_hps[0]
                rounds_played += 1

            if rounds_played < ROUNDS_PER_CONFIG:
                obs, _ = env.reset()

    win_rate = wins / ROUNDS_PER_CONFIG
    avg_hp = total_hp / ROUNDS_PER_CONFIG

    return {
        "model": model_label,
        "config": config_name,
        "win_rate": round(win_rate, 3),
        "avg_hp": round(avg_hp, 1),
        "wins": wins,
        "rounds": ROUNDS_PER_CONFIG,
    }

def main():
    models = {

        "control": "checkpoints/control/student_static_final",
    }

    for label, path in models.items():
        if not os.path.exists(path + ".zip"):
            print(f"WARNING: Model not found at {path}.zip — skipping {label}")
            models.pop(label)
            break

    if not models:
        print("No trained models found.  Run train.py and train_control.py first.")
        return

    results = []

    for model_label, model_path in models.items():
        for config_name, config in TEST_CONFIGS.items():
            input(f"\nRestart Java, then press Enter to evaluate {model_label} vs {config_name}...")
            print(f"Evaluating {model_label} vs {config_name}...")
            result = run_evaluation(model_path, model_label, config_name, config)
            results.append(result)
            print(f"  → win_rate={result['win_rate']:.2f}  avg_hp={result['avg_hp']:.0f}")

        # ── Print results table ──────────────────────────────────────────────
        print("\n" + "=" * 65)
        print(f"{'Model':<15} {'Config':<15} {'Win Rate':>10} {'Avg HP':>10}")
        print("-" * 65)
        for r in results:
            print(f"{r['model']:<15} {r['config']:<15} {r['win_rate']:>10.2f} {r['avg_hp']:>10.1f}")
        print("=" * 65)

        # ── Save to CSV ──────────────────────────────────────────────────────
        os.makedirs("results", exist_ok=True)
        with open("results/generalization_test.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print("\nResults saved to results/generalization_test.csv")

if __name__ == "__main__":
    main()