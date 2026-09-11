"""
train_control.py — Train the student agent against a STATIC opponent (control condition).

Same architecture, same total timesteps, same hyperparameters as train.py.
The only difference: the opponent config never changes.

This is your baseline.  Compare the final agent from this script
against the one from train.py to answer your research question.
"""

import asyncio
import os
import threading
import time
from queue import Queue

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback

from config import PPO_PARAMS, TOTAL_TIMESTEPS, CHECKPOINT_FREQ, GAME_HOST, GAME_PORT
from src.Environment import Environment
from src.ParameterizedOpponent import ParameterizedOpponent
from src.GymAI import GymAI

from pyftg.socket.aio.gateway import Gateway


# ── Static opponent config (medium difficulty — doesn't change) ──────────────
STATIC_CONFIG = {
    "aggression": 0.5,
    "combo_frequency": 0.3,
    "special_frequency": 0.2,
    "reaction_delay": 8,
}


def main():
    opponent = ParameterizedOpponent(config=STATIC_CONFIG.copy())

    obs_queue = Queue()
    action_queue = Queue()
    round_result_queue = Queue()

    gym_ai = GymAI(
        obs_queue=obs_queue,
        action_queue=action_queue,
        round_result_queue=round_result_queue,
        agent_name="ControlAgent",
    )

    env = Environment(
        host=GAME_HOST,
        port=GAME_PORT,
        player_number=True,
        character="GARNET",
        opponent_character="GARNET",
    )
    env.obs_queue = obs_queue
    env.action_queue = action_queue
    env.round_result_queue = round_result_queue
    env.connected = True

    def run_gateway():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        gateway = Gateway(host=GAME_HOST, port=GAME_PORT)
        gateway.register_ai("ControlAgent", gym_ai)
        gateway.register_ai("ParameterizedOpponent", opponent)
        loop.run_until_complete(
            gateway.run_game(
                characters=["GARNET", "GARNET"],
                agents=["ControlAgent", "ParameterizedOpponent"],
                game_number=9999,
            )
        )

    game_thread = threading.Thread(target=run_gateway, daemon=True)
    game_thread.start()
    time.sleep(2)

    # ── Simple logging callback (no Director, just records wins) ─────────
    round_log = []

    class LogCallback(BaseCallback):
        def _on_step(self) -> bool:
            e = self.training_env.envs[0]
            if hasattr(e, 'round_just_ended') and e.round_just_ended:
                result = e.last_round_result
                if result is not None:
                    won = result.remaining_hps[0] > result.remaining_hps[1]
                    round_log.append({
                        "round": len(round_log),
                        "student_won": won,
                        "student_hp": result.remaining_hps[0],
                        "opponent_hp": result.remaining_hps[1],
                        "elapsed_frames": result.elapsed_frame,
                    })
                    win_rate = sum(1 for r in round_log[-20:] if r["student_won"]) / min(len(round_log), 20)
                    self.logger.record("control/win_rate", win_rate)
                    self.logger.record("control/avg_hp", result.remaining_hps[0])
                    print(f"  Round end → win_rate={win_rate:.2f}  HP={result.remaining_hps}")
            return True

    os.makedirs("checkpoints/control", exist_ok=True)
    os.makedirs("logs/control", exist_ok=True)

    model = PPO(
        "MlpPolicy",
        env,
        tensorboard_log="./logs/control/",
        **PPO_PARAMS,
    )

    checkpoint_cb = CheckpointCallback(
        save_freq=CHECKPOINT_FREQ,
        save_path="./checkpoints/control/",
        name_prefix="student_static",
    )

    print("=" * 60)
    print("  CONTROL CONDITION — Static opponent (no Director)")
    print("=" * 60)

    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=[LogCallback(), checkpoint_cb],
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")

    model.save("checkpoints/control/student_static_final")

    # Save round log
    import csv
    with open("checkpoints/control/control_log.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=round_log[0].keys() if round_log else [])
        writer.writeheader()
        writer.writerows(round_log)

    print("Model and log saved.")


if __name__ == "__main__":
    main()