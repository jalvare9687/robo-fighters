"""
train.py — Train the student agent with the AI Director (experimental condition).

Before running:
  1. Start the Java game:
     java -cp "FightingICE.jar:lib/*" Main --lightweight-mode \
          --limithp 400 400 --pyftg-mode --mute --fast-mode

  2. Run this script:
     python train.py

The script connects two AIs to the game:
  - P1: GymAI (student, controlled by PPO)
  - P2: ParameterizedOpponent (controlled by Director)

The Director adjusts P2's difficulty between rounds based on P1's performance.
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
from director import AIDirector
from src.GymAI import GymAI

from pyftg.socket.aio.gateway import Gateway


def main():
    class DirectorCallback(BaseCallback):
        def __init__(self, env_ref, director_ref, opponent_ref):
            super().__init__()
            self.env_ref = env_ref  # direct reference, no wrapping
            self.director_ref = director_ref
            self.opponent_ref = opponent_ref
        """
                Fires every step.  When a round ends (env.round_just_ended),
                it reads the result, feeds the Director, and updates the opponent.
                """
        def _on_step(self) -> bool:
            if self.env_ref.round_just_ended:
                result = self.env_ref.last_round_result
                if result is not None:
                    self.director_ref.observe_round(
                        remaining_hps=result.remaining_hps,
                        elapsed_frame=result.elapsed_frame,
                    )

                    new_config = self.director_ref.get_difficulty_config()
                    self.opponent_ref.update_config(new_config)

                    metrics = self.director_ref.get_metrics()
                    for key, value in metrics.items():
                        self.logger.record(key, value)

                    print(f"  Round end → win_rate={metrics['director/win_rate']:.2f}  "
                          f"difficulty={metrics['director/difficulty_level']:.2f}  "
                          f"HP={result.remaining_hps}")
            return True

    #create director & opponent
    director = AIDirector(initial_level=0.3)
    initial_config = director.get_difficulty_config()
    opponent = ParameterizedOpponent(initial_config)

    #shared queues between GymAI and env
    obs_queue = Queue()
    action_queue = Queue()
    round_result_queue = Queue()

    gymAi = GymAI(
        obs_queue=obs_queue,
        action_queue=action_queue,
        round_result_queue=round_result_queue,
        agent_name="StudentAgent",
    )

    env = Environment(
        host=GAME_HOST,
        port=GAME_PORT,
        player_number=True,
        character="GARNET",
        opponent_character="GARNET",
    )

    # Inject our queues directly (bypass the env's own _start_game)
    env.obs_queue = obs_queue
    env.action_queue = action_queue
    env.round_result_queue = round_result_queue
    env.connected = True

    director_cb = DirectorCallback(env, director, opponent)

    def run_gateway():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        gateway = Gateway(host=GAME_HOST, port=GAME_PORT)
        gateway.register_ai("StudentAgent", gymAi)
        gateway.register_ai("ParameterizedOpponent", opponent)

        loop.run_until_complete(
            gateway.run_game(
                characters=["GARNET", "GARNET"],
                agents=["StudentAgent", "ParameterizedOpponent"],
                game_number=9999,  # large number — we control stopping via timesteps
            )
        )

    game_thread = threading.Thread(target=run_gateway, daemon=True)
    game_thread.start()
    time.sleep(2)


    os.makedirs("checkpoints/experimental", exist_ok=True)
    os.makedirs("logs/experimental", exist_ok=True)

    model = PPO(
        "MlpPolicy",
        env,
        tensorboard_log="./logs/experimental/",
        **PPO_PARAMS,
    )
    checkpoint_cb = CheckpointCallback(
        save_freq=CHECKPOINT_FREQ,
        save_path="./checkpoints/experimental/",
        name_prefix="student_director",
    )

    #Train
    print("=" * 60)
    print("  EXPERIMENTAL CONDITION — Director‑adapted opponent")
    print("=" * 60)

    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=[director_cb, checkpoint_cb],
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")

    # ── Save ─────────────────────────────────────────────────────────────
    model.save("checkpoints/experimental/student_director_final")
    director.save_log("checkpoints/experimental/director_log.csv")
    print("Model and Director log saved.")

if __name__ == "__main__":
    main()