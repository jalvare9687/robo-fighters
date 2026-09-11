"""
human_vs_ai.py — Play against your trained agent.

You play P1 (left) with keyboard. The AI plays P2 (right).

Start Java in visual mode:
    --pyftg-mode --limithp 400 400

Then:
    python human_vs_ai.py

Keyboard controls (default FightingICE):
    Movement:  A/D = left/right, W = jump, S = crouch
    Attacks:    J = A button, K = B button, U = C button
    Guard:      Hold back (A or D away from opponent)
"""

import asyncio
import threading
import time
import argparse
from queue import Queue, Empty

import numpy as np
from stable_baselines3 import PPO

from src.config import GAME_HOST, GAME_PORT, ACTION_NAMES
from src.GymAI import GymAI
from pyftg.socket.aio.gateway import Gateway


def run_predict_loop(obs_q, act_q, rr_q, model, player_number, rounds):
    """Predict loop for the AI player."""
    rounds_played = 0
    wins = 0

    while rounds_played < rounds:
        try:
            frame = obs_q.get(timeout=120.0)
        except Empty:
            break

        if frame is None:
            try:
                result = rr_q.get_nowait()
                idx = 0 if player_number else 1
                opp_idx = 1 - idx
                won = result.remaining_hps[idx] > result.remaining_hps[opp_idx]
                if won:
                    wins += 1
                rounds_played += 1
                print(f"  Round {rounds_played}: "
                      f"AI {'WIN' if won else 'LOSS'}  "
                      f"HP: {result.remaining_hps[idx]} (AI) vs {result.remaining_hps[opp_idx]} (Human)")
            except Empty:
                rounds_played += 1
            continue

        if frame.empty_flag or frame.current_frame_number < 0:
            act_q.put("NEUTRAL")
            continue

        my = frame.get_character(player_number)
        opp = frame.get_character(not player_number)

        if my is None or opp is None:
            act_q.put("NEUTRAL")
            continue

        obs = np.array([
            my.hp / 400.0, my.energy / 300.0,
            my.x / 960.0, my.y / 640.0,
            my.speed_x / 15.0, my.speed_y / 15.0,
            1.0 if my.control else 0.0,
            my.remaining_frame / 70.0,
            my.action.to_int() / 55.0,
            my.state.to_int() / 3.0,
            my.hit_count / 10.0,

            opp.hp / 400.0, opp.energy / 300.0,
            opp.x / 960.0, opp.y / 640.0,
            opp.speed_x / 15.0, opp.speed_y / 15.0,
            1.0 if opp.control else 0.0,
            opp.remaining_frame / 70.0,
            opp.action.to_int() / 55.0,
            opp.state.to_int() / 3.0,
            opp.hit_count / 10.0,

            (my.x - opp.x) / 960.0,
            (my.y - opp.y) / 640.0,
            1.0 if my.front else -1.0,
            1.0 if opp.hit_confirm else 0.0,
            1.0 if my.hit_confirm else 0.0,
            len(frame.get_projectiles_by_player(not player_number)) / 3.0,
        ], dtype=np.float32)
        obs = np.clip(obs, -1.0, 1.0)

        action, _ = model.predict(obs, deterministic=True)
        act_q.put(ACTION_NAMES[int(action)])

    print(f"\n  Final: AI won {wins}/{rounds_played}")


def main():
    parser = argparse.ArgumentParser(description="Human vs trained AI")
    parser.add_argument("--model", type=str,
                        default="src/checkpoints/experimental/student_director_final",
                        help="Path to the AI model")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--character", type=str, default="GARNET")
    args = parser.parse_args()

    obs_q = Queue()
    act_q = Queue()
    rr_q = Queue()

    ai = GymAI(obs_q, act_q, rr_q, agent_name="TrainedAI")
    model = PPO.load(args.model)

    def run_gateway():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        gw = Gateway(host=GAME_HOST, port=GAME_PORT)
        gw.register_ai("TrainedAI", ai)
        loop.run_until_complete(
            gw.run_game(
                [args.character, args.character],
                ["Keyboard", "TrainedAI"],
                args.rounds,
            )
        )

    print(f"\n{'='*50}")
    print(f"  HUMAN vs AI — {args.rounds} rounds")
    print(f"  P1 (Left):  YOU (keyboard)")
    print(f"  P2 (Right): Trained AI")
    print(f"{'='*50}")
    print(f"\n  Controls:")
    print(f"    Move:    A/D = left/right, W = jump, S = crouch")
    print(f"    Attack:  J = A, K = B, U = C")
    print(f"    Guard:   Hold direction away from opponent")
    print(f"\n  Waiting for game connection...\n")

    gw_thread = threading.Thread(target=run_gateway, daemon=True)
    gw_thread.start()
    time.sleep(2)

    predict_thread = threading.Thread(
        target=run_predict_loop,
        args=(obs_q, act_q, rr_q, model, False, args.rounds),
        daemon=True,
    )
    predict_thread.start()
    predict_thread.join(timeout=args.rounds * 180)


if __name__ == "__main__":
    main()