"""
fight.py — Two trained agents fight each other live.

Start Java in visual mode:
    --pyftg-mode --limithp 400 400

Then:
    python fight.py
"""

import asyncio
import threading
import time
import argparse
from queue import Queue, Empty

import numpy as np
from stable_baselines3 import PPO

from config import GAME_HOST, GAME_PORT, OBS_SIZE, ACTION_NAMES
from src.GymAI import GymAI
from pyftg.socket.aio.gateway import Gateway


def make_agent(name, model_path, player_number):
    """Create a GymAI + queues + loaded model for one player."""
    obs_q = Queue()
    act_q = Queue()
    rr_q = Queue()
    ai = GymAI(obs_q, act_q, rr_q, agent_name=name)
    model = PPO.load(model_path)
    return ai, obs_q, act_q, rr_q, model


def run_predict_loop(name, obs_q, act_q, rr_q, model, player_number, rounds, results):
    """
    Runs on its own thread. Reads frames from obs_q,
    predicts actions, pushes them to act_q.
    """
    rounds_played = 0
    wins = 0

    while rounds_played < rounds:
        try:
            frame = obs_q.get(timeout=60.0)
        except Empty:
            break

        if frame is None:
            # Round ended
            try:
                result = rr_q.get_nowait()
                idx = 0 if player_number else 1
                opp_idx = 1 - idx
                won = result.remaining_hps[idx] > result.remaining_hps[opp_idx]
                if won:
                    wins += 1
                rounds_played += 1
                print(f"  {name}: Round {rounds_played} → "
                      f"{'WIN' if won else 'LOSS'}  "
                      f"HP: {result.remaining_hps[idx]} vs {result.remaining_hps[opp_idx]}")
            except Empty:
                rounds_played += 1
            continue

        if frame.empty_flag or frame.current_frame_number < 0:
            act_q.put("NEUTRAL")
            continue

        # Extract obs
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

    results[name] = {"wins": wins, "rounds": rounds_played}


def main():
    parser = argparse.ArgumentParser(description="Agent vs Agent fight")
    parser.add_argument("--p1", type=str,
                        default="checkpoints/experimental/student_director_final",
                        help="P1 model path")
    parser.add_argument("--p2", type=str,
                        default="checkpoints/control/student_static_final",
                        help="P2 model path")
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--character", type=str, default="GARNET")
    args = parser.parse_args()

    # Create both agents
    p1_ai, p1_obs, p1_act, p1_rr, p1_model = make_agent(
        "Director-Agent", args.p1, True
    )
    p2_ai, p2_obs, p2_act, p2_rr, p2_model = make_agent(
        "Control-Agent", args.p2, False
    )

    # Gateway thread
    def run_gateway():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        gw = Gateway(host=GAME_HOST, port=GAME_PORT)
        gw.register_ai("Director-Agent", p1_ai)
        gw.register_ai("Control-Agent", p2_ai)
        loop.run_until_complete(
            gw.run_game(
                [args.character, args.character],
                ["Director-Agent", "Control-Agent"],
                args.rounds,
            )
        )

    gw_thread = threading.Thread(target=run_gateway, daemon=True)
    gw_thread.start()
    time.sleep(2)

    # Prediction loops — one thread per agent
    results = {}

    p1_thread = threading.Thread(
        target=run_predict_loop,
        args=("Director-Agent", p1_obs, p1_act, p1_rr, p1_model, True, args.rounds, results),
        daemon=True,
    )
    p2_thread = threading.Thread(
        target=run_predict_loop,
        args=("Control-Agent", p2_obs, p2_act, p2_rr, p2_model, False, args.rounds, results),
        daemon=True,
    )

    print(f"\n{'='*55}")
    print(f"  AGENT vs AGENT — {args.rounds} rounds")
    print(f"  P1 (Left):  Director-trained")
    print(f"  P2 (Right): Control (static)")
    print(f"{'='*55}\n")

    p1_thread.start()
    p2_thread.start()

    p1_thread.join(timeout=args.rounds * 120)
    p2_thread.join(timeout=args.rounds * 120)

    # Final summary
    print(f"\n{'='*55}")
    for name, r in results.items():
        print(f"  {name}: {r['wins']}/{r['rounds']} wins")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()