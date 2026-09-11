"""
run_demo.py — Run a trained agent in visual mode for your demo.

This connects your best trained model as P1 and a parameterized
opponent as P2, then plays live matches with the game's graphics ON.

Before running:
  1. Start FightingICE in STANDARD mode (not lightweight):
     java -cp "FightingICE.jar:lib/*" Main --pyftg-mode --limithp 400 400

     NOTE: no --lightweight-mode flag → graphics will render!

  2. python run_demo.py

You can also do agent‑vs‑agent by running two instances:
  Terminal 1: python run_demo.py --model checkpoints/experimental/student_director_final --player 1
  Terminal 2: python run_demo.py --model checkpoints/control/student_static_final --player 2
"""

import argparse
import asyncio
import threading
import time
from queue import Queue

from stable_baselines3 import PPO

from config import GAME_HOST, GAME_PORT
from src.Environment import Environment
from src.ParameterizedOpponent import ParameterizedOpponent
from src.GymAI import GymAI
from pyftg.socket.aio.gateway import Gateway


def main():
    parser = argparse.ArgumentParser(description="Run a trained agent for demo")
    parser.add_argument("--model", type=str,
                        default="checkpoints/experimental/student_director_final",
                        help="Path to the saved PPO model (without .zip)")
    parser.add_argument("--player", type=int, default=1, choices=[1, 2],
                        help="Which player slot (1 or 2)")
    parser.add_argument("--opponent", type=str, default="param",
                        choices=["param", "sandbox"],
                        help="'param' for parameterized opponent, 'sandbox' for external")
    parser.add_argument("--difficulty", type=float, default=1.0,
                        help="Opponent difficulty level (0.0–1.0)")
    parser.add_argument("--rounds", type=int, default=5,
                        help="Number of rounds to play")
    parser.add_argument("--character", type=str, default="GARNET",
                        help="Character for the agent")
    args = parser.parse_args()

    player_number = args.player == 1

    level = args.difficulty
    config = {
        "aggression": round(0.2 + 0.6 * level, 3),
        "combo_frequency": round(0.1 + 0.5 * level, 3),
        "special_frequency": round(0.05 + 0.35 * level, 3),
        "reaction_delay": int(12 - 10 * level),
    }

    opponent = ParameterizedOpponent(config)

    # ── Queues ────────────────────────────────────────────────────────────
    obs_queue = Queue()
    action_queue = Queue()
    round_result_queue = Queue()

    gym_ai = GymAI(
        obs_queue=obs_queue,
        action_queue=action_queue,
        round_result_queue=round_result_queue,
        agent_name="DemoAgent",
    )

    env = Environment (
        host=GAME_HOST, port=GAME_PORT,
        player_number=player_number,
        character=args.character,
        opponent_character=args.character,
    )
    env.obs_queue = obs_queue
    env.action_queue = action_queue
    env.round_result_queue = round_result_queue
    env.connected = True

    # ── Gateway ───────────────────────────────────────────────────────────
    def run_gateway():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        gateway = Gateway(host=GAME_HOST, port=GAME_PORT)
        gateway.register_ai("DemoAgent", gym_ai)
        if args.opponent == "param":
            gateway.register_ai("ParameterizedOpponent", opponent)
            agents = (
                ["DemoAgent", "ParameterizedOpponent"]
                if player_number
                else ["ParameterizedOpponent", "DemoAgent"]
            )
        else:
            agents = (
                ["DemoAgent", "Sandbox"]
                if player_number
                else ["Sandbox", "DemoAgent"]
            )
        loop.run_until_complete(
            gateway.run_game(
                characters=[args.character, args.character],
                agents=agents,
                game_number=args.rounds,
            )
        )

    game_thread = threading.Thread(target=run_gateway, daemon=True)
    game_thread.start()
    time.sleep(2)

    # ── Load model ────────────────────────────────────────────────────────
    print(f"Loading model from {args.model}...")
    model = PPO.load(args.model)

    # ── Run demo ──────────────────────────────────────────────────────────
    print(f"\n{'='*50}")
    print(f"  LIVE DEMO — {args.rounds} rounds at difficulty {args.difficulty}")
    print(f"{'='*50}\n")

    rounds_played = 0
    wins = 0

    obs, _ = env.reset()
    while rounds_played < args.rounds:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)

        if done:
            if env.last_round_result:
                r = env.last_round_result
                student_idx = 0 if player_number else 1
                opp_idx = 1 - student_idx
                won = r.remaining_hps[student_idx] > r.remaining_hps[opp_idx]
                if won:
                    wins += 1
                rounds_played += 1
                print(f"  Round {rounds_played}: "
                      f"{'WIN' if won else 'LOSS'}  "
                      f"HP: {r.remaining_hps[student_idx]} vs {r.remaining_hps[opp_idx]}")

            if rounds_played < args.rounds:
                obs, _ = env.reset()

    print(f"\nFinal: {wins}/{rounds_played} wins ({wins/rounds_played*100:.0f}%)")


if __name__ == "__main__":
    main()