"""Throwaway test — delete after it works."""
import asyncio, threading, time
from queue import Queue
from pyftg.socket.aio.gateway import Gateway
from src.GymAI import GymAI
from src.Environment import Environment
from src.ParameterizedOpponent import ParameterizedOpponent
from config import GAME_HOST, GAME_PORT

# Shared queues
obs_q = Queue()
act_q = Queue()
rr_q = Queue()

gym_ai = GymAI(obs_q, act_q, rr_q, agent_name="TestAgent")
opponent = ParameterizedOpponent()

# Start gateway in background
def run_gw():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    gw = Gateway(host=GAME_HOST, port=GAME_PORT)
    gw.register_ai("TestAgent", gym_ai)
    gw.register_ai("ParameterizedOpponent", opponent)
    loop.run_until_complete(
        gw.run_game(["GARNET","GARNET"], ["TestAgent","ParameterizedOpponent"], 1)
    )

threading.Thread(target=run_gw, daemon=True).start()
time.sleep(2)

# Wire queues into env
env = Environment()
env.obs_queue = obs_q
env.action_queue = act_q
env.round_result_queue = rr_q
env.connected = True

# Take 100 random actions and see if it stays alive
obs, _ = env.reset()
print(f"First obs: {obs}")
for i in range(100):
    action = env.action_space.sample()  # random
    obs, reward, done, trunc, info = env.step(action)
    if i % 20 == 0:
        print(f"  step {i}: reward={reward:.3f} done={done} obs[:4]={obs[:4]}")
    if done:
        print(f"Round ended at step {i}")
        break

print("Connection test passed!")