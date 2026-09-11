"""
FightingICEEnv — a Gymnasium src that wraps FightingICE via pyftg.

Usage:
    env = FightingICEEnv()
    obs, info = env.reset()
    while True:
        action = model.predict(obs)
        obs, reward, done, truncated, info = env.step(action)
        if done:
            obs, info = env.reset()

Architecture:
    The Java game runs on its own thread/process and calls GymAI callbacks.
    This env runs on the RL thread.  Two queues synchronize them:
      obs_queue    — GymAI pushes FrameData each frame, env.step() pulls
      action_queue — env.step() pushes action names, GymAI.processing() pulls
"""

import asyncio
import threading
from queue import Queue, Empty

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from pyftg.socket.aio.gateway import Gateway

from config import (
    ACTION_NAMES, NUM_ACTIONS, OBS_SIZE,
    MAX_HP, MAX_ENERGY, STAGE_WIDTH, STAGE_HEIGHT,
    MAX_SPEED, MAX_REMAINING_FRAME,
    GAME_HOST, GAME_PORT,
)
from src.GymAI import GymAI


class Environment(gym.Env):
    """
    Gymnasium wrapper for DareFightingICE.

    The game must already be running with --pyftg-mode before you create this env.
    On reset(), the env connects to the game and starts a match.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        host: str = GAME_HOST,
        port: int = GAME_PORT,
        player_number: bool = True,
        character: str = "GARNET",
        opponent_character: str = "GARNET",
        opponent_ai: str = "Sandbox",      # "Sandbox" = external / our parameterized AI
        game_num: int = 1,
        agent_name: str = "StudentAgent",
    ):
        super().__init__()

        # ── Spaces ────────────────────────────────────────────────────────────
        self.action_space = spaces.Discrete(NUM_ACTIONS)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(OBS_SIZE,), dtype=np.float32
        )

        # ── Connection parameters ─────────────────────────────────────────────
        self.host = host
        self.port = port
        self.player_number = player_number
        self.character = character
        self.opponent_character = opponent_character
        self.opponent_ai = opponent_ai
        self.game_num = game_num
        self.agent_name = agent_name

        # ── Synchronization ───────────────────────────────────────────────────
        self.obs_queue = Queue()
        self.action_queue = Queue()
        self.round_result_queue = Queue()

        # ── State ─────────────────────────────────────────────────────────────
        self.prev_frame = None
        self.last_frame = None
        self.connected = False
        self.game_thread = None

        # ── Public: last round result (read by Director callback) ─────────────
        self.last_round_result = None
        self.round_just_ended = False

    # ═════════════════════════════════════════════════════════════════════════
    # Gym interface
    # ═════════════════════════════════════════════════════════════════════════

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if not self.connected:
            self._start_game()

        # Wait for the first valid frame
        frame_data = self._wait_for_frame(timeout=60.0)
        obs = self._extract_obs(frame_data)

        self.prev_frame = frame_data
        self.last_frame = frame_data

        return obs, {}

    def step(self, action: int):
        self.round_just_ended = False

        # Send action to the game thread
        action_name = ACTION_NAMES[action]
        self.action_queue.put(action_name)

        # Wait for next frame
        try:
            frame_data = self.obs_queue.get(timeout=30.0)
        except Empty:
            # Timeout — game likely crashed or round ended unexpectedly
            obs = self._extract_obs(self.last_frame) if self.last_frame else np.zeros(OBS_SIZE, dtype=np.float32)
            return obs, 0.0, True, False, {}

        # ── Round ended? ──────────────────────────────────────────────────
        done = False
        if frame_data is None:
            done = True
            self.round_just_ended = True
            frame_data = self.last_frame
            # Grab round result
            try:
                self.last_round_result = self.round_result_queue.get_nowait()
            except Empty:
                self.last_round_result = None

        # ── Observation & reward ──────────────────────────────────────────
        obs = self._extract_obs(frame_data)
        reward = self._compute_reward(self.prev_frame, frame_data, done)

        self.prev_frame = frame_data
        if not done:
            self.last_frame = frame_data

        return obs, reward, done, False, {}

    def close(self):
        self.connected = False

    # ═════════════════════════════════════════════════════════════════════════
    # Observation extraction
    # ═════════════════════════════════════════════════════════════════════════

    def _extract_obs(self, frame_data) -> np.ndarray:
        """Turn a FrameData into a flat numpy vector."""
        if frame_data is None or frame_data.empty_flag:
            return np.zeros(OBS_SIZE, dtype=np.float32)

        my = frame_data.get_character(self.player_number)
        opp = frame_data.get_character(not self.player_number)

        if my is None or opp is None:
            return np.zeros(OBS_SIZE, dtype=np.float32)

        obs = np.array([
            # My state (11 features)
            my.hp / MAX_HP,
            my.energy / MAX_ENERGY,
            my.x / STAGE_WIDTH,
            my.y / STAGE_HEIGHT,
            my.speed_x / MAX_SPEED,
            my.speed_y / MAX_SPEED,
            1.0 if my.control else 0.0,
            my.remaining_frame / MAX_REMAINING_FRAME,
            my.action.to_int() / 55.0,  # what am I doing right now
            my.state.to_int() / 3.0,  # STAND/CROUCH/AIR/DOWN
            my.hit_count / 10.0,  # combo counter

            # Opponent state (8 features)
            opp.hp / MAX_HP,
            opp.energy / MAX_ENERGY,
            opp.x / STAGE_WIDTH,
            opp.y / STAGE_HEIGHT,
            opp.speed_x / MAX_SPEED,
            opp.speed_y / MAX_SPEED,
            1.0 if opp.control else 0.0,
            opp.remaining_frame / MAX_REMAINING_FRAME,
            opp.action.to_int() / 55.0,  # what is opponent doing NOW
            opp.state.to_int() / 3.0,  # opponent stance
            opp.hit_count / 10.0,  # opponent combo counter

            # Relational features (3 features)
            (my.x - opp.x) / STAGE_WIDTH,   # signed horizontal distance
            (my.y - opp.y) / STAGE_HEIGHT,   # signed vertical distance
            1.0 if my.front else -1.0,        # facing direction
            1.0 if opp.hit_confirm else 0.0,
            1.0 if my.hit_confirm else 0.0,
            len(frame_data.get_projectiles_by_player(not self.player_number)) / 3.0,  # incoming projectiles
        ], dtype=np.float32)

        return np.clip(obs, -1.0, 1.0)

    # ═════════════════════════════════════════════════════════════════════════
    # Reward computation
    # ═════════════════════════════════════════════════════════════════════════

    def _compute_reward(self, prev_frame, curr_frame, done: bool) -> float:
        """
        Frame‑level reward: HP delta (damage dealt − damage taken), normalized.
        Round‑end bonus: +1 for win, −1 for loss.
        """
        if prev_frame is None or curr_frame is None:
            return 0.0
        if prev_frame.empty_flag or curr_frame.empty_flag:
            return 0.0

        my_prev = prev_frame.get_character(self.player_number)
        my_curr = curr_frame.get_character(self.player_number)
        opp_prev = prev_frame.get_character(not self.player_number)
        opp_curr = curr_frame.get_character(not self.player_number)

        if any(x is None for x in [my_prev, my_curr, opp_prev, opp_curr]):
            return 0.0

        damage_dealt = (opp_prev.hp - opp_curr.hp) / MAX_HP
        damage_taken = (my_prev.hp - my_curr.hp) / MAX_HP
        reward = damage_dealt - damage_taken

        if done:
            my_hp = my_curr.hp
            opp_hp = opp_curr.hp
            if my_hp > opp_hp:
                reward += 1.0   # win bonus
            elif my_hp < opp_hp:
                reward -= 1.0   # loss penalty
            # tie → no bonus

        return reward

    # ═════════════════════════════════════════════════════════════════════════
    # Game connection (runs pyftg Gateway in a background thread)
    # ═════════════════════════════════════════════════════════════════════════

    def _start_game(self):
        """Launch the pyftg Gateway + GymAI in a daemon thread."""
        self.gym_ai = GymAI(
            obs_queue=self.obs_queue,
            action_queue=self.action_queue,
            round_result_queue=self.round_result_queue,
            agent_name=self.agent_name,
        )

        self.game_thread = threading.Thread(
            target=self._run_gateway,
            daemon=True,
        )
        self.game_thread.start()
        self.connected = True

    def _run_gateway(self):
        """Runs in the game thread — starts asyncio event loop for pyftg."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        gateway = Gateway(host=self.host, port=self.port)
        gateway.register_ai(self.agent_name, self.gym_ai)

        characters = [self.character, self.opponent_character]
        if self.player_number:
            agents = [self.agent_name, self.opponent_ai]
        else:
            agents = [self.opponent_ai, self.agent_name]

        loop.run_until_complete(
            gateway.run_game(characters, agents, self.game_num)
        )

    def _wait_for_frame(self, timeout: float = 30.0):
        """Pull frames from obs_queue until we get a valid one."""
        while True:
            try:
                frame = self.obs_queue.get(timeout=timeout)
            except Empty:
                return None
            if frame is None:
                continue  # skip round‑end sentinel during reset
            if not frame.empty_flag and frame.current_frame_number >= 0:
                return frame