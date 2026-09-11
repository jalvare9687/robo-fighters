"""
GymAI — the bridge between FightingICE's game loop and your Gymnasium env.

pyftg calls these methods on its own async/thread schedule:
    initialize → (get_information → processing → input) × N → round_end → game_end

Your RL thread calls env.step(), which blocks on obs_queue.  The processing()
method here feeds frames into that queue and pulls actions back out.

This is the trickiest part of the whole project.  If you get stuck, check that:
  1. obs_queue and action_queue are the SAME objects in GymAI and FightingICEEnv.
  2. The sentinel `None` in obs_queue only fires once per round‑end.
  3. You never call command_center.command_call() when the skill queue is busy.
"""

import threading
from queue import Queue

from pyftg.aiinterface.ai_interface import AIInterface
from pyftg.aiinterface.command_center import CommandCenter
from pyftg.models.audio_data import AudioData
from pyftg.models.frame_data import FrameData
from pyftg.models.game_data import GameData
from pyftg.models.key import Key
from pyftg.models.round_result import RoundResult
from pyftg.models.screen_data import ScreenData


class GymAI(AIInterface):
    """
    Implements pyftg's AIInterface and synchronizes with a Gym env
    through two thread‑safe queues.

    obs_queue:    GymAI pushes FrameData every frame → Gym env pops it in step()
    action_queue: Gym env pushes action string      → GymAI pops it in processing()
    round_end_event: signals the Gym env that the round finished
    """

    def __init__(
        self,
        obs_queue: Queue,
        action_queue: Queue,
        round_result_queue: Queue,
        agent_name: str = "GymAgent",
        blind: bool = False,
    ):
        self.obs_queue = obs_queue
        self.action_queue = action_queue
        self.round_result_queue = round_result_queue
        self._name = agent_name
        self._blind = blind
        self.command_center = CommandCenter()
        self.frame_data: FrameData = None
        self.is_control: bool = False
        self.player_number: bool = True  # True = P1

    # ── pyftg required methods ───────────────────────────────────────────────

    def name(self) -> str:
        return self._name

    def is_blind(self) -> bool:
        return self._blind

    def initialize(self, game_data: GameData, player_number: bool):
        self.player_number = player_number
        self.game_data = game_data

    def get_non_delay_frame_data(self, frame_data: FrameData):
        pass  # we use delayed data (more realistic for competition)

    def get_information(self, frame_data: FrameData, is_control: bool):
        self.frame_data = frame_data
        self.is_control = is_control
        if frame_data and not frame_data.empty_flag:
            self.command_center.set_frame_data(frame_data, self.player_number)

    def get_screen_data(self, screen_data: ScreenData):
        pass

    def get_audio_data(self, audio_data: AudioData):
        pass

    def processing(self):
        """
        Called by pyftg every frame on the game thread.
        This is where we synchronize: push observation, wait for action.
        """
        # Skip invalid frames (round hasn't started or frame is empty)
        if self.frame_data is None or self.frame_data.empty_flag:
            return
        if self.frame_data.current_frame_number < 0:
            return

        # 1. Send current observation to the RL thread
        self.obs_queue.put(self.frame_data)

        # 2. Wait for the RL thread to pick an action
        #    (env.step() will put a string like "STAND_A" into the queue)
        action_name = self.action_queue.get()

        # 3. Convert action to key commands (only if no keys pending)
        if not self.command_center.get_skill_flag():
            self.command_center.command_call(action_name)

    def input(self) -> Key:
        """Return the next key press to the game engine."""
        return self.command_center.get_skill_key()

    def round_end(self, round_result: RoundResult):
        """Signal the RL thread that the round ended."""
        self.round_result_queue.put(round_result)
        self.obs_queue.put(None)  # sentinel → env.step() knows round is done

    def game_end(self):
        pass

    def close(self):
        pass