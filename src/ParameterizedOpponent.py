"""
ParameterizedOpponent — a scripted AI whose behavior is controlled
by a difficulty configuration dict.

The AI Director adjusts these knobs between rounds.  The opponent is
intentionally simple; it exists to provide a *tunable* challenge,
not to be a world‑class fighter.

Difficulty config keys:
    aggression       (0.0–1.0)  probability of attacking vs being passive
    combo_frequency  (0.0–1.0)  probability of chaining attacks
    special_frequency(0.0–1.0)  probability of using special moves
    reaction_delay   (0–15)     frames of "thinking time" before acting
"""

import random

from pyftg.aiinterface.ai_interface import AIInterface
from pyftg.aiinterface.command_center import CommandCenter
from pyftg.models.audio_data import AudioData
from pyftg.models.frame_data import FrameData
from pyftg.models.game_data import GameData
from pyftg.models.key import Key
from pyftg.models.round_result import RoundResult
from pyftg.models.screen_data import ScreenData


# ─── Default "medium" config ─────────────────────────────────────────────────
DEFAULT_CONFIG = {
    "aggression": 0.5,
    "combo_frequency": 0.3,
    "special_frequency": 0.2,
    "reaction_delay": 8,
}

BASIC_ATTACKS = ["STAND_A", "STAND_B", "CROUCH_A", "CROUCH_B"]
FORWARD_ATTACKS = ["STAND_FA", "STAND_FB", "CROUCH_FA", "CROUCH_FB"]
SPECIAL_ATTACKS = [
    "STAND_D_DF_FA", "STAND_D_DF_FB",
    "STAND_F_D_DFA", "STAND_F_D_DFB",
    "STAND_D_DB_BA", "STAND_D_DB_BB",
    "STAND_D_DF_FC",
]
AIR_ATTACKS = ["AIR_A", "AIR_B", "AIR_FA", "AIR_FB", "AIR_DA", "AIR_DB"]
MOVEMENT = ["FORWARD_WALK", "DASH"]
DEFENSIVE = ["STAND_GUARD", "CROUCH_GUARD", "BACK_STEP"]


class ParameterizedOpponent(AIInterface):
    """
    Scripted opponent with tunable difficulty parameters.
    Plug this into pyftg as the P2 agent.
    """

    def __init__(self, config: dict = None):
        self.config = config or DEFAULT_CONFIG.copy()
        self.command_center = CommandCenter()
        self.frame_data: FrameData = None
        self.player_number: bool = False  # usually P2
        self.delay_counter = 0

    # ── Allow live updates from the Director ──────────────────────────────────
    def update_config(self, new_config: dict):
        """Called by the Director between rounds."""
        self.config.update(new_config)

    # ── pyftg AIInterface ─────────────────────────────────────────────────────

    def name(self) -> str:
        return "ParameterizedOpponent"

    def is_blind(self) -> bool:
        return False

    def initialize(self, game_data: GameData, player_number: bool):
        self.player_number = player_number
        self.game_data = game_data

    def get_non_delay_frame_data(self, frame_data: FrameData):
        pass

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
        if self.frame_data is None or self.frame_data.empty_flag:
            return
        if self.frame_data.current_frame_number < 0:
            return
        if self.command_center.get_skill_flag():
            return  # still executing previous action

        # ── Reaction delay ────────────────────────────────────────────────
        self.delay_counter += 1
        if self.delay_counter < self.config["reaction_delay"]:
            return
        self.delay_counter = 0

        # ── Read game state ───────────────────────────────────────────────
        my = self.frame_data.get_character(self.player_number)
        opp = self.frame_data.get_character(not self.player_number)
        if my is None or opp is None:
            return

        distance = abs(my.x - opp.x)

        # ── Decide action ─────────────────────────────────────────────────
        action = self._decide(my, opp, distance)
        self.command_center.command_call(action)

    def input(self) -> Key:
        return self.command_center.get_skill_key()

    def round_end(self, round_result: RoundResult):
        self.delay_counter = 0

    def game_end(self):
        pass

    def close(self):
        pass

    # ── Decision logic ────────────────────────────────────────────────────────

    def _decide(self, my, opp, distance: int) -> str:
        aggression = self.config["aggression"]
        combo_freq = self.config["combo_frequency"]
        special_freq = self.config["special_frequency"]

        # If opponent is in the air, try an anti‑air
        if opp.y < 300 and distance < 250:
            if random.random() < aggression:
                return random.choice(["STAND_F_D_DFA", "STAND_B", "AIR_A"])

        # Passive behavior gate
        if random.random() > aggression:
            if distance < 120:
                return random.choice(DEFENSIVE)
            return random.choice(MOVEMENT + ["NEUTRAL"])

        # Close range — basic or combo
        if distance < 150:
            roll = random.random()
            if roll < special_freq and my.energy >= 50:
                return random.choice(SPECIAL_ATTACKS)
            elif roll < special_freq + combo_freq:
                # "Combo": just pick a fast attack (the game chains if timed right)
                return random.choice(BASIC_ATTACKS)
            else:
                return random.choice(BASIC_ATTACKS + FORWARD_ATTACKS)

        # Mid range — forward attacks or approach
        if distance < 300:
            roll = random.random()
            if roll < special_freq and my.energy >= 50:
                return random.choice(SPECIAL_ATTACKS)
            elif roll < 0.5:
                return random.choice(FORWARD_ATTACKS)
            else:
                return "DASH"

        # Long range — approach or projectile
        if random.random() < special_freq and my.energy >= 50:
            return random.choice(["STAND_D_DF_FA", "STAND_D_DF_FC"])
        return random.choice(["FORWARD_WALK", "DASH", "FOR_JUMP"])