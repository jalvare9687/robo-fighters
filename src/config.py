"""
Shared constants for the FightingICE RL project.
All normalization values, action mappings, and hyperparameters live here.
"""

# ─── Game constants (from GameSetting.java / LaunchSetting.java) ─────────────
STAGE_WIDTH = 960
STAGE_HEIGHT = 640
MAX_HP = 400
MAX_ENERGY = 300
FPS = 60
ROUND_FRAMES = 3600          # 60 seconds per round
MAX_SPEED = 15               # rough max for normalization
MAX_REMAINING_FRAME = 70     # longest recovery animation

# ─── Commandable actions (subset of the 56‑entry Action enum) ────────────────
# These are the actions the agent can *choose*. Non‑commandable states like
# STAND_RECOV or DOWN are excluded — the game enters those automatically.
ACTION_NAMES = [
    "NEUTRAL",
    # Movement
    "FORWARD_WALK", "DASH", "BACK_STEP",
    "CROUCH", "JUMP", "FOR_JUMP", "BACK_JUMP",
    # Guards
    "STAND_GUARD", "CROUCH_GUARD", "AIR_GUARD",
    # Throws
    "THROW_A", "THROW_B",
    # Basic attacks
    "STAND_A", "STAND_B",
    "CROUCH_A", "CROUCH_B",
    "AIR_A", "AIR_B",
    "AIR_DA", "AIR_DB",
    # Forward attacks
    "STAND_FA", "STAND_FB",
    "CROUCH_FA", "CROUCH_FB",
    "AIR_FA", "AIR_FB",
    "AIR_UA", "AIR_UB",
    # Special moves (quarter‑circle / DP inputs)
    "STAND_D_DF_FA", "STAND_D_DF_FB",
    "STAND_F_D_DFA", "STAND_F_D_DFB",
    "STAND_D_DB_BA", "STAND_D_DB_BB",
    "AIR_D_DF_FA", "AIR_D_DF_FB",
    "AIR_F_D_DFA", "AIR_F_D_DFB",
    "AIR_D_DB_BA", "AIR_D_DB_BB",
    "STAND_D_DF_FC",
]
NUM_ACTIONS = len(ACTION_NAMES)   # 40

# ─── Observation space size ──────────────────────────────────────────────────
# 8 features per character × 2 + 3 relational = 19  (expandable later)
OBS_SIZE = 28

# ─── PPO hyperparameters (Stable‑Baselines3) ────────────────────────────────
PPO_PARAMS = dict(
    learning_rate=3e-4,
    n_steps=2048,         # frames buffered before each gradient update
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,        # encourage exploration early on
    verbose=1,
)

# ─── Training budget ────────────────────────────────────────────────────────
TOTAL_TIMESTEPS = 1_000_000     # ~2‑3 hours in fast mode; increase if time allows
CHECKPOINT_FREQ = 10_000      # save model every N steps

# ─── Director settings ──────────────────────────────────────────────────────
DIRECTOR_HISTORY = 20         # rolling window of rounds for performance stats
DIRECTOR_UP_THRESHOLD = 0.7   # win rate above this → increase difficulty
DIRECTOR_DOWN_THRESHOLD = 0.3 # win rate below this → decrease difficulty
DIRECTOR_STEP = 0.05          # how much to change difficulty per adjustment

# ─── Network ─────────────────────────────────────────────────────────────────
GAME_HOST = "127.0.0.1"
GAME_PORT = 31415