# AI Director — FightingICE Research Project

## What this is
A research system studying whether an adaptive AI Director produces a more
sample-efficient RL fighter agent than standard training.

Two conditions:
- **Control**:      Student trains against a static opponent
- **Experimental**: Student trains against a Director-tuned adaptive opponent

## File map
```
python/
├── environment.py     ← Gymnasium wrapper (FrameData → obs, action → Key)
├── director.py        ← AI Director logic + OpponentConfig dataclass
├── opponent.py        ← Rule-based opponent controlled by Director
├── train.py           ← Training entry point (control or experimental)
├── evaluate.py        ← Generalization test against unseen configs
├── plot_results.py    ← Generates paper figures
└── requirements.txt   ← pip dependencies
```

## Setup

### 1. Java — FightingICE
- Requires Java 21
- Check: `java -version`
- Download OpenJDK 21 if needed: https://adoptium.net

### 2. Python dependencies
```bash
pip install -r requirements.txt
```

### 3. Verify FightingICE runs
From your FightingICE directory:
```
Windows: run-windows-amd64.bat
Mac:     ./run-macos-arm64.sh
Linux:   ./run-linux-amd64.sh
```
Add `--fastmode` for training (disables rendering, ~10x faster).

## Running

### Train control agent (no Director)
```bash
python train.py --mode control --episodes 300
```

### Train experimental agent (Director active)
```bash
python train.py --mode experimental --episodes 300
```

### Train both sequentially
```bash
python train.py --mode both --episodes 300
```

### Evaluate generalization
```bash
python evaluate.py
```

### Generate paper figures
```bash
python plot_results.py
```
Figures saved to `figures/`.

## Output files
```
models/
├── student_control.zip        ← saved PPO model (control)
└── student_experimental.zip   ← saved PPO model (experimental)

logs/
├── director_control.csv       ← per-round Director log (enabled=False)
├── director_experimental.csv  ← per-round Director log (enabled=True)
├── sb3_control/               ← SB3 training logs
├── sb3_experimental/          ← SB3 training logs
└── evaluation_results.csv     ← generalization test results

figures/
├── fig1_learning_curves.png   ← win rate over rounds, both conditions
├── fig2_generalization.png    ← bar chart: unseen config win rates
└── fig3_director_difficulty.png ← Director difficulty escalation over time
```

## Research question
> "Does a student agent trained against an adaptive AI Director generalize
> better to unseen opponent configurations than one trained against a static
> opponent, and does the Director produce faster convergence?"

## Key parameters to tune
In `director.py`:
- `UPPER_THRESHOLD` (default 0.65) — win rate that triggers difficulty increase
- `LOWER_THRESHOLD` (default 0.35) — win rate that triggers difficulty decrease
- `WINDOW_SIZE` (default 15)       — rolling window for win rate calculation
- `STEP_SIZE` (default 0.1)        — how aggressively Director changes difficulty

In `environment.py`:
- Reward weights `W_DAMAGE_DEALT`, `W_DAMAGE_TAKEN`, `W_WIN_BONUS`

In `train.py`:
- PPO hyperparameters: `learning_rate`, `n_steps`, `batch_size`
