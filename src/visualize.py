"""
visualize_results.py — Generate all charts for the demo presentation.

Saves PNGs to results/ folder.
Run: python visualize_results.py
"""

import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['font.size'] = 13
matplotlib.rcParams['figure.dpi'] = 150
os.makedirs("results", exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════
# 1. Generalization test — side-by-side bar chart
# ═══════════════════════════════════════════════════════════════════════

configs =       ["passive", "balanced", "aggressive", "combo_heavy", "special_heavy"]
exp_win_rate =  [1.00,      0.67,       0.03,         0.93,          0.00]
ctrl_win_rate = [1.00,      0.93,       0.23,         0.90,          0.47]

x = np.arange(len(configs))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 5))
bars1 = ax.bar(x - width/2, exp_win_rate, width, label="Director-trained", color="#534AB7")
bars2 = ax.bar(x + width/2, ctrl_win_rate, width, label="Control (static)", color="#1D9E75")

ax.set_ylabel("Win rate")
ax.set_title("Generalization test: win rate across unseen opponent configs")
ax.set_xticks(x)
ax.set_xticklabels(configs, rotation=15)
ax.set_ylim(0, 1.15)
ax.legend()
ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.4)

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
            f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=10)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
            f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig("results/generalization_test.png")
plt.close()
print("Saved: results/generalization_test.png")


# ═══════════════════════════════════════════════════════════════════════
# 2. Generalization — average HP remaining
# ═══════════════════════════════════════════════════════════════════════

exp_avg_hp =  [370.3, 140.8,   9.8, 197.0,   0.0]
ctrl_avg_hp = [374.2, 171.0,  15.7, 199.2,  34.6]

fig, ax = plt.subplots(figsize=(10, 5))
bars1 = ax.bar(x - width/2, exp_avg_hp, width, label="Director-trained", color="#534AB7")
bars2 = ax.bar(x + width/2, ctrl_avg_hp, width, label="Control (static)", color="#1D9E75")

ax.set_ylabel("Average HP remaining")
ax.set_title("Generalization test: average HP remaining across unseen configs")
ax.set_xticks(x)
ax.set_xticklabels(configs, rotation=15)
ax.legend()

plt.tight_layout()
plt.savefig("results/generalization_hp.png")
plt.close()
print("Saved: results/generalization_hp.png")


# ═══════════════════════════════════════════════════════════════════════
# 3. Head-to-head results
# ═══════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# Test 1: Director as P1
labels = ["Director", "Control"]
test1 = [18, 12]
colors = ["#534AB7", "#1D9E75"]
axes[0].bar(labels, test1, color=colors)
axes[0].set_title("Director as P1")
axes[0].set_ylabel("Wins (out of 30)")
axes[0].set_ylim(0, 32)
for i, v in enumerate(test1):
    axes[0].text(i, v + 0.5, str(v), ha='center', fontweight='bold')

# Test 2: Director as P2
test2 = [24, 6]
axes[1].bar(labels, test2, color=colors)
axes[1].set_title("Director as P2")
axes[1].set_ylabel("Wins (out of 30)")
axes[1].set_ylim(0, 32)
for i, v in enumerate(test2):
    axes[1].text(i, v + 0.5, str(v), ha='center', fontweight='bold')

# Combined
combined = [42, 18]
axes[2].bar(labels, combined, color=colors)
axes[2].set_title("Combined (60 rounds)")
axes[2].set_ylabel("Wins (out of 60)")
axes[2].set_ylim(0, 50)
for i, v in enumerate(combined):
    pct = v / 60 * 100
    axes[2].text(i, v + 0.5, f'{v} ({pct:.0f}%)', ha='center', fontweight='bold')

fig.suptitle("Head-to-head: Director-trained vs Control agent", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("results/head_to_head.png")
plt.close()
print("Saved: results/head_to_head.png")


# ═══════════════════════════════════════════════════════════════════════
# 4. Director difficulty curve over training
# ═══════════════════════════════════════════════════════════════════════

director_log = "checkpoints/experimental/director_log.csv"
if os.path.exists(director_log):
    rounds, difficulties, win_flags, student_hps = [], [], [], []

    with open(director_log, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rounds.append(int(row['round']))
            difficulties.append(float(row['difficulty_level']))
            win_flags.append(row['student_won'] == 'True')
            student_hps.append(int(row['student_hp']))

    # Rolling win rate (window of 20)
    window = 20
    rolling_wr = []
    for i in range(len(win_flags)):
        start = max(0, i - window + 1)
        rolling_wr.append(sum(win_flags[start:i+1]) / (i - start + 1))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    # Top: difficulty level
    ax1.plot(rounds, difficulties, color="#534AB7", linewidth=1.5)
    ax1.set_ylabel("Difficulty level")
    ax1.set_title("AI Director: difficulty progression during training")
    ax1.set_ylim(-0.05, 1.1)
    ax1.axhline(y=1.0, color='red', linestyle='--', alpha=0.3, label='Max difficulty')
    ax1.legend(loc='lower right')
    ax1.fill_between(rounds, difficulties, alpha=0.15, color="#534AB7")

    # Bottom: rolling win rate
    ax2.plot(rounds, rolling_wr, color="#1D9E75", linewidth=1.5)
    ax2.set_ylabel("Rolling win rate (20-round window)")
    ax2.set_xlabel("Round")
    ax2.set_ylim(-0.05, 1.1)
    ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.4, label='50% baseline')
    ax2.legend(loc='upper right')
    ax2.fill_between(rounds, rolling_wr, alpha=0.15, color="#1D9E75")

    plt.tight_layout()
    plt.savefig("results/director_curriculum.png")
    plt.close()
    print("Saved: results/director_curriculum.png")
else:
    print(f"Director log not found at {director_log} — skipping curriculum chart")


# ═══════════════════════════════════════════════════════════════════════
# 5. Summary stats printout
# ═══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("  RESULTS SUMMARY")
print("=" * 60)
print(f"\n  Generalization (avg win rate across 5 configs):")
print(f"    Director-trained:  {np.mean(exp_win_rate):.2f}")
print(f"    Control (static):  {np.mean(ctrl_win_rate):.2f}")
print(f"\n  Head-to-head (60 rounds, side-balanced):")
print(f"    Director-trained:  42/60 (70%)")
print(f"    Control (static):  18/60 (30%)")
print(f"\n  Charts saved to results/")
print("=" * 60)