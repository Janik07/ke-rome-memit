"""Figures for the report. Usage: python plots.py RUNS_DIR OUT_DIR (RUNS_DIR also holds summary.json)."""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
S = json.load(open(os.path.join(D, "summary.json")))
COL = {"rome": "#2a78d6", "memit": "#eb6834"}
MK = {"rome": "o", "memit": "s"}
LBL = {"rome": "ROME", "memit": "MEMIT"}
CPS = [10, 25, 50, 100]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False})


def style(ax, ylabel):
    ax.set_xticks(CPS)
    ax.set_xlabel("Number of sequential edits")
    ax.set_ylabel(ylabel)
    ax.set_ylim(-0.03, 1.03)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def per_order(key):
    """values[method][order] -> list over checkpoints (None if missing)"""
    out = {"rome": {}, "memit": {}}
    for o, v in S["orders"].items():
        for m in ("rome", "memit"):
            out[m][o] = [v["checkpoints"].get(str(c), v["checkpoints"].get(c, {})).get(f"{key}_{m}") for c in CPS]
    return out


def plot_lines(ax, vals, title):
    for m in ("rome", "memit"):
        rows = [r for r in vals[m].values()]
        for r in rows:  # thin line per order
            ax.plot(CPS, r, color=COL[m], linewidth=1, alpha=0.35, marker=MK[m], markersize=4)
        mean = [sum(x[i] for x in rows) / len(rows) for i in range(len(CPS))]
        ax.plot(CPS, mean, color=COL[m], linewidth=2.2, marker=MK[m], markersize=8,
                markeredgecolor="white", markeredgewidth=1.5, label=f"{LBL[m]} (mean of 3 orders)")
        ax.annotate(f"{LBL[m]} {mean[-1]:.2f}", (CPS[-1], mean[-1]), xytext=(6, 0),
                    textcoords="offset points", va="center", color=INK, fontsize=9)
    ax.set_title(title, loc="left", color=INK, fontsize=11)
    ax.legend(frameon=False, loc="lower left")


# Figure 1: retention of shared, initially successful edits
fig, ax = plt.subplots(figsize=(6.6, 3.8))
plot_lines(ax, per_order("retention"), "Retention of earlier edits (thin lines: individual orders)")
style(ax, "Share of earlier edits still passing")
ax.set_xlim(5, 112)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig1_retention.png"), dpi=200)

# Figure 2: paraphrase and neighbourhood rates at each checkpoint, per method (mean over orders)
fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.4), sharey=True)
for ax, key, title in ((axes[0], "para_rate", "Reworded prompts (generalisation)"),
                       (axes[1], "neigh_rate", "Neighbourhood prompts (locality)")):
    for m in ("rome", "memit"):
        runs = [v for k, v in S["runs"].items() if k.startswith(m + "_")]
        if not runs:
            continue
        ys = [sum(r["checkpoints"][str(c)][key] for r in runs) / len(runs) for c in CPS]
        base = sum(r["baseline"][key] for r in runs) / len(runs)
        ax.plot(CPS, ys, color=COL[m], linewidth=2, marker=MK[m], markersize=7,
                markeredgecolor="white", markeredgewidth=1.2, label=LBL[m])
        if m == "rome":
            ax.axhline(base, color=MUTED, linewidth=1, linestyle="--")
            ax.annotate(f"unedited model {base:.2f}", (CPS[-1], base), xytext=(0, -12),
                        textcoords="offset points", color=MUTED, fontsize=8, ha="right")
    ax.set_title(title, loc="left", color=INK, fontsize=10.5)
    style(ax, "Pass rate" if key == "para_rate" else "")
axes[0].legend(frameon=False, loc="lower left")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig2_para_neigh.png"), dpi=200)
print("figures written to", OUT)
