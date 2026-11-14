"""Figures 1-3 of the manuscript. Usage: python plots.py RUNS_DIR OUT_DIR

RUNS_DIR must hold summary.json (Figures 1-2) and error_summary.json (Figure 3, written by error_analysis.py).
"""
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
DPI = 300
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
fig.savefig(os.path.join(OUT, "fig1_retention.png"), dpi=DPI)

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
fig.savefig(os.path.join(OUT, "fig2_para_neigh.png"), dpi=DPI)

# Figure 3: ROME retention by edit position (a) and categories of all failure cases (b)
EPATH = os.path.join(D, "error_summary.json")
if os.path.exists(EPATH):
    E = json.load(open(EPATH))
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.5), gridspec_kw={"width_ratios": [1, 1.25]})
    ax = axes[0]
    pb = E["subgroups"]["position_block"]
    names = list(pb.keys())
    rates = [pb[k]["rate"] for k in names]
    bars = ax.bar(range(len(names)), rates, color=COL["rome"], width=0.62)
    for i, k in enumerate(names):
        ax.annotate(f"{pb[k]['retained']}/{pb[k]['n']}", (i, rates[i]), xytext=(0, 3),
                    textcoords="offset points", ha="center", fontsize=8, color=INK)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names)
    ax.set_xlabel("Position of the edit in the sequence")
    ax.set_ylabel("ROME retention after 100 edits")
    ax.set_ylim(0, 1.1)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title("(a) ROME: retained edits by position", loc="left", color=INK, fontsize=10.5)
    ax = axes[1]
    cat = E["categories"]
    rows = [("ROME lost: displaced", cat.get("ROME retention failure | rome | R2 displaced", 0), COL["rome"]),
            ("ROME lost: reverted", cat.get("ROME retention failure | rome | R1 reverted", 0), COL["rome"]),
            ("ROME not inserted", sum(v for k, v in cat.items() if k.startswith("insertion failure | rome")), COL["rome"]),
            ("MEMIT not inserted", sum(v for k, v in cat.items() if k.startswith("insertion failure | memit")), COL["memit"]),
            ("MEMIT lost", 0, COL["memit"])]
    ys = list(range(len(rows)))[::-1]
    ax.barh(ys, [r[1] for r in rows], color=[r[2] for r in rows], height=0.6)
    for y, r in zip(ys, rows):
        ax.annotate(str(r[1]), (r[1], y), xytext=(4, 0), textcoords="offset points", va="center", fontsize=9, color=INK)
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlabel("Number of cases (three orders pooled)")
    ax.set_xlim(0, max(r[1] for r in rows) * 1.18)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title(f"(b) All {E['n_cases']} failure cases", loc="left", color=INK, fontsize=10.5)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig3_subgroups_errors.png"), dpi=DPI)
print("figures written to", OUT)
