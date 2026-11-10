"""Print the derived numbers used in Tables 1-2 of the report from results/summary.json.

Derived values: mean and sample SD over the three orders, Cohen's h, and the exact McNemar
p-value written in full (summary.json rounds it to six decimals).

Usage: python report_numbers.py ../results/summary.json
"""
import json
import math
import statistics
import sys
from math import comb

path = sys.argv[1] if len(sys.argv) > 1 else "../results/summary.json"
s = json.load(open(path))
runs, orders = s["runs"], s["orders"]
ORDERS = ["0", "1", "2"]
NAMES = {"0": "A (seed 1)", "1": "B (seed 2)", "2": "C (seed 3)"}


def msd(values, scale=100.0, digits=1):
    m = statistics.mean(values) * scale
    sd = statistics.stdev(values) * scale
    vals = ", ".join(f"{v * scale:.{digits + 1}f}" for v in values)
    return f"{m:.{digits}f} +- {sd:.{digits}f}   (orders A, B, C: {vals})"


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) * 0.5 ** n)


def cohens_h(p1, p2):
    return abs(2 * math.asin(math.sqrt(p1)) - 2 * math.asin(math.sqrt(p2)))


print("TABLE 1 (after 100 edits; mean +- sample SD over 3 orders)")
print("Unedited baseline (identical in every run):", runs["rome_order0"]["baseline"])
for method in ("rome", "memit"):
    r = [runs[f"{method}_order{o}"] for o in ORDERS]
    ret = [orders[o]["checkpoints"]["100"][f"retention_{method}"] for o in ORDERS]
    print(f"\n{method.upper()}")
    print("  Retention of shared earlier edits, % :", msd(ret))
    print("  Immediate success, %                 :", msd([x["immediate_success_rate"] for x in r]))
    for key, label in (("rewrite_rate", "Main prompts passing, %"),
                       ("para_rate", "Reworded prompts passing, %"),
                       ("neigh_rate", "Neighbourhood facts kept, %")):
        print(f"  {label:<37}:", msd([x["checkpoints"]["100"][key] for x in r]))
    print("  Seconds per edit                     :",
          msd([x["mean_seconds_per_edit"] for x in r], scale=1.0))
    print("  Minutes per run                      :",
          msd([x["total_minutes"] for x in r], scale=1.0))

print("\nTABLE 2 (retention after 100 edits, per order)")
diffs = []
for o in ORDERS:
    c = orders[o]["checkpoints"]["100"]
    pr, pm = c["retention_rome"], c["retention_memit"]
    b, m = c["discordant_rome_only"], c["discordant_memit_only"]
    lo, hi = c["bootstrap_95ci_difference"]
    diffs.append(pr - pm)
    print(f"Order {NAMES[o]}: shared facts {c['n_facts']}, ROME {pr * 100:.1f}%, MEMIT {pm * 100:.1f}%, "
          f"difference {(pr - pm) * 100:.1f} pp, 95% CI {lo * 100:.1f} to {hi * 100:.1f} pp, "
          f"discordant MEMIT only / ROME only {m} / {b}, exact McNemar p = {mcnemar_exact(b, m):.2e}, "
          f"Cohen's h = {cohens_h(pr, pm):.2f}")
print(f"Mean difference {statistics.mean(diffs) * 100:.1f} pp "
      f"(range {min(diffs) * 100:.1f} to {max(diffs) * 100:.1f}); Bonferroni level 0.05/3 = {0.05 / 3:.4f}")
