"""
Analysis of the six runs (ROME/MEMIT x 3 orders).

Retention at checkpoint c (per order): among edits that succeeded immediately under BOTH
methods, made before checkpoint c (the edit made at step c itself is excluded), the share
that still passes the rewrite check (new target log-prob > true target log-prob).

Main test (per order, at 100 edits): paired difference ROME - MEMIT on the same facts,
exact McNemar test on discordant facts, 95% bootstrap CI (10,000 resamples of facts).

Usage: python analyze.py RESULTS_DIR  -> writes RESULTS_DIR/summary.json and prints a report.
"""
import json
import os
import random
import sys
from math import comb

D = sys.argv[1] if len(sys.argv) > 1 else "/content/ke_results"
METHODS = ["rome", "memit"]
ORDERS = [0, 1, 2]
CPS = [10, 25, 50, 100]
B = 10_000


def load(m, o):
    p = os.path.join(D, f"{m}_order{o}.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def mcnemar_exact(b, c):
    """Two-sided exact McNemar p-value: binomial test on discordant pairs (p=0.5)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def bootstrap_ci(x, y, seed=0):
    rng = random.Random(seed)
    n = len(x)
    if n == 0:
        return (None, None)
    diffs = []
    for _ in range(B):
        idx = [rng.randrange(n) for _ in range(n)]
        diffs.append(sum(x[i] - y[i] for i in idx) / n)
    diffs.sort()
    return (round(diffs[int(0.025 * B)], 4), round(diffs[int(0.975 * B) - 1], 4))


def rate(v):
    return round(sum(v) / len(v), 4) if v else None


def main():
    runs = {(m, o): load(m, o) for m in METHODS for o in ORDERS}
    out = {"runs": {}, "orders": {}, "overall": {}}

    # per-run descriptive results
    for (m, o), r in runs.items():
        if r is None:
            continue
        edits = r["edits"]
        secs = [e["seconds"] for e in edits]
        out["runs"][f"{m}_order{o}"] = {
            "n_edits_done": len(edits),
            "immediate_success_rate": rate([e["success"] for e in edits]),
            "immediate_success_by_quarter": [rate([e["success"] for e in edits[q * 25:(q + 1) * 25]])
                                             for q in range(4)],
            "nan_edits": sum(e.get("nan", False) for e in edits),
            "mean_seconds_per_edit": round(sum(secs) / len(secs), 2) if secs else None,
            "total_minutes": round(r.get("total_seconds", 0) / 60, 1),
            "baseline": r["baseline"]["summary"],
            "checkpoints": {c: v["summary"] for c, v in r["checkpoints_eval"].items()},
        }

    # paired comparison per order
    for o in ORDERS:
        R, M = runs[("rome", o)], runs[("memit", o)]
        if R is None or M is None:
            continue
        order = R["order"]
        assert order == M["order"], "orders must match"
        succ_r = [e["success"] for e in R["edits"]]
        succ_m = [e["success"] for e in M["edits"]]
        shared = [i for i in range(min(len(succ_r), len(succ_m))) if succ_r[i] and succ_m[i]]
        res_o = {"shared_successful_edits": len(shared), "checkpoints": {}}
        for c in CPS:
            sc = str(c)
            if sc not in R["checkpoints_eval"] or sc not in M["checkpoints_eval"]:
                continue
            pr = R["checkpoints_eval"][sc]["per_record"]
            pm = M["checkpoints_eval"][sc]["per_record"]
            elig = [i for i in shared if i < c - 1]          # exclude the edit made at step c
            xr = [int(pr[i]["main"][0] > pr[i]["main"][1]) for i in elig]
            xm = [int(pm[i]["main"][0] > pm[i]["main"][1]) for i in elig]
            # final success across ALL earlier requests (not only shared)
            allr = [int(pr[i]["main"][0] > pr[i]["main"][1]) for i in range(c - 1)]
            allm = [int(pm[i]["main"][0] > pm[i]["main"][1]) for i in range(c - 1)]
            b = sum(1 for a, d in zip(xr, xm) if a == 1 and d == 0)   # ROME kept, MEMIT lost
            cc = sum(1 for a, d in zip(xr, xm) if a == 0 and d == 1)  # MEMIT kept, ROME lost
            entry = {
                "n_facts": len(elig),
                "retention_rome": rate(xr), "retention_memit": rate(xm),
                "difference_rome_minus_memit": round(rate(xr) - rate(xm), 4) if elig else None,
                "discordant_rome_only": b, "discordant_memit_only": cc,
                "final_success_all_earlier_rome": rate(allr),
                "final_success_all_earlier_memit": rate(allm),
            }
            if c == 100:
                entry["mcnemar_exact_p"] = round(mcnemar_exact(b, cc), 6)
                entry["significant_at_0.05/3"] = entry["mcnemar_exact_p"] < 0.05 / 3
                entry["bootstrap_95ci_difference"] = bootstrap_ci(xr, xm, seed=o)
            res_o["checkpoints"][c] = entry
        out["orders"][o] = res_o

    diffs = [v["checkpoints"][100]["difference_rome_minus_memit"]
             for v in out["orders"].values() if 100 in v["checkpoints"]]
    if diffs:
        out["overall"] = {"orders_with_100": len(diffs),
                          "mean_difference_at_100": round(sum(diffs) / len(diffs), 4),
                          "min_difference": min(diffs), "max_difference": max(diffs),
                          "same_sign_all_orders": all(d > 0 for d in diffs) or all(d < 0 for d in diffs)
                          or all(d == 0 for d in diffs)}

    with open(os.path.join(D, "summary.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
