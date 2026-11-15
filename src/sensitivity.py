"""Sensitivity of the main result to the definition of retention (Section 4.1 of the paper).

Three alternative definitions, all evaluated after 100 edits on the 99 earlier edits:
  1. Shared retention without the facts that the unedited model already passed
     (pre-edit log-prob of the new target above that of the true target).
  2. Intention-to-treat: every earlier edit counts, so insertion failures count as losses.
  3. Own-success retention: each method on the edits that it inserted itself.

The pre-edit log-probs come from results/baseline_main.json (the baseline of the
re-execution logs; the unedited model gives the same values in every run up to 1e-4).

Usage: python sensitivity.py ../results
"""
import json
import os
import statistics
import sys
from math import comb

D = sys.argv[1] if len(sys.argv) > 1 else "../results"
ORDERS = [0, 1, 2]
NAMES = {0: "A", 1: "B", 2: "C"}


def load(name):
    with open(os.path.join(D, name)) as f:
        return json.load(f)


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) * 0.5 ** n)


def passes(x):
    return x[0] > x[1]


def pct(v):
    return 100.0 * sum(v) / len(v) if v else float("nan")


base = {int(k): v for k, v in load("baseline_main.json")["main_logprobs_by_record"].items()}
pre_pass = {k for k, v in base.items() if passes(v)}
print(f"Facts that the unedited model already passes: {len(pre_pass)}")

res = {"excluding_pre_pass": {}, "intention_to_treat": {}, "own_success": {}}
for o in ORDERS:
    R, M = load(f"rome_order{o}.json"), load(f"memit_order{o}.json")
    assert R["order"] == M["order"]
    order = R["order"]
    sr = [e["success"] for e in R["edits"]]
    sm = [e["success"] for e in M["edits"]]
    pr = R["checkpoints_eval"]["100"]["per_record"]
    pm = M["checkpoints_eval"]["100"]["per_record"]
    earlier = range(99)  # the edit made at step 100 is excluded

    # 1. shared, without pre-edit passes
    elig = [i for i in earlier if sr[i] and sm[i] and order[i] not in pre_pass]
    xr = [int(passes(pr[i]["main"])) for i in elig]
    xm = [int(passes(pm[i]["main"])) for i in elig]
    b = sum(1 for a, d in zip(xr, xm) if a and not d)
    c = sum(1 for a, d in zip(xr, xm) if d and not a)
    res["excluding_pre_pass"][NAMES[o]] = {"n": len(elig), "rome": pct(xr), "memit": pct(xm),
                                           "discordant_memit_only": c, "discordant_rome_only": b,
                                           "mcnemar_p": mcnemar_exact(b, c)}
    # 2. intention-to-treat
    res["intention_to_treat"][NAMES[o]] = {
        "rome": pct([int(passes(pr[i]["main"])) for i in earlier]),
        "memit": pct([int(passes(pm[i]["main"])) for i in earlier])}
    # 3. own successful edits
    res["own_success"][NAMES[o]] = {
        "rome": pct([int(passes(pr[i]["main"])) for i in earlier if sr[i]]),
        "memit": pct([int(passes(pm[i]["main"])) for i in earlier if sm[i]])}

for key, title in (("excluding_pre_pass", "1. Shared retention without pre-edit passes"),
                   ("intention_to_treat", "2. Intention-to-treat (all 99 earlier edits)"),
                   ("own_success", "3. Each method's own successful edits")):
    print(f"\n{title}")
    for o, v in res[key].items():
        extra = (f"  n = {v['n']}, discordant MEMIT/ROME {v['discordant_memit_only']}/"
                 f"{v['discordant_rome_only']}, exact McNemar p = {v['mcnemar_p']:.1e}") if "n" in v else ""
        print(f"  Order {o}: ROME {v['rome']:.1f}%, MEMIT {v['memit']:.1f}%{extra}")
    for m in ("rome", "memit"):
        print(f"  Mean {m.upper()}: {statistics.mean(v[m] for v in res[key].values()):.1f}%")

with open(os.path.join(D, "sensitivity.json"), "w") as f:
    json.dump(res, f, indent=1)
