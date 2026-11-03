"""
Error analysis and subgroup analysis (Assignment 4, Part 4).

Failure cases analysed:
  (a) ROME retention failures: edits that worked immediately under BOTH methods but fail the
      rewrite check after 100 edits under ROME (MEMIT had no retention failures).
  (b) Immediate insertion failures under either method.

Each case is assigned to ONE category by fixed rules applied in this order (defined before
looking at individual cases; examples are then read manually):
  ROME retention failures
    R1 reverted      - the true answer is back within 2 nats of its pre-edit log-prob
    R2 displaced     - both answers are now unlikely (new < -6 and true < pre-edit true - 2):
                       the model predicts something else
    R3 near tie      - new answer still within 1 nat of the true answer
    R4 other loss
  Insertion failures
    I1 near miss         - new answer within 1 nat of the true answer right after the edit
    I2 strong prior      - unedited model was confident in the true answer (log-prob > -1.5)
    I3 multi-token target- the new answer has 2+ GPT-2 tokens
    I4 other

Usage: python error_analysis.py RESULTS_DIR   (needs sample.json and the six run files;
       pre-edit log-probs are recomputed with the unedited model if the run files lack them)
"""
import csv
import json
import os
import sys

D = sys.argv[1] if len(sys.argv) > 1 else "/content/ke_results"
S = json.load(open(os.path.join(D, "sample.json")))
REC = S["records"]
runs = {}
for m in ("rome", "memit"):
    for o in (0, 1, 2):
        runs[(m, o)] = json.load(open(os.path.join(D, f"{m}_order{o}.json")))

from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("gpt2-xl")
ntok = {i: len(tok(" " + r["target_new"].strip()).input_ids) for i, r in enumerate(REC)}

# pre-edit (unedited model) log-probs for every record's main prompt
base = {}
full = [r for r in runs.values() if r["baseline"].get("per_record")]
if full:
    for pos, rid in enumerate(full[0]["order"]):
        base[rid] = full[0]["baseline"]["per_record"][pos]["main"]
else:
    import torch
    from run_experiment import evaluate_records
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained("gpt2-xl", dtype=torch.float32).cuda().eval()
    ev = evaluate_records(model, tok, REC)
    base = {i: ev[i]["main"] for i in range(len(REC))}
    del model


def passes(x):
    return x[0] > x[1]


rows = []
for o in (0, 1, 2):
    R, M = runs[("rome", o)], runs[("memit", o)]
    order = R["order"]
    sr = [e["success"] for e in R["edits"]]
    sm = [e["success"] for e in M["edits"]]
    pr100 = R["checkpoints_eval"]["100"]["per_record"]
    for pos, rid in enumerate(order):
        r = REC[rid]
        later_same_rel = sum(1 for p2 in range(pos + 1, len(order)) if REC[order[p2]]["relation_id"] == r["relation_id"])
        later_same_target = sum(1 for p2 in range(pos + 1, len(order)) if REC[order[p2]]["target_new"] == r["target_new"])
        common = dict(order=o, position=pos + 1, record=rid, case_id=r["case_id"], relation=r["relation_id"],
                      prompt=r["prompt"].format(r["subject"]), true=r["target_true"], new=r["target_new"],
                      new_tokens=ntok[rid], pre_new=base[rid][0], pre_true=base[rid][1],
                      later_same_relation=later_same_rel, later_same_target=later_same_target)
        # (a) ROME retention failures (shared successes, excluding the 100th edit)
        if pos < 99 and sr[pos] and sm[pos] and not passes(pr100[pos]["main"]):
            n, t = pr100[pos]["main"]
            if t >= base[rid][1] - 2:
                cat = "R1 reverted"
            elif n < -6 and t < base[rid][1] - 2:
                cat = "R2 displaced"
            elif n > t - 1:
                cat = "R3 near tie"
            else:
                cat = "R4 other loss"
            rows.append(dict(common, kind="ROME retention failure", method="rome", cat=cat, after_new=n, after_true=t))
        # (b) immediate insertion failures
        for m, s, run in (("rome", sr, R), ("memit", sm, M)):
            if not s[pos]:
                n, t = run["edits"][pos]["immediate"]
                if n > t - 1:
                    cat = "I1 near miss"
                elif base[rid][1] > -1.5:
                    cat = "I2 strong prior"
                elif ntok[rid] >= 2:
                    cat = "I3 multi-token target"
                else:
                    cat = "I4 other"
                rows.append(dict(common, kind="insertion failure", method=m, cat=cat, after_new=n, after_true=t))

with open(os.path.join(D, "error_cases.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

out = {"n_cases": len(rows), "categories": {}, "examples": {}, "subgroups": {}}
for row in rows:
    key = f"{row['kind']} | {row['method']} | {row['cat']}"
    out["categories"][key] = out["categories"].get(key, 0) + 1
    out["examples"].setdefault(key, [])
    if len(out["examples"][key]) < 2:
        out["examples"][key].append({k: row[k] for k in ("order", "position", "case_id", "relation", "prompt", "true", "new",
                                                         "pre_true", "after_new", "after_true", "new_tokens")})

# subgroup / sensitivity: ROME retention at 100 by edit position, target length and prior confidence
ret = []
for o in (0, 1, 2):
    R, M = runs[("rome", o)], runs[("memit", o)]
    sr = [e["success"] for e in R["edits"]]
    sm = [e["success"] for e in M["edits"]]
    pr100 = R["checkpoints_eval"]["100"]["per_record"]
    for pos, rid in enumerate(R["order"][:99]):
        if sr[pos] and sm[pos]:
            ret.append((pos + 1, ntok[rid], base[rid][1], REC[rid]["relation_id"], int(passes(pr100[pos]["main"]))))


def grp(name, f):
    d = {}
    for x in ret:
        k = f(x)
        d.setdefault(k, []).append(x[4])
    out["subgroups"][name] = {str(k): {"n": len(v), "retained": sum(v), "rate": round(sum(v) / len(v), 4)} for k, v in sorted(d.items())}


grp("position_block", lambda x: f"{(x[0] - 1) // 20 * 20 + 1}-{(x[0] - 1) // 20 * 20 + 20}")
grp("new_target_tokens", lambda x: "1 token" if x[1] == 1 else "2+ tokens")
grp("pre_edit_true_logprob", lambda x: "confident (> -1.5)" if x[2] > -1.5 else "less confident (<= -1.5)")
rel_counts = {}
for x in ret:
    rel_counts.setdefault(x[3], []).append(x[4])
out["subgroups"]["relation_min10"] = {k: {"n": len(v), "rate": round(sum(v) / len(v), 4)} for k, v in rel_counts.items() if len(v) >= 10}

with open(os.path.join(D, "error_summary.json"), "w") as f:
    json.dump(out, f, indent=1)
print("ERROR_SUMMARY " + json.dumps(out))
