"""
Sequential knowledge editing experiment: ROME vs MEMIT on GPT-2 XL (CounterFact).

One process = one run (method x order). The model is reloaded from the original
weights for every run. Edits are applied one at a time and accumulate.

Usage:
  python run_experiment.py prepare              # build the 100-record sample + 3 orders
  python run_experiment.py run rome 0           # method in {rome, memit}, order in {0,1,2}
  python run_experiment.py run rome 0 --n 5 --checkpoints 2,5 --tag smoke
"""
import argparse
import json
import os
import random
import sys
import time

import torch

EE_DIR = os.environ.get("EE_DIR", "/content/EasyEdit")
OUT_DIR = os.environ.get("KE_OUT", "/content/ke_results")
STATS_DIR = os.environ.get("KE_STATS", "/content/EasyEdit/data/stats")
CF_PATH = os.environ.get("KE_CF", "/content/counterfact.json")
SAMPLE_PATH = os.path.join(OUT_DIR, "sample.json")

SAMPLE_SEED = 42
ORDER_SEEDS = [1, 2, 3]
N_EDITS = 100
CHECKPOINTS = [10, 25, 50, 100]
N_PARA = 2      # reworded prompts per record (CounterFact provides 2)
N_NEIGH = 10    # neighbourhood prompts per record (CounterFact provides 10)


# ----------------------------------------------------------------------------- data
def prepare():
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(CF_PATH) as f:
        cf = json.load(f)
    print(f"CounterFact records loaded: {len(cf)}")

    eligible = []
    for r in cf:
        rw = r["requested_rewrite"]
        subj = rw["subject"].strip()
        if not subj or "{}" not in rw["prompt"]:
            continue
        if len(r.get("paraphrase_prompts", [])) < 1 or len(r.get("neighborhood_prompts", [])) < 1:
            continue
        if rw["target_new"]["str"].strip() == rw["target_true"]["str"].strip():
            continue
        eligible.append(r)
    print(f"Eligible after inclusion criteria: {len(eligible)}")

    rng = random.Random(SAMPLE_SEED)
    rng.shuffle(eligible)
    chosen, seen = [], set()
    for r in eligible:
        s = r["requested_rewrite"]["subject"].strip().lower()
        if s in seen:
            continue
        # avoid one subject being contained in another selected subject
        if any(s in t or t in s for t in seen):
            continue
        seen.add(s)
        chosen.append(r)
        if len(chosen) == N_EDITS:
            break

    sample = []
    for r in chosen:
        rw = r["requested_rewrite"]
        sample.append({
            "case_id": r["case_id"],
            "relation_id": rw.get("relation_id"),
            "prompt": rw["prompt"],
            "subject": rw["subject"],
            "target_new": rw["target_new"]["str"],
            "target_true": rw["target_true"]["str"],
            "paraphrase_prompts": r["paraphrase_prompts"][:N_PARA],
            "neighborhood_prompts": r["neighborhood_prompts"][:N_NEIGH],
        })
    orders = []
    for s in ORDER_SEEDS:
        idx = list(range(len(sample)))
        random.Random(s).shuffle(idx)
        orders.append(idx)

    with open(SAMPLE_PATH, "w") as f:
        json.dump({"sample_seed": SAMPLE_SEED, "order_seeds": ORDER_SEEDS,
                   "records": sample, "orders": orders}, f, indent=1)
    rels = {}
    for r in sample:
        rels[r["relation_id"]] = rels.get(r["relation_id"], 0) + 1
    print(f"Sample size: {len(sample)}; relation types: {len(rels)}")
    print("Relations:", json.dumps(dict(sorted(rels.items(), key=lambda x: -x[1]))))
    print(f"Saved {SAMPLE_PATH}")


# ----------------------------------------------------------------------------- scoring
@torch.no_grad()
def score_pairs(model, tok, pairs, bs=24):
    """Average token log-prob of each target given its prompt. pairs = [(prompt, target)]."""
    out = []
    dev = next(model.parameters()).device
    for i in range(0, len(pairs), bs):
        chunk = pairs[i:i + bs]
        seqs, spans = [], []
        for p, t in chunk:
            p_ids = tok(p).input_ids
            t_ids = tok(" " + t.strip()).input_ids
            seqs.append(p_ids + t_ids)
            spans.append((len(p_ids), len(p_ids) + len(t_ids)))
        L = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), L), tok.eos_token_id, dtype=torch.long)
        att = torch.zeros((len(seqs), L), dtype=torch.long)
        for j, s in enumerate(seqs):
            ids[j, :len(s)] = torch.tensor(s)
            att[j, :len(s)] = 1
        logits = model(input_ids=ids.to(dev), attention_mask=att.to(dev)).logits.float()
        lp = torch.log_softmax(logits, dim=-1)
        for j, (a, b) in enumerate(spans):
            tgt = ids[j, a:b].to(dev)
            vals = lp[j, a - 1:b - 1].gather(1, tgt.unsqueeze(1)).squeeze(1)
            out.append(float(vals.mean()))
    return out


def evaluate_records(model, tok, recs):
    """For each record: main prompt, reworded prompts and neighbourhood prompts,
    log-prob of new target and of true target."""
    pairs, index = [], []
    for k, r in enumerate(recs):
        main = r["prompt"].format(r["subject"])
        for kind, prompts in (("main", [main]), ("para", r["paraphrase_prompts"]),
                              ("neigh", r["neighborhood_prompts"])):
            for p in prompts:
                pairs.append((p, r["target_new"]))
                pairs.append((p, r["target_true"]))
                index.append((k, kind))
    vals = score_pairs(model, tok, pairs)
    res = [{"main": None, "para": [], "neigh": []} for _ in recs]
    for n, (k, kind) in enumerate(index):
        new, true = round(vals[2 * n], 4), round(vals[2 * n + 1], 4)
        if kind == "main":
            res[k]["main"] = [new, true]
        else:
            res[k][kind].append([new, true])
    return res


def summarise(res):
    """rewrite pass: new > true on main prompt; para: share new > true; neigh: share true > new."""
    rw = [int(x["main"][0] > x["main"][1]) for x in res]
    pa = [sum(n > t for n, t in x["para"]) / len(x["para"]) for x in res]
    ne = [sum(t > n for n, t in x["neigh"]) / len(x["neigh"]) for x in res]
    m = lambda v: round(sum(v) / len(v), 4) if v else None
    return {"rewrite_rate": m(rw), "para_rate": m(pa), "neigh_rate": m(ne), "n": len(res)}


# ----------------------------------------------------------------------------- run
def run(method, order_id, n_edits, checkpoints, tag):
    sys.path.insert(0, EE_DIR)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if method == "rome":
        from easyeditor import ROMEHyperParams as HP
        from easyeditor.models.rome import apply_rome_to_model as apply_fn
        hp_path = os.path.join(EE_DIR, "hparams/ROME/gpt2-xl.yaml")
    else:
        from easyeditor import MEMITHyperParams as HP
        from easyeditor.models.memit import apply_memit_to_model as apply_fn
        hp_path = os.path.join(EE_DIR, "hparams/MEMIT/gpt2-xl.yaml")

    hp = HP.from_hparams(hp_path)
    hp.model_name = "gpt2-xl"
    hp.stats_dir = STATS_DIR
    hp.device = 0
    if method == "rome":
        # match the original ROME configuration for GPT-2 XL (Meng et al., 2022)
        hp.mom2_adjustment = True

    with open(SAMPLE_PATH) as f:
        S = json.load(f)
    recs = S["records"]
    order = S["orders"][order_id][:n_edits]

    torch.manual_seed(0)
    random.seed(0)
    tok = AutoTokenizer.from_pretrained("gpt2-xl")
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained("gpt2-xl", torch_dtype=torch.float32).cuda().eval()

    name = f"{method}_order{order_id}" + (f"_{tag}" if tag else "")
    out_path = os.path.join(OUT_DIR, name + ".json")
    hp_dump = {k: (v if isinstance(v, (int, float, str, bool, list, type(None))) else str(v))
               for k, v in vars(hp).items()}
    log = {
        "method": method, "order_id": order_id, "order": order, "n_edits": n_edits,
        "checkpoints": checkpoints, "hparams": hp_dump,
        "versions": {"torch": torch.__version__,
                     "transformers": __import__("transformers").__version__,
                     "gpu": torch.cuda.get_device_name(0)},
        "baseline": None, "edits": [], "checkpoints_eval": {},
    }

    def save():
        with open(out_path, "w") as f:
            json.dump(log, f)

    t0 = time.time()
    base = evaluate_records(model, tok, [recs[i] for i in order])
    log["baseline"] = {"summary": summarise(base), "per_record": base}
    print(f"[baseline] {log['baseline']['summary']}  ({time.time() - t0:.0f}s)", flush=True)
    save()

    for step, rid in enumerate(order, start=1):
        r = recs[rid]
        req = {"prompt": r["prompt"], "subject": r["subject"],
               "target_new": r["target_new"], "ground_truth": r["target_true"]}
        te = time.time()
        apply_fn(model, tok, [req], hp, copy=False, return_orig_weights=False)
        dt = time.time() - te
        imm = evaluate_records(model, tok, [r])[0]
        ok = int(imm["main"][0] > imm["main"][1])
        nan = not all(map(lambda v: v == v, imm["main"]))
        log["edits"].append({"step": step, "record": rid, "case_id": r["case_id"],
                             "seconds": round(dt, 2), "immediate": imm["main"], "success": ok,
                             "nan": nan})
        print(f"[{name}] edit {step}/{n_edits} case {r['case_id']} success={ok} "
              f"new={imm['main'][0]:.2f} true={imm['main'][1]:.2f} {dt:.1f}s", flush=True)

        if step in checkpoints:
            tc = time.time()
            done = [recs[i] for i in order[:step]]
            ev = evaluate_records(model, tok, done)
            log["checkpoints_eval"][str(step)] = {"summary": summarise(ev), "per_record": ev}
            print(f"[{name}] CHECKPOINT {step}: {summarise(ev)} ({time.time() - tc:.0f}s)", flush=True)
            save()
    log["total_seconds"] = round(time.time() - t0, 1)
    save()
    print(f"[{name}] DONE in {log['total_seconds']/60:.1f} min -> {out_path}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prepare", "run"])
    ap.add_argument("method", nargs="?", choices=["rome", "memit"])
    ap.add_argument("order", nargs="?", type=int, default=0)
    ap.add_argument("--n", type=int, default=N_EDITS)
    ap.add_argument("--checkpoints", default=",".join(map(str, CHECKPOINTS)))
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    if a.cmd == "prepare":
        prepare()
    else:
        run(a.method, a.order, a.n, [int(x) for x in a.checkpoints.split(",")], a.tag)
