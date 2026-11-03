"""Convert the compact RESULT_JSON lines captured from the Colab output into the run-file
format that analyze.py reads. Usage: python from_compact.py compact_dir out_dir"""
import json
import os
import sys

src, dst = sys.argv[1], sys.argv[2]
os.makedirs(dst, exist_ok=True)
for fn in sorted(os.listdir(src)):
    if not fn.endswith(".json"):
        continue
    c = json.load(open(os.path.join(src, fn)))
    run = {
        "method": c["method"], "order_id": c["order_id"], "order": c["order"],
        "hparams": c["hparams"], "versions": c["versions"], "total_seconds": c["total_seconds"],
        "baseline": {"summary": c["baseline"]},
        "edits": [{"step": i + 1, "success": e[0], "seconds": e[1], "immediate": [e[2], e[3]], "nan": bool(e[4])}
                  for i, e in enumerate(c["edits"])],
        "checkpoints_eval": {k: {"summary": v["summary"], "per_record": [{"main": m} for m in v["main"]]}
                             for k, v in c["cp"].items()},
    }
    json.dump(run, open(os.path.join(dst, f"{c['method']}_order{c['order_id']}.json"), "w"))
    print("converted", fn)
