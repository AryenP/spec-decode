"""Turn results/*.jsonl into a markdown table. Usage: uv run python bench/report.py results/x.jsonl"""
import json, sys
from pathlib import Path

for path in sys.argv[1:]:
    rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    if not rows:
        continue
    r0 = rows[0]
    kv = next((r["tokens_per_s"] for r in rows if r["method"] == "kv_cache"), None)
    print(f"**{r0['target']}** ({r0['target_params_M']} M) target, **{r0['draft']}** ({r0['draft_params_M']} M) draft · "
          f"{r0.get('chip', r0.get('gpu', '?'))}, {r0.get('memory_gb','?')} GB · {r0['device']} {r0['dtype']} · "
          f"torch {r0['torch']} · prompt {r0['prompt_tokens']} tok, {r0['new_tokens']} new tok, T={r0['temperature']} · "
          f"median of {r0['repeats']} after {r0['warmup']} warmup\n")
    print("| method | γ | tok/s | vs KV | wall s | target fwds | draft fwds | accept rate | accepted/round | identical |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        g = r["gamma"] if r["gamma"] else "—"
        sp = f"{r['tokens_per_s']/kv:.2f}×" if kv else "—"
        acc = f"{r['acceptance_rate']:.2f}" if r["acceptance_rate"] is not None else "—"
        apr = f"{r['mean_accepted_per_round']:.2f}" if r["mean_accepted_per_round"] is not None else "—"
        ident = {True: "yes", False: "**NO**", None: "n/a"}[r["output_identical_to_kv_greedy"]]
        print(f"| {r['method']} | {g} | {r['tokens_per_s']:.1f} | {sp} | {r['wall_s_median']:.2f} | {r['target_forwards']} | {r['draft_forwards']} | {acc} | {apr} | {ident} |")
    print()
