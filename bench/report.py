"""Turn results/*.jsonl into markdown tables. Usage: uv run python bench/report.py results/x.jsonl ..."""
import json, sys
from pathlib import Path


def hw(r):
    return f"{r.get('gpu') or r.get('chip', '?')}" + (f", {r['memory_gb']} GB" if 'memory_gb' in r and not r.get('gpu') else "")


def decode_table(rows):
    r0 = rows[0]
    kv = next((r["tokens_per_s"] for r in rows if r["method"] == "kv_cache"), None)
    print(f"**{r0['target']}** ({r0['target_params_M']} M) target, **{r0['draft']}** ({r0['draft_params_M']} M) draft · "
          f"{hw(r0)} · {r0['device']} {r0['dtype']} · torch {r0['torch']} · prompt {r0['prompt_tokens']} tok, "
          f"{r0['new_tokens']} new tok, T={r0['temperature']} · median of {r0['repeats']} after {r0['warmup']} warmup\n")
    print("| method | γ | tok/s | vs KV | wall s | target fwds | draft fwds | accept rate | accepted/round | identical |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        g = r["gamma"] if r["gamma"] else "—"
        sp = f"{r['tokens_per_s']/kv:.2f}×" if kv else "—"
        acc = f"{r['acceptance_rate']:.2f}" if r["acceptance_rate"] is not None else "—"
        apr = f"{r['mean_accepted_per_round']:.2f}" if r["mean_accepted_per_round"] is not None else "—"
        ident = {True: "yes", False: "**NO**", None: "n/a"}[r["output_identical_to_kv_greedy"]]
        print(f"| {r['method']} | {g} | {r['tokens_per_s']:.1f} | {sp} | {r['wall_s_median']:.2f} | {r['target_forwards']} | {r['draft_forwards']} | {acc} | {apr} | {ident} |")


def batch_table(rows):
    r0 = rows[0]
    print(f"**{r0['target']}** target, **{r0['draft']}** draft · {hw(r0)} · {r0['device']} {r0['dtype']} · "
          f"prompt {r0['prompt_tokens']} tok, {r0['new_tokens']} new tok per row, greedy, γ={next(r['gamma'] for r in rows if r['gamma'])}\n")
    print("| B | method | total tok/s | tok/s per row | vs KV at same B | target fwds | accept rate | identical |")
    print("|---|---|---|---|---|---|---|---|")
    kv_by_b = {r["batch"]: r["tokens_per_s"] for r in rows if r["method"] == "kv_cache"}
    for r in rows:
        sp = f"{r['tokens_per_s']/kv_by_b[r['batch']]:.2f}×" if r["batch"] in kv_by_b else "—"
        acc = f"{r['acceptance_rate']:.2f}" if r["acceptance_rate"] is not None else "—"
        ident = {True: "yes", False: "**NO**", None: "n/a"}[r["output_identical_to_kv_greedy"]]
        print(f"| {r['batch']} | {r['method']} | {r['tokens_per_s']:.1f} | {r['tokens_per_s_per_row']:.1f} | {sp} | {r['target_forwards']} | {acc} | {ident} |")


def scaling_table(rows):
    r0 = rows[0]
    print(f"**{r0['model']}** ({r0['params_M']} M) · {hw(r0)} · {r0['device']} {r0['dtype']} · one cached forward over k new positions "
          f"after a {r0['prefix_tokens']}-token prefix · median of {r0['repeats']}\n")
    print("| k positions | ms | vs k=1 | ms per position | implied GB/s if bandwidth-bound |")
    print("|---|---|---|---|---|")
    base = rows[0]["ms_median"]
    for r in rows:
        print(f"| {r['k_positions']} | {r['ms_median']:.2f} | {r['ms_median']/base:.2f}× | {r['ms_per_position']:.2f} | {r['implied_GBps_if_bandwidth_bound']:.0f} |")


for path in sys.argv[1:]:
    rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    if not rows:
        continue
    print(f"<!-- {path} -->")
    if "k_positions" in rows[0]:
        scaling_table(rows)
    elif "batch" in rows[0]:
        batch_table(rows)
    else:
        decode_table(rows)
    print()
