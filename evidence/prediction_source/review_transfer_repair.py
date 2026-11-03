"""Training-only transfer repairs, with explicit abstention and finite-set diagnostics.

No benchmark validation/test labels are opened. Candidate guards are applied to
design, calibration AND deployment, before fitting/quantile construction. The
alpha margin is a sensitivity analysis, not an estimated guarantee. Random
priority still uses the conflict-free independent set and is not IID sampling.
"""
from __future__ import annotations

import argparse
import json
import resource
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import binom

import crossfit_certify as cc
import crossfit_original as original
import crossfit_validity as cv
from certify_split import records, sequences, wilson
from crossfit_engine import ADVERSARIES, Pool
from review_transfer_multifamily import split as stress_split
from review_transfer_original import EVENTS

RAW = Path("results/raw")
ALPHAS = (.05, .075, .1)


def guard(row, mode):
    # The identity proxy is deliberately broader than identity30 + coverage80.
    return mode == "none" or row["identity"] < .3


def fit(design, calibration, task, event, mode):
    d = [r.copy() for r in design if guard(r, mode)]
    c = [r.copy() for r in calibration if guard(r, mode)]
    key = f"leak_{task}_{event}"
    if not d or len({bool(r[key]) for r in d}) < 2:
        return {"unavailable": "guarded design lacks two outcome classes",
                "design_n": len(d), "calibration_n": len(c)}
    rule = cv.frozen(d, c, task, event, cc.PER_TEST["pooled"], True)
    rule.update(design_n=len(d), calibration_n=len(c), guard=mode)
    return rule


def evaluate(pool, rule, rows, seqs, task, event, alpha):
    if "unavailable" in rule:
        return {"selected": [], "reason": rule["unavailable"], "choice_q": None}
    idx = cv.select(rule, alpha)
    chosen = []
    if idx is not None:
        cut = rule["trace"][idx]["cut"][0]
        for row in rows:
            if (not pool.eligible[task][row["index"]] or row["sequence"] in seqs
                    or not guard(row, rule["guard"])):
                continue
            score = cc.copy_score(row, task, rule["model"])
            if score < cut:
                chosen.append({"id": row["id"], "family": row["family"],
                               "pdb": row["pdb"], "leak": bool(row[f"leak_{task}_{event}"]),
                               "identity": row["identity"], "score": score})
    return {"selected": chosen, "choice_q": cc.Q[idx] if idx is not None else None,
            "calibration": rule["trace"][idx] if idx is not None else None}


def run(pool, training, seq, b, args):
    family_pdbs = defaultdict(set)
    for f, p in zip(pool.family_name, pool.pdb_name):
        family_pdbs[f].add(p)
    counts = {f: len(p) for f, p in family_pdbs.items()}
    out = {"replicate": b, "scenarios": {}}
    for scenario in args.scenarios:
        stress = scenario == "stress"
        salt = f"validity-multifamily:{b}:" if stress else f"validity:{b}:"
        tier = "mmseqs30" if stress else scenario
        test, train = stress_split(pool, salt, counts) if stress else cv.held_out(pool, tier, salt)
        retained = [training[i] for i in np.flatnonzero(train)]
        naive = {t: cc.naive_prediction(retained, t)[2] for t in pool.tasks if t != "ec"}
        queries = np.flatnonzero(test)
        if args.max_test_chains:
            queries = queries[:args.max_test_chains]
        train_seq = {seq[i] for i in np.flatnonzero(train)}
        # Full deployment library in the stress scenario; family is retained.
        key = pool.family if scenario == "mmseqs30" else pool.pdb
        rows = pool.rows(queries, train, key, naive, ADVERSARIES["homology"])
        for row in rows:
            row["sequence"] = seq[row["index"]]
        entry = {"salt": salt, "train_chains": int(train.sum()),
                 "test_chains": int(test.sum()), "evaluated_chains": len(rows), "variants": {}}
        for priority in args.priorities:
            reps = cc.Representatives(pool, priority=priority, chains=train, tiers=(tier,))
            design, calibration, _ = reps.rows(tier, naive, ADVERSARIES["homology"])
            for mode in args.guards:
                variant = entry["variants"][f"{priority}/{mode}"] = {}
                for name in EVENTS[args.dataset]:
                    task, event = name.split("_", 1)
                    rule = fit(design[task], calibration[task], task, event, mode)
                    variant[name] = {"rule": rule, "levels": {
                        str(a): evaluate(pool, rule, rows, train_seq, task, event, a)
                        for a in ALPHAS}}
        out["scenarios"][scenario] = entry
    return out


def summarize(path):
    cells = defaultdict(list)
    for line in path.open():
        result = json.loads(line)
        for scenario, entry in result["scenarios"].items():
            for variant, tasks in entry["variants"].items():
                for task, payload in tasks.items():
                    for alpha, cell in payload["levels"].items():
                        cells[(scenario, variant, task, alpha)].append(cell)
    summary = []
    for (scenario, variant, task, alpha), values in sorted(cells.items()):
        rows = [r for c in values for r in c["selected"]]
        n, k = len(rows), sum(r["leak"] for r in rows)
        nonempty = [v for v in values if v["selected"]]
        above = sum(sum(r["leak"] for r in v["selected"]) > .1 * len(v["selected"]) for v in nonempty)
        # Diagnostic only: chains and repetitions share families and are not IID.
        binomial_rejections = sum(bool(binom.sf(sum(r["leak"] for r in v["selected"]) - 1,
                                              len(v["selected"]), .1) < .025) for v in nonempty)
        summary.append({"scenario": scenario, "variant": variant, "task": task,
                        "calibration_alpha": float(alpha), "target_alpha": .1,
                        "replicates": len(values), "nonempty": len(nonempty),
                        "n": n, "leaks": k, "rate": k / n if n else None,
                        "unique_chains": len({r["id"] for r in rows}),
                        "families": len({r["family"] for r in rows}),
                        "above_alpha": above, "above_all": above / len(values),
                        "above_nonempty": above / len(nonempty) if nonempty else None,
                        "above_all_wilson_descriptive": wilson(above, len(values)),
                        "binomial_rejections_descriptive": binomial_rejections,
                        "status": "no evidence (empty)" if not n else "pooled within alpha" if k/n <= .1 else "pooled exceeds alpha"})
    output = path.with_suffix(".summary.json")
    output.write_text(json.dumps({"source": str(path), "source_sha256": original.sha(path),
                                 "note": "Repeated training splits are dependent; realized exceedance is not delta error.",
                                 "cells": summary}, indent=2) + "\n")
    return output


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dataset", choices=("ec", "go"))
    p.add_argument("--replicates", type=int, nargs=2, default=(0, 10))
    p.add_argument("--scenarios", nargs="+", choices=("mmseqs30", "pdb", "stress"), default=("mmseqs30", "pdb", "stress"))
    p.add_argument("--priorities", nargs="+", choices=("degree", "random"), default=("degree", "random"))
    p.add_argument("--guards", nargs="+", choices=("none", "identity30"), default=("none", "identity30"))
    p.add_argument("--max-test-chains", type=int)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--summary-only", action="store_true")
    args = p.parse_args()
    if args.summary_only:
        print(summarize(args.output)); return
    if args.output.exists():
        raise FileExistsError(args.output)
    original.adapt(args.dataset)
    pool = Pool(f"orig-{args.dataset}")
    root = original.ROOT / args.dataset
    training = records(root / "train.jsonl")
    seqmap = sequences(root / "train.fasta")
    seq = [seqmap[name] for name in pool.ids]
    inputs = {str(path): original.sha(path) for path in
              (Path(__file__), root / "train.jsonl", root / "train.fasta",
               RAW / f"crossfit-orig-{args.dataset}-mmseqs30_cluster.tsv", RAW / "crossfit-protocol-v2.json")}
    with args.output.open("x") as handle:
        for b in range(*args.replicates):
            started = time.monotonic()
            result = run(pool, training, seq, b, args)
            result.update(inputs=inputs, seconds=round(time.monotonic() - started, 2),
                          peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            handle.write(json.dumps(result, separators=(",", ":")) + "\n"); handle.flush()
            print(f"replicate {b} {result['seconds']}s peak {result['peak_rss_bytes']/1e9:.3f} GB", flush=True)
    print(summarize(args.output), flush=True)


if __name__ == "__main__":
    main()
