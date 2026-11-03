"""Pseudo-test transfer check on the frozen original DeepFRI training sets.

For each salted split, rebuild the HoLT-2 score and representative certificate
using only the retained training labels. PDB-held-out queries are evaluated
twice with the same family-tier rule: with their family excluded from retrieval,
and with relatives in the retained library. This isolates Assumption 2.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import crossfit_certify as cc
import crossfit_original
import crossfit_validity as cv
from certify_split import records, sequences
from crossfit_engine import ADVERSARIES, Pool


RAW = Path("results/raw")
ALPHA = 0.1
EVENTS = {"ec": ("ec_exact",), "go": ("bp_raw", "mf_raw", "cc_raw")}


def evaluate(pool, rules, rows, left_sequences):
    result = {}
    for name, rule in rules.items():
        task, event = name.split("_", 1)
        key = f"leak_{task}_{event}"
        eligible = [row for row in rows if pool.eligible[task][row["index"]]
                    and row["sequence"] not in left_sequences]
        choice = cv.select(rule, ALPHA)
        selected = []
        if choice is not None:
            cut = rule["trace"][choice]["cut"][0]
            for row in eligible:
                score = cc.copy_score(row, task, rule["model"])
                if score < cut:
                    selected.append({"id": row["id"], "family": row["family"],
                                     "pdb": row["pdb"], "leak": bool(row[key]),
                                     "labels": row[f"labels_{task}"], "score": score})
        result[name] = {"eligible": len(eligible), "calibration_n": rule["calibration_n"],
                        "choice_q": cc.Q[choice] if choice is not None else None,
                        "calibration": ([rule["trace"][choice]["n"],
                                         rule["trace"][choice]["leaks"]] if choice is not None else None),
                        "selected": selected}
    return result


def run_split(pool, train_records, seq, replicate, max_test_chains=None):
    salt = f"validity:{replicate}:"
    result = {"replicate": replicate, "salt": salt, "scenarios": {}}
    for tier in ("mmseqs30", "pdb"):
        start = time.monotonic()
        test, train = cv.held_out(pool, tier, salt)
        retained = [train_records[i] for i in np.flatnonzero(train)]
        naive = {task: cc.naive_prediction(retained, task)[2]
                 for task in pool.tasks if task != "ec"}
        train_sequences = {seq[i] for i in np.flatnonzero(train)}
        queries = np.flatnonzero(test)
        if max_test_chains is not None:
            queries = queries[:max_test_chains]
        design = cc.Representatives(pool, chains=train, tiers=(tier,))
        rules = cv.holt2(pool, design, tier, naive)
        key = pool.family if tier == "mmseqs30" else pool.pdb
        rows = pool.rows(queries, train, key, naive, ADVERSARIES["homology"])
        for row in rows:
            row["sequence"] = seq[row["index"]]
        result["scenarios"][tier] = {"train_chains": int(train.sum()),
                                    "test_chains": int(test.sum()),
                                    "evaluated_chains": len(rows),
                                    "tasks": evaluate(pool, rules, rows, train_sequences),
                                    "seconds": round(time.monotonic() - start, 2)}
        if tier == "pdb":
            # Hold the pseudo-test IDs fixed. Only the retrieval exclusion key
            # changes, so this is a paired test of the transfer assumption.
            start = time.monotonic()
            family_design = cc.Representatives(pool, chains=train, tiers=("mmseqs30",))
            family_rules = cv.holt2(pool, family_design, "mmseqs30", naive)
            excluded = pool.rows(queries, train, pool.family, naive, ADVERSARIES["homology"])
            for row in excluded:
                row["sequence"] = seq[row["index"]]
            result["scenarios"]["family_rule_pdb_holdout_family_excluded"] = {
                "train_chains": int(train.sum()), "test_chains": int(test.sum()),
                "evaluated_chains": len(excluded),
                "tasks": evaluate(pool, family_rules, excluded, train_sequences)}
            result["scenarios"]["family_rule_pdb_holdout_full_library"] = {
                "train_chains": int(train.sum()), "test_chains": int(test.sum()),
                "evaluated_chains": len(rows),
                "tasks": evaluate(pool, family_rules, rows, train_sequences),
                "seconds_pair": round(time.monotonic() - start, 2)}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=("ec", "go"))
    parser.add_argument("--replicates", nargs=2, type=int, default=(0, 20),
                        metavar=("FIRST", "STOP"))
    parser.add_argument("--max-test-chains", type=int)
    args = parser.parse_args()
    first, stop = args.replicates
    if not 0 <= first < stop:
        parser.error("expected 0 <= FIRST < STOP")
    crossfit_original.adapt(args.dataset)
    cv.EVENTS[f"orig-{args.dataset}"] = EVENTS[args.dataset]
    pool = Pool(f"orig-{args.dataset}")
    data = crossfit_original.ROOT / args.dataset
    train_records = records(data / "train.jsonl")
    sequence_map = sequences(data / "train.fasta")
    seq = [sequence_map[name] for name in pool.ids]
    inputs = {str(path): crossfit_original.sha(path) for path in
              (data / "train.jsonl", data / "train.fasta",
               RAW / f"crossfit-orig-{args.dataset}-mmseqs30_cluster.tsv",
               RAW / "crossfit-protocol-v2.json")}
    suffix = f"-smoke{args.max_test_chains}" if args.max_test_chains else ""
    output = RAW / f"crossfit-{args.dataset}-original-transfer{suffix}-{first}-{stop}.jsonl"
    if output.exists() and output.stat().st_size:
        raise FileExistsError(output)
    with output.open("w") as handle:
        for b in range(first, stop):
            started = time.monotonic()
            line = run_split(pool, train_records, seq, b, args.max_test_chains)
            line["inputs"] = inputs
            handle.write(json.dumps(line, separators=(",", ":")) + "\n")
            handle.flush()
            print(f"replicate {b} {time.monotonic() - started:.1f}s {output}", flush=True)


if __name__ == "__main__":
    main()
