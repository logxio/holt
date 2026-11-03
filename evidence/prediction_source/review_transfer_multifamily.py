"""Stress test transfer where every pseudo-test family retains another PDB entry."""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

import crossfit_certify as cc
import crossfit_original
import crossfit_validity as cv
from certify_split import records, sequences
from crossfit_engine import ADVERSARIES, Pool, unit_hash
from review_transfer_original import EVENTS, RAW, evaluate


def split(pool, salt, pdb_counts):
    eligible_family = np.array([pdb_counts[name] > 1 for name in pool.family_name])
    pick = eligible_family & np.array([unit_hash(name, salt) < cv.HOLDOUT
                                       for name in pool.family_name])
    entry_hash = np.array([unit_hash(name, salt + "entry:") for name in pool.pdb_name])
    first = {}
    for i in np.flatnonzero(pick):
        f = pool.family[i]
        if f not in first or entry_hash[i] < entry_hash[first[f]]:
            first[f] = i
    held = np.zeros(pool.pdb.max() + 1, bool)
    held[pool.pdb[list(first.values())]] = True
    test = held[pool.pdb]
    if any(pdb_counts[pool.family_name[i]] <= 1 for i in np.flatnonzero(test)):
        raise AssertionError("pseudo-test family has no other PDB entry")
    return test, ~test


def run(pool, train_records, seq, b, pdb_counts, max_test_chains=None):
    salt = f"validity-multifamily:{b}:"
    test, train = split(pool, salt, pdb_counts)
    retained = [train_records[i] for i in np.flatnonzero(train)]
    naive = {task: cc.naive_prediction(retained, task)[2]
             for task in pool.tasks if task != "ec"}
    train_sequences = {seq[i] for i in np.flatnonzero(train)}
    design = cc.Representatives(pool, chains=train, tiers=("mmseqs30",))
    rules = cv.holt2(pool, design, "mmseqs30", naive)
    queries = np.flatnonzero(test)
    if max_test_chains is not None:
        queries = queries[:max_test_chains]
    scenarios = {}
    for name, key in (("family_excluded", pool.family), ("full_library", pool.pdb)):
        rows = pool.rows(queries, train, key, naive, ADVERSARIES["homology"])
        for row in rows:
            row["sequence"] = seq[row["index"]]
        scenarios[name] = {"train_chains": int(train.sum()), "test_chains": int(test.sum()),
                           "evaluated_chains": len(rows),
                           "tasks": evaluate(pool, rules, rows, train_sequences)}
    return {"replicate": b, "salt": salt, "scenarios": scenarios}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=("ec", "go"))
    parser.add_argument("--replicates", nargs=2, type=int, default=(0, 100))
    parser.add_argument("--max-test-chains", type=int)
    args = parser.parse_args()
    first, stop = args.replicates
    if not 0 <= first < stop:
        parser.error("expected 0 <= FIRST < STOP")
    crossfit_original.adapt(args.dataset)
    cv.EVENTS[f"orig-{args.dataset}"] = EVENTS[args.dataset]
    pool = Pool(f"orig-{args.dataset}")
    family_pdbs = defaultdict(set)
    for i, name in enumerate(pool.family_name):
        family_pdbs[name].add(pool.pdb_name[i])
    pdb_counts = {name: len(v) for name, v in family_pdbs.items()}
    data = crossfit_original.ROOT / args.dataset
    train_records = records(data / "train.jsonl")
    sequence_map = sequences(data / "train.fasta")
    seq = [sequence_map[name] for name in pool.ids]
    inputs = {str(path): crossfit_original.sha(path) for path in
              (data / "train.jsonl", data / "train.fasta",
               RAW / f"crossfit-orig-{args.dataset}-mmseqs30_cluster.tsv",
               RAW / "crossfit-protocol-v2.json")}
    suffix = f"-smoke{args.max_test_chains}" if args.max_test_chains else ""
    output = RAW / f"crossfit-{args.dataset}-original-multifamily{suffix}-{first}-{stop}.jsonl"
    if output.exists() and output.stat().st_size:
        raise FileExistsError(output)
    with output.open("w") as handle:
        for b in range(first, stop):
            started = time.monotonic()
            line = run(pool, train_records, seq, b, pdb_counts, args.max_test_chains)
            line["inputs"] = inputs
            handle.write(json.dumps(line, separators=(",", ":")) + "\n")
            handle.flush()
            print(f"replicate {b} {time.monotonic() - started:.1f}s {output}", flush=True)


if __name__ == "__main__":
    main()
