"""Does the certificate hold when calibration chains are dependent? Pseudo-test splits inside training.

Each replicate b holds out training chains as pseudo-test chains, runs a HoLT design on the remaining training
chains alone, freezes its rules, and measures how often the homology adversary answers the pseudo-test chains each
rule certifies, retrieving the remaining training chains as a test chain retrieves the training set.

* tier ``mmseqs30``: 20% of families (salted hash) are held out whole, as for a query whose family is new;
* tier ``pdb``: one PDB entry (the first by salted hash) of 20% of families is held out, the rest of each family
  stays, as for a query whose relatives are in training; held-out chains with an identical sequence left in
  training never enter a certified set, as in crossfit_accept.py.

Designs: HoLT-1 (every chain a calibration query against all other chains, chains as units, design by PDB hash,
Bonferroni over 7 events x 12 candidates); HoLT-2 representatives (primary); HoLT-2 split. Each uses its own test.
Pseudo-test chains are training chains, so this checks dependence among calibration chains and library size, not
the shift from training to the benchmark's test set, which crossfit_accept.py measures on the real test set.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.stats import binom

import crossfit_certify as cc
from certify_split import cp_upper, records, sequences
from crossfit_engine import ADVERSARIES, Pool, unit_hash
from crossfit_retrieve import DATA


RAW = Path("results/raw")
ALPHAS = (.05, .1, .15)
HOLDOUT = .2
EVENTS = {"ec": ("ec_exact",), "go": ("bp_raw", "mf_raw", "cc_raw")}
ATTACK = ADVERSARIES["homology"]


def held_out(pool, tier, salt):
    """Pseudo-test chains of one replicate and the training chains left."""
    n = len(pool.ids)
    pick = np.array([unit_hash(name, salt) < HOLDOUT for name in pool.family_name])
    if tier == "mmseqs30":
        test = pick
    else:
        entry_hash = np.array([unit_hash(pool.pdb_name[i], salt + "entry:") for i in range(n)])
        first = {}
        for i in np.flatnonzero(pick):
            f = pool.family[i]
            if f not in first or entry_hash[i] < entry_hash[first[f]]:
                first[f] = i
        held = np.zeros(pool.pdb.max() + 1, bool)
        held[pool.pdb[list(first.values())]] = True
        test = held[pool.pdb]
    return test, ~test


def frozen(design_rows, calibration, task, event, level, sequential, unit=None):
    """Learned score, design path and one test along it (the parts of crossfit_certify.certify a rule needs)."""
    key = f"leak_{task}_{event}"
    model = cc.learn(design_rows, task, key)
    name = f"score_{task}_{event}"
    for row in [*design_rows, *calibration]:
        row[name] = cc.copy_score(row, task, model)
    cuts = cc.quantile_path(design_rows, (name,))
    found = cc.test_path(calibration, cuts, (name,), key, level, sequential, unit)
    return {"model": model, "trace": found["trace"], "level": level,
            "test": "fixed_sequence" if sequential else "bonferroni", "name": name}


def select(rule, alpha):
    if rule["test"] == "fixed_sequence":
        return cc.fixed_sequence(rule["trace"], alpha, rule["level"])
    return cc.bonferroni(rule["trace"], alpha)


def holt1(pool, train, tier, naive):
    key = pool.cluster if tier == "mmseqs30" else pool.pdb
    rows = pool.rows(np.flatnonzero(train), train, key, naive, ATTACK)
    rules = {}
    for name in EVENTS[pool.dataset]:
        task, event = name.split("_", 1)
        eligible = [row for row in rows if pool.eligible[task][row["index"]]]
        design = [row for row in eligible if cc.pdb_design(row["id"])]
        calibration = [row for row in eligible if not cc.pdb_design(row["id"])]
        rules[name] = frozen(design, calibration, task, event, cc.DELTA / (7 * len(cc.Q)), False)
        rules[name]["calibration_n"] = len(calibration)
    return rules


def holt2(pool, design_of, tier, naive):
    design, calibration, _ = (design_of.rows(tier, naive, ATTACK, members=False) if isinstance(design_of, cc.Split)
                              else design_of.rows(tier, naive, ATTACK))
    rules = {}
    for name in EVENTS[pool.dataset]:
        task, event = name.split("_", 1)
        rules[name] = frozen(design[task], calibration[task], task, event, cc.PER_TEST["pooled"], True)
        rules[name]["calibration_n"] = len(calibration[task])
    return rules


def deploy(pool, rules, test_rows, tier):
    out = {}
    for name, rule in rules.items():
        task, event = name.split("_", 1)
        key = f"leak_{task}_{event}"
        rows = [row for row in test_rows if pool.eligible[task][row["index"]] and not row["duplicate"]]
        scores = np.array([cc.copy_score(row, task, rule["model"]) for row in rows])
        leaks = np.array([bool(row[key]) for row in rows])
        families = np.array([row["family"] for row in rows])
        out[name] = {"calibration_n": rule["calibration_n"], "eligible_test": len(rows)}
        for alpha in ALPHAS:
            i = select(rule, alpha)
            if i is None:
                out[name][str(alpha)] = {"q": None, "n": 0, "leaks": 0, "families": 0}
                continue
            inside = scores < rule["trace"][i]["cut"][0]
            out[name][str(alpha)] = {"q": cc.Q[i], "calibration": [rule["trace"][i]["n"], rule["trace"][i]["leaks"]],
                                     "n": int(inside.sum()), "leaks": int(leaks[inside].sum()),
                                     "families": int(len(set(families[inside])))}
    return out


def replicate(pool, sequence_of, naive, b, tiers, designs):
    salt = f"validity:{b}:"
    result = {"replicate": b}
    for tier in tiers:
        test, train = held_out(pool, tier, salt)
        key = pool.family if tier == "mmseqs30" else pool.pdb
        test_rows = pool.rows(np.flatnonzero(test), train, key, naive, ATTACK)
        left = {sequence_of[i] for i in np.flatnonzero(train)}
        for row in test_rows:
            row["duplicate"] = sequence_of[row["index"]] in left
        result[tier] = {"test_chains": int(test.sum()), "train_chains": int(train.sum())}
        for design in designs:
            started = time.time()
            if design == "holt1":
                rules = holt1(pool, train, tier, naive)
            elif design == "representatives":
                rules = holt2(pool, cc.Representatives(pool, chains=train, tiers=(tier,)), tier, naive)
            elif design == "representatives_random":
                rules = holt2(pool, cc.Representatives(pool, "random", chains=train, tiers=(tier,)), tier, naive)
            else:
                rules = holt2(pool, cc.Split(pool, chains=train), tier, naive)
            result[tier][design] = deploy(pool, rules, test_rows, tier)
            result[tier][design]["seconds"] = round(time.time() - started, 1)
    return result


def summarise(lines):
    """Per design, tier, label set and alpha: realized rate over replicates and how often it exceeds alpha."""
    out = {}
    for line in lines:
        for tier, by_design in line.items():
            if tier == "replicate":
                continue
            for design, by_event in by_design.items():
                if not isinstance(by_event, dict):
                    continue
                for event, cells in by_event.items():
                    if not isinstance(cells, dict):
                        continue
                    for alpha in map(str, ALPHAS):
                        cell = cells[alpha]
                        acc = out.setdefault(tier, {}).setdefault(design, {}).setdefault(event, {}).setdefault(
                            alpha, {"replicates": 0, "nonempty": 0, "n": 0, "leaks": 0, "above_alpha": 0,
                                    "evidence_above": 0, "rates": []})
                        acc["replicates"] += 1
                        if not cell["n"]:
                            continue
                        rate = cell["leaks"] / cell["n"]
                        acc["nonempty"] += 1
                        acc["n"] += cell["n"]
                        acc["leaks"] += cell["leaks"]
                        acc["rates"].append(rate)
                        acc["above_alpha"] += rate > float(alpha)
                        # One-sided: k answered chains of n are unlikely (p < 0.05) if the rule's risk were alpha.
                        acc["evidence_above"] += binom.sf(cell["leaks"] - 1, cell["n"], float(alpha)) < .05
    for tier in out.values():
        for design in tier.values():
            for event in design.values():
                for alpha, acc in event.items():
                    rates = acc.pop("rates")
                    acc["pooled_rate"] = acc["leaks"] / acc["n"] if acc["n"] else None
                    acc["mean_rate"] = float(np.mean(rates)) if rates else None
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=("ec", "go"))
    parser.add_argument("--replicates", nargs=2, type=int, default=(0, 20), metavar=("FIRST", "STOP"))
    parser.add_argument("--tiers", nargs="+", default=["mmseqs30", "pdb"])
    parser.add_argument("--designs", nargs="+", default=["holt1", "representatives", "split"])
    args = parser.parse_args()
    pool = Pool(args.dataset)
    seq = sequences(DATA[args.dataset] / "train.fasta")
    sequence_of = [seq[name] for name in pool.ids]
    train = records(DATA[args.dataset] / "train.jsonl")
    naive = {task: cc.naive_prediction(train, task)[2] for task in pool.tasks if task != "ec"}
    tag = "" if args.designs == ["holt1", "representatives", "split"] else "-" + "+".join(args.designs)
    path = RAW / f"crossfit-{args.dataset}-validity{tag}-{args.replicates[0]}-{args.replicates[1]}.jsonl"
    with path.open("a") as handle:
        for b in range(*args.replicates):
            started = time.time()
            line = replicate(pool, sequence_of, naive, b, args.tiers, args.designs)
            handle.write(json.dumps(line) + "\n")
            handle.flush()
            print(b, round(time.time() - started), "s", flush=True)
    print(path, flush=True)


if __name__ == "__main__":
    main()
