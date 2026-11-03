"""Certify low label-transfer risk on a frozen monotone search-threshold path.

Only validation labels choose thresholds. Test labels are used after freezing.
Bonferroni learn-then-test permits the largest passing candidate even if an
earlier candidate has too few observations to pass a fixed sequence test.
The binomial guarantee assumes iid calibration chains conditional on entry.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

from scipy.stats import beta, binom

from review_paths import dataset_dir


TASKS = {"ec": ("ec",), "go": ("bp", "mf", "cc")}


def kind(dataset):
    """The label family of a dataset, 'ec' or 'go'; ProteinShake splits are named 'ps-<task>-<split>'."""
    return dataset.split("-")[1] if dataset.startswith("ps-") else dataset


# Frozen before inspecting calibration outcomes. Every coordinate is nondecreasing.
PATH = ((0.05, 25), (0.10, 50), (0.15, 75), (0.20, 100),
        (0.25, 150), (0.30, 200), (0.35, 300), (0.40, 400),
        (0.50, 600), (0.70, 1000), (1.01, 1e9))
ROUTES = ("mmseqs", "foldseek", "esm2", "esmc")


def records(path):
    with path.open() as file:
        return [json.loads(line) for line in file]


def sequences(path):
    result = {}
    name, parts = None, []
    with path.open() as file:
        for line in file:
            line = line.strip()
            if line.startswith(">"):
                if name is not None:
                    result[name] = "".join(parts)
                name, parts = line[1:].split()[0], []
            elif line:
                parts.append(line)
    if name is not None:
        result[name] = "".join(parts)
    return result


def f1(left, right):
    a, b = set(left), set(right)
    return 2 * len(a & b) / (len(a) + len(b)) if a and b else 0.0


def wilson(k, n, z=1.959963984540054):
    if not n:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    center = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [max(0, center - half), min(1, center + half)]


def cp_upper(k, n, delta):
    return float(beta.ppf(1 - delta, k + 1, n - k)) if n and k < n else 1.0


def hit_files(root, base, dataset, part):
    if dataset.startswith("ps-"):
        # ProteinShake: the audit's held-out-to-train tables (scripts/run_audit_external.sh) and the
        # embedding neighbours of scripts/proteinshake_embed.py.
        key = dataset[3:]
        stems = {"mmseqs": f"audit-ps-{key}-mmseqs-{part}.tsv",
                 "foldseek": f"audit-ps-{key}-foldseek-{part}.tsv",
                 "esm2": f"esm2-650m-{dataset}-{part}-hits.tsv",
                 "esmc": f"esmc-600m-{dataset}-{part}-hits.tsv"}
    elif base == "deepfri":
        stems = {"mmseqs": f"audit-deepfri-original-{dataset}-mmseqs-{part}.tsv",
                 "foldseek": f"foldseek-deepfri-{dataset}-{part}-hits.tsv",
                 "esm2": f"esm2-650m-deepfri-{dataset}-{part}-hits.tsv",
                 "esmc": f"esmc-600m-deepfri-{dataset}-{part}-hits.tsv"}
    else:
        # EC sequence routes read the restored residue letters, as the audit's MMseqs2 tables do
        # (scripts/run_seqfix_ec.sh); the esm2-650m-ec/esmc-600m-ec tables hold MotiL's merged letters.
        sf = "sf" if dataset == "ec" else ""
        stems = {"mmseqs": f"audit-deepfri-{dataset}-mmseqs-{part}.tsv",
                 "foldseek": f"foldseek-{dataset}-{part}-hits.tsv",
                 "esm2": f"esm2{sf}-650m-{dataset}-{part}-hits.tsv",
                 "esmc": f"esmc{sf}-600m-{dataset}-{part}-hits.tsv"}
    return {route: root / name for route, name in stems.items()}


def best_transfer(path, route, targets, queries, tasks):
    """Stream hits, retaining best score and best annotated hit per task/query."""
    best_score = defaultdict(float)
    best_annotated = {task: {} for task in tasks}
    if not path.exists():
        raise FileNotFoundError(f"{route} reference hits absent: {path}")
    with path.open() as file:
        for line in file:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            q, t = parts[:2]
            if q not in queries or t not in targets or q == t:
                continue
            rank = float(parts[6] if route == "mmseqs" else parts[2])
            if rank > best_score[q]:
                best_score[q] = rank
            for task in tasks:
                if targets[t].get(task):
                    old = best_annotated[task].get(q)
                    if old is None or rank > old[0]:
                        best_annotated[task][q] = (rank, t)
    return best_score, best_annotated


def identity_scores(path, targets, queries):
    maximum = defaultdict(float)
    with path.open() as file:
        for line in file:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 7:
                raise ValueError(f"MMseqs identity/coverage columns missing: {path}")
            q, t = parts[:2]
            if q in queries and t in targets and q != t:
                maximum[q] = max(maximum[q], float(parts[2]))
    return maximum


def chain_table(data, raw, base, dataset, part, gamma):
    train = records(data / "train.jsonl")
    held = records(data / f"{part}.jsonl")
    tasks = TASKS[dataset]
    train_labels = {r["id"]: r["labels"] for r in train}
    query_labels = {r["id"]: r["labels"] for r in held}
    train_seq = sequences(data / "train.fasta")
    query_seq = sequences(data / f"{part}.fasta")
    if set(query_labels) != set(query_seq) or set(train_labels) != set(train_seq):
        raise ValueError(f"{data} {part}: FASTA/JSONL IDs differ")
    train_entries = {name.split("-", 1)[0] for name in train_labels}
    entry_labels = defaultdict(list)
    for name, labels in train_labels.items():
        entry_labels[name.split("-", 1)[0]].append(labels)
    train_sequences = set(train_seq.values())
    paths = hit_files(raw, base, dataset, part)
    identity = identity_scores(paths["mmseqs"], train_labels, query_labels)
    route_scores = {}
    annotated = {}
    for route in ROUTES:
        route_scores[route], annotated[route] = best_transfer(
            paths[route], route, train_labels, query_labels, tasks)
    table = []
    for row in held:
        q = row["id"]
        labels = row["labels"]
        sibling = q.split("-", 1)[0] in train_entries
        duplicate = query_seq[q] in train_sequences
        result = {"id": q, "pdb": q.split("-", 1)[0], "length": len(query_seq[q]),
                  "identity": identity[q], "bits": route_scores["foldseek"][q],
                  "sibling": sibling, "duplicate": duplicate}
        for task in tasks:
            result[f"labels_{task}"] = len(labels[task])
            route_f1 = {}
            for route in ROUTES:
                hit = annotated[route][task].get(q)
                route_f1[route] = f1(train_labels[hit[1]][task], labels[task]) if hit else 0.0
                result[f"f1_{route}_{task}"] = route_f1[route]
            route_f1["sibling"] = max(
                (f1(neighbor[task], labels[task]) for neighbor in entry_labels[result["pdb"]]
                 if neighbor[task]), default=0.0)
            result[f"f1_sibling_{task}"] = route_f1["sibling"]
            result[f"leak_{task}"] = bool(labels[task] and max(route_f1.values()) >= gamma)
            result[f"leak_lm_only_{task}"] = bool(labels[task] and
                max(route_f1["esm2"], route_f1["esmc"]) >= gamma and
                max(route_f1["mmseqs"], route_f1["foldseek"], route_f1["sibling"]) < gamma)
        table.append(result)
    return table


def eligible(row, identity, bits):
    return (row["identity"] < identity and row["bits"] < bits
            and not row["sibling"] and not row["duplicate"])


def select_path(valid, task, alpha, delta):
    trace = []
    selected = None
    per_test_delta = delta / (4 * len(PATH))
    for index, (identity, bits) in enumerate(PATH):
        group = [r for r in valid if r[f"labels_{task}"] and eligible(r, identity, bits)]
        n = len(group)
        k = sum(r[f"leak_{task}"] for r in group)
        p = float(binom.cdf(k, n, alpha)) if n else 1.0
        passed = p <= per_test_delta
        trace.append({"index": index, "identity": identity, "bits": bits,
                      "n": n, "leaks": k, "p": p, "upper": cp_upper(k, n, per_test_delta),
                      "reject_null": passed})
        if passed:
            selected = index
    return selected, trace


def test_rate(rows, task, mask):
    picked = [r for r in rows if r[f"labels_{task}"] and mask(r)]
    n = len(picked)
    k = sum(r[f"leak_{task}"] for r in picked)
    return {"n": n, "leaks": k, "rate": k / n if n else None,
            "wilson95": wilson(k, n),
            "lm_only": sum(r[f"leak_lm_only_{task}"] for r in picked)}


def gamma_curve(rows, task, mask):
    group = [r for r in rows if r[f"labels_{task}"] and mask(r)]
    return {str(gamma): {"n": len(group), "leaks": sum(
        max(r[f"f1_{route}_{task}"] for route in (*ROUTES, "sibling")) >= gamma
        for r in group)} for gamma in (.25, .5, .75, 1.0)}


def strata(valid, test, task, picked, alpha, delta, fixed_bits):
    lengths = sorted(r["length"] for r in valid if r[f"labels_{task}"])
    densities = sorted(r[f"labels_{task}"] for r in valid if r[f"labels_{task}"])
    def cut(values, frac):
        return values[min(len(values) - 1, int(frac * len(values)))]
    boundaries = {"length": [cut(lengths, 1/3), cut(lengths, 2/3)],
                  "labels": [cut(densities, 1/3), cut(densities, 2/3)]}
    result = {}
    for field, edges in boundaries.items():
        for bucket in range(3):
            def inside(r, field=field, edges=edges, bucket=bucket):
                value = r["length"] if field == "length" else r[f"labels_{task}"]
                return (value <= edges[0] if bucket == 0 else
                        edges[0] < value <= edges[1] if bucket == 1 else value > edges[1])
            result[f"{field}_{bucket}"] = {
                "valid_all": test_rate(valid, task, inside),
                "test_all": test_rate(test, task, inside),
                "certified": test_rate(test, task, lambda r: inside(r) and picked(r)),
                "fixed30": test_rate(test, task, lambda r: inside(r) and r["identity"] < .30),
                "fixed_30_bstar": test_rate(test, task,
                    lambda r: inside(r) and eligible(r, .30, fixed_bits))}
            # Six disjoint/overlapping reported bins share a Bonferroni budget.
            local_index, _ = select_path([r for r in valid if inside(r)], task, alpha, delta / 6)
            result[f"{field}_{bucket}"]["mondrian_index"] = local_index
            result[f"{field}_{bucket}"]["mondrian_test"] = test_rate(test, task,
                lambda r: inside(r) and local_index is not None and eligible(r, *PATH[local_index]))
    return {"edges": boundaries, "cells": result}


def reweighted_validation(valid, test, task, edges):
    """Descriptive covariate-shift sensitivity, never used as a finite-sample certificate."""
    def cell(row):
        length = row["length"]
        labels = row[f"labels_{task}"]
        return (sum(length > edge for edge in edges["length"]),
                sum(labels > edge for edge in edges["labels"]))
    v = [r for r in valid if r[f"labels_{task}"]]
    t = [r for r in test if r[f"labels_{task}"]]
    cv = defaultdict(int)
    ct = defaultdict(int)
    for r in v:
        cv[cell(r)] += 1
    for r in t:
        ct[cell(r)] += 1
    weights = {key: ((ct[key] + .5) / (len(t) + 4.5)) /
                    ((cv[key] + .5) / (len(v) + 4.5)) for key in set(cv) | set(ct)}
    estimates = []
    for index, (identity, bits) in enumerate(PATH):
        selected = [r for r in v if eligible(r, identity, bits)]
        denominator = sum(weights[cell(r)] for r in selected)
        numerator = sum(weights[cell(r)] * r[f"leak_{task}"] for r in selected)
        estimates.append({"index": index, "n": len(selected),
                          "weighted_risk": numerator / denominator if denominator else None,
                          "effective_n": denominator ** 2 /
                          sum(weights[cell(r)] ** 2 for r in selected) if denominator else 0})
    return {"weights_by_bin": {str(key): value for key, value in weights.items()},
            "estimates": estimates}


def save_chains(path, rows, selected):
    out = []
    for row in rows:
        entry = dict(row)
        for task, index in selected.items():
            entry[f"certified_{task}"] = bool(index is not None and eligible(row, *PATH[index]))
            entry[f"fixed30_{task}"] = eligible(row, .30, float("inf"))
        out.append(entry)
    with path.open("w") as file:
        writer = csv.DictWriter(file, fieldnames=list(out[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/prepared"))
    parser.add_argument("--raw-root", type=Path, default=Path("results/raw"))
    parser.add_argument("--base", default=None)
    parser.add_argument("--alpha", type=float, default=.1)
    parser.add_argument("--delta", type=float, default=.1)
    parser.add_argument("--gamma", type=float, default=.5)
    args = parser.parse_args()
    base = args.base or ("deepfri" if args.data_root.name == "deepfri" else "old")
    output = {"base": base, "alpha": args.alpha, "delta": args.delta,
              "gamma": args.gamma, "path": PATH, "tasks": {}}
    for dataset in TASKS:
        data = dataset_dir(args.data_root, dataset)
        valid = chain_table(data, args.raw_root, base, dataset, "valid", args.gamma)
        test = chain_table(data, args.raw_root, base, dataset, "test", args.gamma)
        audit_stem = f"audit-deepfri-{'original-' if base == 'deepfri' else ''}{dataset}-chains.tsv"
        with (args.raw_root / audit_stem).open() as file:
            official = {row["id"]: row["official_lt30"] == "True"
                        for row in csv.DictReader(file, delimiter="\t")}
        for row in test:
            row["official_lt30"] = official[row["id"]]
        selected = {}
        for task in TASKS[dataset]:
            # Bonferroni across four reported ontology risks.
            index, trace = select_path(valid, task, args.alpha, args.delta)
            selected[task] = index
            mask = (lambda r, index=index: index is not None and eligible(r, *PATH[index]))
            report_stem = f"audit-deepfri-{'original-' if base == 'deepfri' else ''}{dataset}.json"
            fixed_bits = json.loads((args.raw_root / report_stem).read_text())["calibration"]["strong_bits"]
            output["tasks"][task] = {
                "selected_index": index, "selected_threshold": PATH[index] if index is not None else None,
                "trace": trace, "validation": test_rate(valid, task, mask),
                "test": test_rate(test, task, mask),
                "official_lt30": test_rate(test, task, lambda r: r["official_lt30"]),
                "fixed30": test_rate(test, task, lambda r: r["identity"] < .30),
                "fixed25": test_rate(test, task, lambda r: r["identity"] < .25),
                "fixed40": test_rate(test, task, lambda r: r["identity"] < .40),
                "fixed_bits50": test_rate(test, task, lambda r: r["bits"] < 50),
                "fixed_bstar": test_rate(test, task, lambda r: r["bits"] < fixed_bits) if fixed_bits else None,
                "fixed_30_bstar": test_rate(test, task, lambda r: eligible(r, .30, fixed_bits)) if fixed_bits else None,
                "strata": strata(valid, test, task, mask, args.alpha, args.delta, fixed_bits),
                "gamma_curve_fixed_30_bstar": gamma_curve(test, task,
                    lambda r: eligible(r, .30, fixed_bits)) if fixed_bits else None,
                "gamma_curve_certified": gamma_curve(test, task, mask),
                "shift": {"valid_n": sum(bool(r[f"labels_{task}"]) for r in valid),
                          "test_n": sum(bool(r[f"labels_{task}"]) for r in test),
                          "valid_mean_length": sum(r["length"] for r in valid if r[f"labels_{task}"]) /
                              sum(bool(r[f"labels_{task}"]) for r in valid),
                          "test_mean_length": sum(r["length"] for r in test if r[f"labels_{task}"]) /
                              sum(bool(r[f"labels_{task}"]) for r in test),
                          "valid_mean_labels": sum(r[f"labels_{task}"] for r in valid) /
                              sum(bool(r[f"labels_{task}"]) for r in valid),
                          "test_mean_labels": sum(r[f"labels_{task}"] for r in test) /
                              sum(bool(r[f"labels_{task}"]) for r in test)},
                "full": test_rate(test, task, lambda r: True)}
            output["tasks"][task]["other_alphas"] = {
                str(alpha): select_path(valid, task, alpha, args.delta)[0]
                for alpha in (.05, .20)}
            output["tasks"][task]["covariate_weighted_diagnostic"] = reweighted_validation(
                valid, test, task, output["tasks"][task]["strata"]["edges"])
        save_chains(args.raw_root / f"certify-{base}-{dataset}-valid-chains.tsv", valid, selected)
        save_chains(args.raw_root / f"certify-{base}-{dataset}-chains.tsv", test, selected)
        print(dataset, {task: (output["tasks"][task]["selected_threshold"],
                              output["tasks"][task]["test"]) for task in TASKS[dataset]}, flush=True)
    (args.raw_root / f"certify-{base}-summary.json").write_text(json.dumps(output, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
