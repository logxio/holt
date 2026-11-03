"""Four-score learn-then-test certification on the existing DeepFRI old split.

The path construction uses validation scores only. Labels enter only the
reference attack and the binomial test; test labels enter after selection.
"""
from __future__ import annotations

import csv
import argparse
import gzip
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import binom

from certify_split import (PATH, ROUTES, TASKS, cp_upper, dataset_dir,
                           eligible, f1, hit_files, kind, records, sequences, wilson)

RAW = Path("results/raw")
FAMILY = Path("data/external/sifts-family/pdb_chain_pfam.tsv.gz")
QUANTILES = tuple(range(1, 101))  # Frozen before reference outcomes are read.
GAMMAS = (.25, .5, .75, 1.)


def ec4(labels):
    return {x for x in labels if len(x.split(".")) == 4 and all(p and p != "-" for p in x.split("."))}


def copied(pred, true, task, gamma):
    if task == "ec":
        return bool(ec4(pred) & ec4(true))
    return f1(pred, true) >= gamma


def hit_panel(path, route, targets, queries, tasks):
    """Keep five annotated hits per route/task; also the top raw hit score."""
    scores = defaultdict(float)
    top = {task: defaultdict(list) for task in tasks}
    with path.open() as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if len(fields) < (7 if route == "mmseqs" else 3):
                continue
            q, t = fields[:2]
            if q not in queries or t not in targets or q == t:
                continue
            value = float(fields[6] if route == "mmseqs" else fields[2])
            scores[q] = max(scores[q], value)
            for task in tasks:
                if not targets[t].get(task):
                    continue
                hits = top[task][q]
                hits.append((value, t))
                hits.sort(reverse=True)
                del hits[5:]
    return scores, top


def vote(hits, labels):
    counts = Counter(term for _, name in hits for term in set(labels[name]))
    return {term for term, count in counts.items() if count > len(hits) / 2}


def panel_features(panels, labels, task):
    """Label-free view of what the attacker sees: mean top-5 vote size over
    routes, the largest share of top-5 annotated hits holding one term, and
    the most routes whose top-1 annotated hit shares one term."""
    terms_of = (lambda name: ec4(labels[name])) if task == "ec" else (lambda name: set(labels[name]))
    sizes, shares, tops = [], [], Counter()
    for hits in panels:
        if not hits:
            sizes.append(0)
            continue
        sizes.append(len(vote(hits, labels)))
        terms = Counter(term for _, name in hits for term in terms_of(name))
        shares.append(max(terms.values(), default=0) / len(hits))
        tops.update(terms_of(hits[0][1]))
    return {"vote_size": float(np.mean(sizes)), "agree": max(shares, default=0.),
            "consensus": max(tops.values(), default=0)}


def build_rows(data, dataset, part, naive_labels=None):
    train = records(data / "train.jsonl")
    held = records(data / f"{part}.jsonl")
    train_labels = {r["id"]: r["labels"] for r in train}
    query_labels = {r["id"]: r["labels"] for r in held}
    train_seq = sequences(data / "train.fasta")
    query_seq = sequences(data / f"{part}.fasta")
    if set(train_labels) != set(train_seq) or set(query_labels) != set(query_seq):
        raise ValueError(f"FASTA/JSONL ID mismatch: {data} {part}")
    paths = hit_files(RAW, "old", dataset, part)
    scores, panel = {}, {}
    for route in ROUTES:
        scores[route], panel[route] = hit_panel(paths[route], route, train_labels,
                                                query_labels, TASKS[kind(dataset)])
    identity = defaultdict(float)
    with paths["mmseqs"].open() as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 7:
                raise ValueError(f"identity column missing: {paths['mmseqs']}")
            q, t = fields[:2]
            if q in query_labels and t in train_labels and q != t:
                identity[q] = max(identity[q], float(fields[2]))
    sibling_labels = defaultdict(list)
    labels_by_task = {task: {name: labels[task] for name, labels in train_labels.items()}
                      for task in TASKS[kind(dataset)]}
    for name, labels in train_labels.items():
        sibling_labels[name.split("-", 1)[0]].append(labels)
    seqset = set(train_seq.values())
    out = []
    for row in held:
        q = row["id"]
        pdb = q.split("-", 1)[0]
        result = {"id": q, "pdb": pdb, "length": len(query_seq[q]),
                  "identity": identity[q], "bits": scores["foldseek"][q],
                  "esm2": scores["esm2"][q], "esmc": scores["esmc"][q],
                  "sibling": pdb in sibling_labels, "duplicate": query_seq[q] in seqset}
        for task in TASKS[kind(dataset)]:
            true = row["labels"][task]
            result[f"labels_{task}"] = len(true)
            for name, value in panel_features([panel[route][task][q] for route in ROUTES],
                                              labels_by_task[task], task).items():
                result[f"{name}_{task}"] = value
            attacks = []
            for route in ROUTES:
                hits = panel[route][task][q]
                if hits:
                    attacks.append(train_labels[hits[0][1]][task])
                    attacks.append(vote(hits, labels_by_task[task]))
            attacks += [neighbor[task] for neighbor in sibling_labels.get(pdb, ()) if neighbor[task]]
            if naive_labels is not None and task != "ec" and true:
                baseline_f1 = f1(naive_labels[task], true)
                result[f"leak_{task}_relative"] = any(
                    f1(pred, true) >= .5 and f1(pred, true) > baseline_f1 for pred in attacks)
            for gamma in GAMMAS:
                result[f"leak_{task}_{gamma}"] = any(copied(pred, true, task, gamma)
                                                    for pred in attacks) if true else False
                if task == "ec":
                    result[f"leak_ec_f1_{gamma}"] = any(f1(pred, true) >= gamma for pred in attacks) if true else False
        out.append(result)
    return out


def passes(row, cut):
    return not (row["sibling"] or row["duplicate"]) and all(
        row[key] < bound for key, bound in zip(("identity", "bits", "esm2", "esmc"), cut))


def path(valid, task):
    coords = ("identity", "bits", "esm2", "esmc")
    # Feature-design PDBs are disjoint from the later risk-calibration PDBs.
    values = {name: np.array([r[name] for r in valid]) for name in coords}
    result = []
    for q in QUANTILES:
        cut = []
        for name in coords:
            v = float(np.percentile(values[name], q))
            cut.append(math.nextafter(v, math.inf) if q == 100 else v)
        result.append(tuple(cut))
    return result


def rate(rows, task, predicate, gamma, ec_f1=False):
    chosen = [r for r in rows if r[f"labels_{task}"] and predicate(r)]
    n = len(chosen)
    key = f"leak_ec_f1_{gamma}" if task == "ec" and ec_f1 else f"leak_{task}_{gamma}"
    k = sum(r[key] for r in chosen)
    return {"n": n, "leaks": k, "rate": k / n if n else None,
            "wilson95": wilson(k, n)}


def certify(valid, task, cuts, alpha, delta, gamma, ec_f1=False):
    trace = []
    chosen = None
    per = delta / (4 * len(cuts))
    for i, cut in enumerate(cuts):
        point = rate(valid, task, lambda r: passes(r, cut), gamma, ec_f1=ec_f1)
        n, k = point["n"], point["leaks"]
        p = float(binom.cdf(k, n, alpha)) if n else 1.
        trace.append({"q": QUANTILES[i], "cut": cut, **point, "upper": cp_upper(k, n, per),
                      "p": p, "certified": p <= per})
        if p <= per:
            chosen = i
    return chosen, trace


def sifts():
    mapping = defaultdict(set)
    with gzip.open(FAMILY, "rt") as handle:
        reader = csv.DictReader((line for line in handle if not line.startswith("#")), delimiter="\t")
        for row in reader:
            mapping[(row["PDB"].upper(), row["CHAIN"])].add(row["PFAM_ID"])
    return mapping


def read_chains(path):
    with path.open() as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    for row in rows:
        for key in ("length", "labels_ec", "labels_bp", "labels_mf", "labels_cc"):
            if key in row:
                row[key] = int(row[key])
        for key in ("identity", "bits", "esm2", "esmc"):
            row[key] = float(row[key])
        for key in list(row):
            if key in ("sibling", "duplicate", "official_lt30") or key.startswith(("leak_", "certified_")):
                row[key] = row[key] == "True"
    return rows


def family_counts(data, test, task, masks, mapping):
    train = records(data / "train.jsonl")
    train_families = set().union(*(mapping.get(tuple(r["id"].split("-", 1)), set()) for r in train))
    output = {}
    for name, mask in masks.items():
        chosen = [r for r in test if r[f"labels_{task}"] and mask(r)]
        families = [mapping.get(tuple(r["id"].split("-", 1)), set()) for r in chosen]
        mapped = sum(bool(x) for x in families)
        shared = sum(bool(x & train_families) for x in families)
        output[name] = {"n": len(chosen), "mapped": mapped, "mapped_fraction": mapped / len(chosen) if chosen else None,
                        "shared": shared, "shared_all_fraction": shared / len(chosen) if chosen else None,
                        "shared_mapped_fraction": shared / mapped if mapped else None}
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reuse-chains", action="store_true",
                        help="recalculate from saved per-chain attacks without rereading hit tables")
    args = parser.parse_args()
    all_results = {"definition": "top1 or strict-majority top5 per route; same-PDB sibling; EC exact level 4, GO F1>=0.5",
                   "delta": .1, "quantiles": QUANTILES, "tasks": {},
                   "sifts_source_header": "2026/10/04 - 08:31 | PDB: 40.26 | UniProt: 2026.03"}
    mapping = sifts()
    for dataset in TASKS:
        data = dataset_dir(Path("data/prepared"), dataset)
        if args.reuse_chains:
            valid_all = read_chains(RAW / f"certify-four-{dataset}-valid-chains.tsv")
            test = read_chains(RAW / f"certify-four-{dataset}-test-chains.tsv")
        else:
            valid_all, test = build_rows(data, dataset, "valid"), build_rows(data, dataset, "test")
        def design_pdb(row):
            digest = hashlib.sha256(row["pdb"].encode()).digest()
            return int.from_bytes(digest[:4], "big") / 2**32 < .2
        design = [r for r in valid_all if design_pdb(r)]
        valid = [r for r in valid_all if not design_pdb(r)]
        for row in valid_all:
            row["validation_role"] = "feature_design" if design_pdb(row) else "risk_calibration"
        with (RAW / f"audit-deepfri-{dataset}-chains.tsv").open() as handle:
            official = {row["id"]: row["official_lt30"] == "True"
                        for row in csv.DictReader(handle, delimiter="\t")}
        bstar = json.loads((RAW / f"audit-deepfri-{dataset}.json").read_text())["calibration"]["strong_bits"]
        for row in test:
            row["official_lt30"] = official[row["id"]]
        for task in TASKS[kind(dataset)]:
            cuts = path(design, task)
            results = {}
            for alpha in (.05, .1, .2):
                idx, trace = certify(valid, task, cuts, alpha, .1, 1. if task == "ec" else .5)
                mask = (lambda r, idx=idx: idx is not None and passes(r, cuts[idx]))
                results[str(alpha)] = {"selected_index": idx, "selected_q": QUANTILES[idx] if idx is not None else None,
                                       "cut": cuts[idx] if idx is not None else None,
                                       "trace": trace, "valid": rate(valid, task, mask, 1. if task == "ec" else .5),
                                       "test": rate(test, task, mask, 1. if task == "ec" else .5),
                                       "gamma_curve_f1": {str(g): rate(test, task, mask, g, ec_f1=True) for g in GAMMAS}}
                for row in test:
                    row[f"certified_{task}_{alpha}"] = bool(mask(row))
            # The old two-score path is scored with this stronger, frozen reference panel.
            two = []
            for a, b in PATH:
                point = rate(valid_all, task, lambda r, a=a, b=b: eligible(r, a, b), 1. if task == "ec" else .5)
                two.append({"identity": a, "bits": b, **point,
                            "upper": cp_upper(point["leaks"], point["n"], .1 / (4 * len(PATH)))})
            current = lambda r: eligible(r, .30, bstar)
            masks = {"certified": lambda r: r[f"certified_{task}_0.1"],
                     "current_hard": current, "official_lt30": lambda r: r["official_lt30"]}
            edges = {field: [float(np.percentile([r[field] for r in valid if r[f"labels_{task}"]], q))
                             for q in (100/3, 200/3)] for field in ("length", f"labels_{task}")}
            strata = {}
            for field, bounds in edges.items():
                for bucket in range(3):
                    inside = lambda r, field=field, bounds=bounds, bucket=bucket: (
                        r[field] <= bounds[0] if bucket == 0 else
                        bounds[0] < r[field] <= bounds[1] if bucket == 1 else r[field] > bounds[1])
                    strata[f"{field}_{bucket}"] = rate(test, task,
                        lambda r: inside(r) and r[f"certified_{task}_0.1"], 1. if task == "ec" else .5)
            all_results["tasks"][task] = {"alphas": results, "two_dimensional": two,
                "feature_design_n": len(design), "risk_calibration_n": len(valid),
                "two_dimensional_min_observed": min((x["rate"] for x in two if x["n"]), default=None),
                "gamma_sensitivity": {str(g): {
                    "selected_q": (QUANTILES[i] if i is not None else None),
                    "test": rate(test, task,
                        lambda r, i=i: i is not None and passes(r, cuts[i]), g,
                        ec_f1=task == "ec")}
                    for g in GAMMAS
                    for i, _ in [certify(valid, task, cuts, .1, .1, g, ec_f1=task == "ec")]},
                "strata_edges": edges, "strata": strata,
                "family": family_counts(data, test, task, masks, mapping),
                "current_hard": rate(test, task, current, 1. if task == "ec" else .5),
                "current_hard_gamma_curve_f1": {
                    str(g): rate(test, task, current, g, ec_f1=task == "ec") for g in GAMMAS},
                "official_lt30": rate(test, task, lambda r: r["official_lt30"], 1. if task == "ec" else .5)}
            print(task, {a: (x["selected_q"], x["test"]) for a, x in results.items()}, flush=True)
        for name, rows in (("valid", valid_all), ("test", test)):
            path_out = RAW / f"certify-four-{dataset}-{name}-chains.tsv"
            with path_out.open("w") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
                writer.writeheader(); writer.writerows(rows)
    (RAW / "certify-four-summary.json").write_text(json.dumps(all_results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
