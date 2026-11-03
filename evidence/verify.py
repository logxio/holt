"""Verify the paper tables against portable input records and chain-level evidence."""
from __future__ import annotations

import csv
import gzip
import hashlib
import importlib.util
import json
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parent
sys.path.insert(0, str(PACKAGE / "src"))
from holt._reproduce import reproduce
from holt._reproduce_scores import block_draws, fmax, sparse_counts


def require(condition, message):
    if not condition:
        raise ValueError(message)


def unpack(name):
    with gzip.open(HERE / name, "rt") as handle:
        return json.load(handle)


def average_precision(predictions, truth, ids, terms):
    pos = {t: i for i, t in enumerate(terms)}
    scores = np.zeros((len(ids), len(terms)), dtype=np.float32)
    labels = np.zeros_like(scores, dtype=np.uint8)
    for i, name in enumerate(ids):
        for term, score in predictions[name].items():
            scores[i, pos[term]] = score
        labels[i, [pos[t] for t in truth[name]]] = 1
    order = np.argsort(-scores.ravel(), kind="stable")
    ranked = scores.ravel()[order]
    ends = np.r_[np.flatnonzero(ranked[:-1] != ranked[1:]), len(ranked) - 1]
    tp = np.cumsum(labels.ravel()[order], dtype=np.int64)[ends]
    return float(np.sum(tp / (ends + 1) * (tp - np.r_[0, tp[:-1]])) / labels.sum())


def verify_numbers():
    inputs = unpack("paper-inputs.json.gz")
    manifest = json.loads((HERE / "paper-inputs-manifest.json").read_text())
    require(set(inputs) == {r["path"] for r in manifest}, "paper input manifest coverage differs")
    for row in manifest:
        content = inputs[row["path"]].encode()
        require(hashlib.sha256(content).hexdigest() == row["stored_sha256"], row["path"] + ": input hash differs")
    with tempfile.TemporaryDirectory(prefix="holt-paper-") as directory:
        root = Path(directory).resolve()
        for name, content in inputs.items():
            path = root / name
            require(path.resolve().is_relative_to(root), "paper input escapes replay directory")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
        spec = importlib.util.spec_from_file_location("holt_paper_numbers", HERE / "_paper_numbers.py")
        numbers = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(numbers)
        numbers.ROOT, numbers.HERE, numbers.RAW = root, root / "paper/recomb2027", root / "results/raw"
        numbers.OUT = numbers.HERE / "generated"
        numbers.OUT.mkdir(exist_ok=True)
        numbers.PREPARED = {key: root / f"data/prepared/deepfri/{key.lower()}" for key in ("EC", "GO")}
        numbers.RELEASED_EC = root / "data/prepared/ec"
        numbers.REPO = root / "holt/certified_sets"
        sys.modules["make_numbers"] = numbers
        numbers.main()
        tables = list((HERE / "paper-tables").glob("*.tex"))
        for expected in tables:
            require((numbers.OUT / expected.name).read_bytes() == expected.read_bytes(),
                    expected.name + ": regenerated paper table differs")
        print(f"Paper evidence: {len(numbers.macros)} number macros and {len(tables)} tables exact")
    return inputs


def verify_equal_length():
    data = unpack("equal-length-ec.json.gz")
    ids = data["ids"]
    require(len(ids) == len(set(ids)) == 338, "equal-length cohort differs")
    substitution = str.maketrans({"L": "I", "E": "K", "Y": "R"})
    for name, row in data["sequences"].items():
        require(len(row["release"]) == len(row["archive"]) and row["release"] != row["archive"]
                and row["release"].translate(substitution) == row["archive"], name + ": substitution or length differs")
    counts = {}
    for method, predictions in data["predictions"].items():
        counts[method] = sparse_counts(ids, data["truth"], predictions)
        require(abs(fmax(counts[method]) - data["scores"][method]["fmax"]) < 1e-10, method + ": equal-length Fmax differs")
        require(abs(average_precision(predictions, data["truth"], ids, data["terms"])
                    - data["scores"][method]["aupr"]) < 1e-10, method + ": equal-length AUPR differs")
    draws = block_draws(counts, data["families"])
    interval = np.percentile(draws["release"] - draws["archive"], [2.5, 97.5])
    require(np.max(np.abs(interval - data["paired"]["ci95"])) < 1e-10, "equal-length paired interval differs")
    print("Equal-length EC: 338 chains, zero truncations, scores and 2,000-draw paired interval exact")


def verify_two_route():
    with gzip.open(PACKAGE / "certified_sets/two-route-predictions.json.gz", "rt") as handle:
        data = json.load(handle)
    with (HERE / "two-route-scores.tsv").open() as handle:
        expected = list(csv.DictReader(handle, delimiter="\t"))
    checked = 0
    for row in expected:
        task = data["tasks"][row["task"]]
        ids = task["subsets"].get(row["subset"])
        if not ids:
            continue
        value = fmax(sparse_counts(ids, task["truth"], task["predictions"]))
        aupr = average_precision(task["predictions"], task["truth"], ids, task["terms"])
        require(abs(value - float(row["fmax"])) < 2e-6 and abs(aupr - float(row["aupr"])) < 2e-6,
                str((row["task"], row["subset"])) + ": two-route score differs")
        checked += 1
    require(checked >= 12, "full, stated-bin or certified two-route scores are absent")
    print(f"Two-route: {checked} Fmax and AUPR cells exact")


def verify_families():
    with np.load(PACKAGE / "certified_sets/temporal-scores.npz", allow_pickle=False) as arrays:
        for task in ("bp", "mf", "cc"):
            lookup = dict(zip(arrays[f"{task}/ids"], arrays[f"{task}/families"]))
            for table in (PACKAGE / "certified_sets/temporal").glob(f"{task}-*/chains.tsv"):
                with table.open() as handle:
                    rows = list(csv.DictReader(handle, delimiter="\t"))
                require(all(row["family_or_cluster"] == lookup[row["id"]] for row in rows), str(table) + ": family keys differ")
                certificate = json.loads(table.with_name("certificate.json").read_text())
                require(hashlib.sha256(table.read_bytes()).hexdigest() == certificate["chains_sha256"], str(table) + ": hash differs")
    print("Temporal family keys: chain TSV, prediction arrays and certificate hashes exact")


def verify_denominators():
    counts = Counter()
    with gzip.open(HERE / "denominator-cohort.tsv.gz", "rt") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            cohort = row["cohort"]
            counts[cohort] += 1
            if row["scored"] == "1":
                counts[cohort, "scored"] += 1
                counts[cohort, "duplicate"] += row["duplicate"] == "1"
                for task in ("bp", "mf", "cc", "ec_complete"):
                    counts[cohort, task] += int(row[task]) > 0
    expected = {"temporal-go": 170243, ("temporal-go", "scored"): 131597,
                ("temporal-go", "duplicate"): 28662, ("temporal-go", "bp"): 79923,
                ("temporal-go", "cc"): 122067, ("temporal-go", "mf"): 88690,
                ("deepfri-ec", "ec_complete"): 1876, "deepfri-go": 3416}
    require(all(counts[key] == value for key, value in expected.items()), f"denominator cohort counts differ: {counts}")
    print("Denominators: all cohort IDs, eligibility counts and exclusion flags exact")


def verify_summary_tables(inputs):
    names = ("coverage-rules.tsv", "threshold-audit.tsv", "expanded-adversaries.tsv",
             "parameter-selection.tsv", "simulation-summary.tsv")
    total = 0
    for name in names:
        with (HERE / name).open() as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
        require(bool(rows), name + ": scalar ledger is empty")
        sources = {}
        for row in rows:
            if row["source"] not in sources:
                sources[row["source"]] = json.loads(inputs[row["source"]])
            value = sources[row["source"]]
            for part in row["key"].split("/")[1:]:
                part = part.replace("~1", "/").replace("~0", "~")
                value = value[int(part)] if isinstance(value, list) else value[part]
            require(value == json.loads(row["value"]), name + ": source scalar differs at " + row["key"])
        total += len(rows)
    source = json.loads(inputs["results/raw/paper-intervals.json"])
    with (HERE / "all-subset-scores.tsv").open() as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    expected = {(s, m, t) for s, methods in source["point"].items()
                for m, tasks in methods.items() for t in tasks}
    require({(r["subset"], r["method"], r["task"]) for r in rows} == expected,
            "all-subset score table coverage differs")
    for row in rows:
        s, m, t = (row[k] for k in ("subset", "method", "task"))
        for metric in ("fmax", "aupr"):
            require(float(row[metric]) == source["point"][s][m][t][metric], "subset score differs")
        interval = source["ci95"][s][m][t]["fmax"]
        require([float(row["ci95_low"]), float(row["ci95_high"])] == interval, "subset interval differs")
    print(f"Scalar ledgers: {total} source values and {len(rows)} complete subset score rows exact")


def main():
    inputs = verify_numbers()
    verify_summary_tables(inputs)
    verify_equal_length()
    verify_two_route()
    verify_families()
    verify_denominators()
    reproduce(PACKAGE / "certified_sets/replay.json.gz", PACKAGE / "certified_sets")


if __name__ == "__main__":
    main()
