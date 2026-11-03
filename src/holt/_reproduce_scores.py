"""Re-score released predictions with the published Fmax convention and its family-block paired bootstrap."""

from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path

import numpy as np


THRESHOLDS = np.arange(0, 1, .01)
REPLICATES = 2000
SEED = 261019


def sparse_counts(ids, truth, predictions):
    """Per chain and threshold: predicted terms, correct terms; and true terms per chain."""
    predicted = np.zeros((len(ids), len(THRESHOLDS)), dtype=np.int32)
    correct = np.zeros_like(predicted)
    truth_counts = np.empty(len(ids), dtype=np.int32)
    for i, identifier in enumerate(ids):
        labels = set(truth[identifier])
        scores = predictions[identifier]
        ranked = np.sort(np.asarray(list(scores.values()), dtype=np.float32))
        hits = np.sort(np.asarray([scores[term] for term in labels if term in scores], dtype=np.float32))
        predicted[i] = len(ranked) - np.searchsorted(ranked, THRESHOLDS, side="right")
        correct[i] = len(hits) - np.searchsorted(hits, THRESHOLDS, side="right")
        truth_counts[i] = len(labels)
    return predicted, correct, truth_counts


def level_counts(levels, truth):
    """The same counts from thresholds-exceeded levels: a term is predicted at threshold j when its level > j."""
    rows, width = len(levels), len(THRESHOLDS) + 1
    offset = np.arange(rows)[:, None] * width
    histogram = np.bincount((offset + levels).ravel(), minlength=rows * width).reshape(rows, width)
    hit = np.bincount((offset + levels).ravel(), weights=truth.ravel(), minlength=rows * width).reshape(rows, width)
    above = lambda h: np.cumsum(h[:, ::-1], axis=1)[:, ::-1][:, 1:]
    return above(histogram).astype(np.int32), above(hit).astype(np.int32), truth.sum(axis=1).astype(np.int32)


def chain_ratios(counts):
    predicted, correct, truth_counts = counts
    active = predicted > 0
    precision = np.divide(correct, predicted, out=np.zeros(predicted.shape), where=active)
    recall = correct / np.maximum(truth_counts, 1)[:, None]
    return precision, active.astype(float), recall


def fmax(counts):
    precision, active, recall = chain_ratios(counts)
    n_active = active.sum(axis=0)
    precision = np.divide(precision.sum(axis=0), n_active, out=np.zeros(len(THRESHOLDS)), where=n_active > 0)
    recall = recall.mean(axis=0)
    return float(np.max(2 * precision * recall / np.maximum(precision + recall, 1e-8)))


def block_draws(counts_by_method, families):
    """Fmax of each method under the same 2,000 resamples of connected families, threshold reoptimized per draw."""
    groups = sorted(set(families))
    index = np.searchsorted(np.asarray(groups), np.asarray(families))
    rng = np.random.default_rng(SEED)
    weights = np.array([np.bincount(rng.integers(0, len(groups), len(groups)), minlength=len(groups))
                        for _ in range(REPLICATES)], dtype=float)
    denominator = (weights @ np.bincount(index, minlength=len(groups)))[:, None]
    draws = {}
    for method, counts in counts_by_method.items():
        pooled = []
        for ratio in chain_ratios(counts):
            per_family = np.zeros((len(groups), ratio.shape[1]))
            np.add.at(per_family, index, ratio)
            pooled.append(weights @ per_family)
        precision, active, recall = pooled
        precision = np.divide(precision, active, out=np.zeros_like(precision), where=active > 0)
        recall = recall / denominator
        draws[method] = np.max(2 * precision * recall / np.maximum(precision + recall, 1e-8), axis=1)
    return draws


def frozen_tables(sets: Path, cohort):
    with (sets / "score-summary.tsv").open(newline="") as handle:
        fmaxes = {(r["task"], r["tier"], r["alpha"], r["method"]): float(r["fmax"])
                  for r in csv.DictReader(handle, delimiter="\t") if r["cohort"] == cohort}
    with (sets / "paired-differences.tsv").open(newline="") as handle:
        pairs = {(r["task"], r["tier"], r["alpha"], r["comparison"]):
                 (float(r["delta_fmax"]), float(r["ci95_low"]), float(r["ci95_high"]))
                 for r in csv.DictReader(handle, delimiter="\t") if r["cohort"] == cohort}
    return fmaxes, pairs


def check_cell(label, cell, counts, families, fmaxes, pairs):
    """Point Fmax, paired deltas and their 95% family-block intervals against the frozen tables."""
    measured = {}
    for method, method_counts in counts.items():
        measured[method] = fmax(method_counts)
        if abs(measured[method] - fmaxes[(*cell, method)]) > 1e-6:
            raise ValueError(f"{label}/{method}: Fmax {measured[method]} differs from frozen {fmaxes[(*cell, method)]}")
    comparisons = {key[3]: value for key, value in pairs.items() if key[:3] == cell}
    draws = block_draws(counts, families)
    for name, (delta, low, high) in comparisons.items():
        left, right = name.split(" - ")
        if left not in measured or right not in measured:
            raise ValueError(f"{label}/{name}: paired method absent")
        actual = measured[left] - measured[right]
        interval = np.quantile(draws[left] - draws[right], [.025, .975])
        if abs(actual - delta) > 2e-6 or np.max(np.abs(interval - (low, high))) > 2e-6:
            raise ValueError(f"{label}/{name}: {actual:+.6f} [{interval[0]:.6f}, {interval[1]:.6f}] differs from "
                             f"frozen {delta:+.6f} [{low:.6f}, {high:.6f}]")
    return len(measured), len(comparisons)


def score_predictions(sets: Path):
    with gzip.open(sets / "predictions.json.gz", "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload["format"] != 1 or payload["protocol"] != "HoLT":
        raise ValueError("unsupported sparse prediction format")
    totals = {}
    fmaxes, pairs = frozen_tables(sets, "deepfri")
    for task, block in payload["tasks"].items():
        for key, cell in block["cells"].items():
            _, _, _, tier, alpha = key.split("/")
            cell_name = f"{task}-{tier}-a{int(round(float(alpha) * 100)):03d}"
            with (sets / "deepfri" / cell_name / "chains.tsv").open(newline="") as handle:
                certified_ids = {row["id"] for row in csv.DictReader(handle, delimiter="\t")}
            if set(cell["ids"]) != certified_ids:
                raise ValueError(f"{key}: score chain IDs differ from certified set")
            counts = {}
            for method in cell["fmax"]:
                predictions = block["methods"][method]
                missing = set(cell["ids"]) - set(predictions)
                if missing:
                    raise ValueError(f"{key}/{method}: {len(missing)} predictions absent")
                counts[method] = sparse_counts(cell["ids"], block["truth"], predictions)
            n = check_cell(key, (task, tier, alpha), counts, cell["families"], fmaxes, pairs)
            totals["DeepFRI"] = [a + b for a, b in zip(totals.get("DeepFRI", (0, 0)), n)]
            print(f"Paired Fmax DeepFRI {task.upper()} {tier} alpha={alpha}: "
                  f"{n[0]} methods, {n[1]} paired deltas and 95% intervals exact")
    fmaxes, pairs = frozen_tables(sets, "temporal")
    with np.load(sets / "temporal-scores.npz") as packed:
        for task, tier, alpha in sorted({key[:3] for key in fmaxes}):
            ids = packed[f"{task}/ids"]
            truth = np.unpackbits(packed[f"{task}/truth"], axis=1, count=len(packed[f"{task}/terms"]))
            rows = packed[f"{task}/cell/{tier}/{alpha}"]
            with (sets / "temporal" / f"{task}-{tier}-a{int(round(float(alpha) * 100)):03d}"
                  / "chains.tsv").open(newline="") as handle:
                certified_ids = {row["id"] for row in csv.DictReader(handle, delimiter="\t")}
            if set(ids[rows].tolist()) != certified_ids:
                raise ValueError(f"post-2020 {task}/{tier}/{alpha}: score chain IDs differ from certified set")
            counts = {method: level_counts(packed[f"{task}/levels/{method}"][rows], truth[rows])
                      for (t, ti, a, method) in fmaxes if (t, ti, a) == (task, tier, alpha)}
            n = check_cell(f"post-2020 {task}/{tier}/{alpha}", (task, tier, alpha), counts,
                           packed[f"{task}/families"][rows].tolist(), fmaxes, pairs)
            totals["post-2020"] = [a + b for a, b in zip(totals.get("post-2020", (0, 0)), n)]
            print(f"Paired Fmax post-2020 {task.upper()} {tier} alpha={alpha}: {len(rows)} chains, "
                  f"{n[0]} methods, {n[1]} paired deltas and 95% intervals exact")
    for cohort, (nmetrics, npairs) in totals.items():
        print(f"Paired scores {cohort}: {nmetrics} Fmax, {npairs} deltas and {npairs} family-block "
              f"95% intervals ({REPLICATES:,} draws, seed {SEED}) exact")
