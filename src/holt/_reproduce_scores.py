"""Re-score released sparse predictions with the published Fmax convention."""

from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path

import numpy as np


THRESHOLDS = np.arange(0, 1, .01)


def fmax_sparse(ids, truth, predictions):
    count = len(ids)
    predicted = np.zeros((count, len(THRESHOLDS)), dtype=np.int32)
    correct = np.zeros_like(predicted)
    truth_counts = np.empty(count, dtype=np.int32)
    for i, identifier in enumerate(ids):
        labels = set(truth[identifier])
        scores = predictions[identifier]
        ranked = np.sort(np.asarray(list(scores.values()), dtype=np.float32))
        hits = np.sort(np.asarray([scores[term] for term in labels if term in scores], dtype=np.float32))
        predicted[i] = len(ranked) - np.searchsorted(ranked, THRESHOLDS, side="right")
        correct[i] = len(hits) - np.searchsorted(hits, THRESHOLDS, side="right")
        truth_counts[i] = len(labels)
    active = predicted > 0
    precision = np.divide(correct, predicted, out=np.zeros_like(correct, dtype=float), where=active)
    precision = np.divide(precision.sum(axis=0), active.sum(axis=0),
                          out=np.zeros(len(THRESHOLDS)), where=active.sum(axis=0) > 0)
    recall = np.divide(correct, truth_counts[:, None],
                       out=np.zeros_like(correct, dtype=float), where=truth_counts[:, None] > 0).mean(axis=0)
    return float(np.max(2 * precision * recall / np.maximum(precision + recall, 1e-8)))


def score_predictions(path: Path, summary_path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload["format"] != 1 or payload["protocol"] != "HoLT":
        raise ValueError("unsupported sparse prediction format")
    with summary_path.open(newline="") as handle:
        summary = {(r["task"], r["tier"], r["alpha"], r["method"]): float(r["fmax"])
                   for r in csv.DictReader(handle, delimiter="\t") if r["cohort"] == "deepfri"}
    nmetrics = npairs = 0
    for task, block in payload["tasks"].items():
        for key, cell in block["cells"].items():
            _, _, _, tier, alpha = key.split("/")
            cell_name = f"{task}-{tier}-a{int(round(float(alpha) * 100)):03d}"
            chains_path = summary_path.parent / "deepfri" / cell_name / "chains.tsv"
            with chains_path.open(newline="") as handle:
                certified_ids = {row["id"] for row in csv.DictReader(handle, delimiter="\t")}
            if set(cell["ids"]) != certified_ids:
                raise ValueError(f"{key}: score chain IDs differ from certified set")
            measured = {}
            for method, expected in cell["fmax"].items():
                predictions = block["methods"][method]
                missing = set(cell["ids"]) - set(predictions)
                if missing:
                    raise ValueError(f"{key}/{method}: {len(missing)} predictions absent")
                actual = fmax_sparse(cell["ids"], block["truth"], predictions)
                if abs(actual - expected) > 1e-6 or abs(actual - summary[(task, tier, alpha, method)]) > 1e-6:
                    raise ValueError(f"{key}/{method}: Fmax {actual} differs from frozen {expected}")
                measured[method] = actual
                nmetrics += 1
            for comparison, expected in cell["paired"].items():
                left, right = comparison.split(" - ")
                if left not in measured or right not in measured:
                    raise ValueError(f"{key}/{comparison}: paired method absent")
                actual = measured[left] - measured[right]
                if abs(actual - expected) > 2e-6:
                    raise ValueError(f"{key}/{comparison}: paired delta {actual} differs from frozen {expected}")
                npairs += 1
            print(f"Paired Fmax {task.upper()} {tier} alpha={alpha}: "
                  f"{len(measured)} methods, {len(cell['paired'])} paired deltas exact")
    print(f"Paired scores: {nmetrics} Fmax and {npairs} deltas exact")
