"""CPU reproduction from released role-separated chain features and predictions."""

from __future__ import annotations

import csv
import gzip
import json
import math
import time
from pathlib import Path

import numpy as np

from . import _certify as cert
from ._common import cp_upper
from ._engine import features


def _rows(packed, task):
    return [{"id": r[0], "pdb": r[1], "family": r[2], "length": r[3],
             "identity": r[4], "bits": r[5], "esm2": r[6], "esmc": r[7],
             f"agree_{task}": r[8], f"consensus_{task}": r[9],
             f"vote_size_{task}": r[10], "event": r[11]} for r in packed]


def _scores(rows, task, model):
    x = np.asarray([features(r, task) for r in rows], dtype=float)
    if not len(x):
        return np.array([], dtype=float)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        scores = ((x - np.asarray(model["mean"])) / np.asarray(model["scale"])) @ np.asarray(model["coef"]) + model["intercept"]
    if not np.all(np.isfinite(scores)):
        raise ValueError("reproduction produced a non-finite score")
    return scores


def _cell(raw, alpha):
    task = raw["task"]
    design, calibration, held = (_rows(raw[role], task)
                                 for role in ("design", "calibration", "test"))
    model = cert.learn(design, task, "event")
    frozen = raw["model"]
    if max(abs(a - b) for a, b in zip(model["coef"], frozen["coef"])) > 1e-6 or \
            abs(model["intercept"] - frozen["intercept"]) > 1e-6:
        raise ValueError("training fit differs from the frozen model")
    design_scores = _scores(design, task, model)
    calibration_scores = _scores(calibration, task, model)
    held_scores = _scores(held, task, model)
    trace = []
    events = np.asarray([r["event"] for r in calibration], dtype=np.int64)
    for q in cert.Q:
        cut = float(np.percentile(design_scores, q))
        if q == 100:
            cut = math.nextafter(cut, math.inf)
        inside = calibration_scores < cut
        n, k = int(inside.sum()), int(events[inside].sum())
        trace.append({"q": q, "cut": [cut], "n": n, "leaks": k,
                      "ucb": cp_upper(k, n, .025)})
    index = cert.fixed_sequence(trace, alpha, .025)
    if index is None:
        return [], None, trace, model
    picked = [(row["id"], row["event"]) for row, score in zip(held, held_scores)
              if score < trace[index]["cut"][0]]
    return picked, index, trace, model


def _expected(sets, task, tier, alpha):
    name = f"{task}-{tier}-a{int(round(alpha * 100)):03d}"
    base = sets / "deepfri" / name
    path = base / "chains.tsv"
    with path.open(newline="") as handle:
        ids = {row["id"] for row in csv.DictReader(handle, delimiter="\t")}
    return ids, json.loads((base / "certificate.json").read_text())


def reproduce(data: Path, sets: Path) -> int:
    started = time.perf_counter()
    with gzip.open(data, "rt", encoding="utf-8") as handle:
        source = json.load(handle)
    if source["format"] != 1 or source["protocol"] != "HoLT" or source["gamma"] != .025:
        raise ValueError("unsupported bundled replay format or protocol")
    total = 0
    for name, cell in source["cells"].items():
        alphas = (.1, .15) if cell["task"] == "mf" else (.1,)
        for alpha in alphas:
            chosen, index, trace, _ = _cell(cell, alpha)
            actual = {name for name, _ in chosen}
            expected, certificate = _expected(sets, cell["task"], cell["tier"], alpha)
            if actual != expected:
                raise ValueError(f"{name} alpha={alpha}: membership mismatch: "
                                 f"missing={len(expected - actual)} extra={len(actual - expected)}")
            selected = trace[index] if index is not None else None
            checks = {
                "protocol": "HoLT", "task": cell["task"], "tier": cell["tier"],
                "alpha": alpha, "delta": .1, "gamma": .025,
                "certified_chains": len(actual), "calibration_representatives": len(cell["calibration"]),
                "selected_index": index,
                "selected_calibration_n": selected["n"] if selected else None,
                "selected_calibration_events": selected["leaks"] if selected else None,
            }
            for key, value in checks.items():
                if certificate[key] != value:
                    raise ValueError(f"{name} alpha={alpha}: certificate {key} differs")
            if selected and (not math.isclose(certificate["selected_cut"], selected["cut"][0], abs_tol=1e-8)
                             or not math.isclose(certificate["binomial_upper_bound"], selected["ucb"], abs_tol=1e-8)):
                raise ValueError(f"{name} alpha={alpha}: certificate cut or binomial bound differs")
            total += 1
            print(f"DeepFRI {cell['task'].upper()} {cell['tier']} alpha={alpha:g}: "
                  f"{len(actual)} chains; exact IDs; calibration "
                  f"{trace[index]['leaks']}/{trace[index]['n']}" if index is not None else
                  f"DeepFRI {cell['task'].upper()} {cell['tier']} alpha={alpha:g}: "
                  "0 chains; exact IDs; abstain")
    print(f"Certified lists: {total}/{total} exact")
    from ._reproduce_scores import score_predictions

    score_predictions(sets / "predictions.json.gz", sets / "score-summary.tsv")
    print(f"Wall time: {time.perf_counter() - started:.3f}s")
    return 0
