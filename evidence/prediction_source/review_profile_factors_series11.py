"""Frozen-representative factorial audit of phmmer and transfer-vote rules.

The three variants use the same profile-era representative design and the
identity<30% guard. Only attacker routes/rules change. Calibration refits the
score and its design-quantile path at delta/4 for each variant.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import resource
import time
from pathlib import Path

import numpy as np

import crossfit_certify as cc
import crossfit_original
from crossfit_engine import ADVERSARIES, BASE_RULES, RULES, Pool, Route
from review_transfer_repair import fit

RAW = Path("results/raw")
VARIANTS = {
    "phmmer_only": (ADVERSARIES["profile"], BASE_RULES),
    "vote_only": (ADVERSARIES["homology"], RULES["profile"]),
    "both": (ADVERSARIES["profile"], RULES["profile"]),
}
DATASETS = {"ec": ("ec",), "orig-go": ("bp", "mf", "cc")}


def stem(dataset, tier, feature_source="series9"):
    if feature_source == "fresh" and tier != "mmseqs30":
        raise ValueError("PDB tier already regenerates task-specific features")
    return f"profile-factors-series11-{dataset}" + ("" if tier == "mmseqs30" else "-pdb") + ("-fresh" if feature_source == "fresh" else "")


def install_pdb_profile_route(dataset):
    path = RAW / f"profile-factors-series11-pdb-{dataset}/full/profile-pdb-train-hits.tsv"
    report = json.loads(path.with_name("summary.json").read_text())
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != report["output_sha256"] or report["query_count"] != report["reference_count"]:
        raise ValueError(f"incomplete PDB phmmer self-search: {path}")
    previous = Pool._extra
    def patched(self, route):
        if route != "profile":
            return previous(self, route)
        q, t, scores = [], [], []
        with path.open() as stream:
            for line in stream:
                query, target, value = line.rstrip("\n").split("\t")
                a, b = self.index[query], self.index[target]
                if self.pdb[a] == self.pdb[b]:
                    raise ValueError("same-PDB hit in PDB-excluded search")
                q.append(a); t.append(b); scores.append(float(value))
        score = np.array(scores, np.float64)
        return Route(np.array(q, np.int32), np.array(t, np.int32), score, score, self.name_rank)
    Pool._extra = patched
    return {"path": str(path), "sha256": digest}


def role_rows(dataset):
    path = RAW / f"crossfit-{dataset}-profile-mmseqs30-full-chains.tsv"
    with path.open() as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    for row in rows:
        for key, value in list(row.items()):
            if value in ("True", "False"):
                row[key] = value == "True"
            elif key not in ("role", "id", "pdb", "family", "cluster") and value:
                try:
                    row[key] = float(value)
                except ValueError:
                    pass
    return rows


def smoke(dataset):
    if dataset == "orig-go":
        crossfit_original.adapt("go")
    pool = Pool(dataset, extra=("profile",))
    rows = role_rows(dataset)
    pool.rules = RULES["profile"]
    checked = {}
    for task in DATASETS[dataset]:
        selected = [r for r in rows if r[f"rep_{task}"] is True][:10]
        query = np.array([pool.index[r["id"]] for r in selected])
        observed = pool.rows(query, np.ones(len(pool.ids), bool), pool.family,
                             None, ADVERSARIES["profile"], (task,))
        key = "leak_ec_exact" if task == "ec" else f"leak_{task}_raw"
        for old, new in zip(selected, observed):
            if old["id"] != new["id"] or bool(old[key]) != bool(new[key]):
                raise ValueError(f"combined-factor smoke mismatch {dataset}/{task}/{old['id']}")
        checked[task] = len(selected)
    print(json.dumps({"dataset": dataset, "combined_events_matched": checked,
                      "max_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}), flush=True)


def calibrate(dataset, output, tier="mmseqs30", feature_source="series9"):
    if dataset == "orig-go":
        crossfit_original.adapt("go")
    profile_input = install_pdb_profile_route(dataset) if tier == "pdb" else None
    started = time.monotonic()
    pool = Pool(dataset, extra=("profile",))
    reps = cc.Representatives(pool, tiers=(tier,))
    saved_rows = role_rows(dataset) if tier == "mmseqs30" and feature_source == "series9" else None
    saved_by_role = {(r["role"], r["id"]): r for r in saved_rows} if saved_rows else {}
    saved = {task: {r["id"] for r in saved_rows if r[f"rep_{task}"] is True}
             for task in DATASETS[dataset]} if saved_rows else {
             task: {pool.ids[i] for i in reps.representatives[tier][task]} for task in DATASETS[dataset]}
    if saved_rows:
        for task in DATASETS[dataset]:
            current = {pool.ids[i] for i in reps.representatives[tier][task]}
            if current != saved[task]:
                raise ValueError(f"frozen representative set changed: {dataset}/{task}: {len(current)} vs {len(saved[task])}")
    result = {"dataset": dataset, "tier": tier, "feature_source": feature_source,
              "alpha": .1, "delta": cc.DELTA,
              "gamma": cc.PER_TEST["pooled"], "guard": "identity<0.3",
              "profile_input": profile_input,
              "representatives": {k: len(v) for k, v in saved.items()},
              "role_feature_audit": {}, "variants": {}, "seconds": None, "max_rss_bytes": None}
    for name, (attackers, rules) in VARIANTS.items():
        pool.rules = rules
        design, calibration, _ = reps.rows(tier, None, attackers)
        cells = {}
        for task in DATASETS[dataset]:
            event = "exact" if task == "ec" else "raw"
            leak_key = f"leak_{task}_{event}"
            if saved_rows:
                fresh_design = {r["id"]: r for r in design[task]}
                fresh_cal = {r["id"]: r for r in calibration[task]}
                anchored_design = [r.copy() for r in saved_rows if r["role"] == "design" and r[f"labels_{task}"]]
                anchored_cal = [r.copy() for r in saved_rows if r["role"] == "calibration" and r[f"rep_{task}"]]
                if set(fresh_design) != {r["id"] for r in anchored_design} or set(fresh_cal) != {r["id"] for r in anchored_cal}:
                    raise ValueError(f"frozen role membership differs {dataset}/{task}")
                if name == "phmmer_only":
                    result["role_feature_audit"][task] = {
                        "saved_design_guard_n": int(sum(r["identity"] < .3 for r in anchored_design)),
                        "task_specific_design_guard_n": int(sum(r["identity"] < .3 for r in fresh_design.values())),
                        "saved_calibration_guard_n": int(sum(r["identity"] < .3 for r in anchored_cal)),
                        "task_specific_calibration_guard_n": int(sum(r["identity"] < .3 for r in fresh_cal.values())),
                        "design_identity_different_n": int(sum(abs(r["identity"] - fresh_design[r["id"]]["identity"]) > 1e-12
                                                               for r in anchored_design)),
                        "calibration_identity_different_n": int(sum(abs(r["identity"] - fresh_cal[r["id"]]["identity"]) > 1e-12
                                                                    for r in anchored_cal)),
                        "design_feature_different_n": {key: int(sum(abs(r[key] - fresh_design[r["id"]][key]) > 1e-12
                                                                     for r in anchored_design))
                                                       for key in ("identity", "bits", "esm2", "esmc")},
                        "calibration_feature_different_n": {key: int(sum(abs(r[key] - fresh_cal[r["id"]][key]) > 1e-12
                                                                          for r in anchored_cal))
                                                            for key in ("identity", "bits", "esm2", "esmc")}}
                for row in anchored_design:
                    row[leak_key] = fresh_design[row["id"]][leak_key]
                for row in anchored_cal:
                    row[leak_key] = fresh_cal[row["id"]][leak_key]
                if name == "both" and any(r[leak_key] != saved_by_role[(r["role"], r["id"])][leak_key]
                                          for r in anchored_design + anchored_cal):
                    raise ValueError(f"combined-factor training events differ from Series9 {dataset}/{task}")
            else:
                anchored_design, anchored_cal = design[task], calibration[task]
            rule = fit(anchored_design, anchored_cal, task, event, "identity30")
            if "unavailable" in rule:
                raise ValueError(f"unavailable score fit {dataset}/{name}/{task}: {rule['unavailable']}")
            i = cc.fixed_sequence(rule["trace"], .1, cc.PER_TEST["pooled"])
            cells[task] = {"design_n": rule["design_n"], "calibration_n": rule["calibration_n"],
                           "selected_index": i, "selected_q": cc.Q[i] if i is not None else None,
                           "selected_cut": rule["trace"][i]["cut"][0] if i is not None else None,
                           "trace": rule["trace"], "model": rule["model"]}
            print(dataset, name, task, i, rule["calibration_n"], flush=True)
        result["variants"][name] = {"routes": list(attackers), "rules": list(rules), "cells": cells}
    result["seconds"] = time.monotonic() - started
    result["max_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if saved_rows:
        original = json.loads((RAW / "guarded-series9" / ("profile-ec.json" if dataset == "ec" else "profile-go.json")).read_text())
        for task in DATASETS[dataset]:
            a = result["variants"]["both"]["cells"][task]
            b = original["cells"][task]
            if (a["selected_index"] != b["selected_index"] or a["calibration_n"] != b["calibration_n"]
                    or a["design_n"] != b["design_n"]):
                raise ValueError(f"combined-factor recertification differs from Series9 {dataset}/{task}")
            for left, right in zip(a["trace"], b["trace"]):
                if (left["n"], left["leaks"]) != (right["n"], right["leaks"]):
                    raise ValueError(f"combined-factor trace differs from Series9 {dataset}/{task}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(output, result["seconds"], result["max_rss_bytes"], flush=True)


def select(dataset, output, tier="mmseqs30", feature_source="series9"):
    frozen = json.loads((RAW / f"{stem(dataset, tier, feature_source)}.json").read_text())
    if dataset == "ec":
        from certify_four import build_rows, ec4
        from crossfit_retrieve import DATA
        rows = build_rows(DATA["ec"], "ec", "test")
        # build_rows carries the frozen test annotations only for eligibility;
        # all feature values used below are label-free.
        true = {r["id"]: r["labels"]["ec"] for r in cc.records(DATA["ec"] / "test.jsonl")}
    else:
        rows = [json.loads(line) for line in (RAW / "crossfit-repair-s13-go-evaluated-rows.jsonl").open()]
    output_rows = {}
    for name, payload in frozen["variants"].items():
        output_rows[name] = {}
        for task, rule in payload["cells"].items():
            cut = rule["selected_cut"]
            if cut is None:
                chosen = []
            else:
                chosen = [r for r in rows if r[f"labels_{task}"] and
                          (task != "ec" or bool(ec4(true[r["id"]]))) and
                          not r["sibling"] and not r["duplicate"] and r["identity"] < .3 and
                          cc.copy_score(r, task, rule["model"]) < cut]
            output_rows[name][task] = {"n": len(chosen), "ids": sorted(r["id"] for r in chosen)}
            print(dataset, name, task, len(chosen), flush=True)
    result = {"dataset": dataset, "tier": tier, "feature_source": feature_source,
              "training": str(RAW / f"{stem(dataset, tier, feature_source)}.json"),
              "test_features": ("data/seqfix/ec/test.jsonl and frozen search hits" if dataset == "ec" else
                                "results/raw/crossfit-repair-s13-go-evaluated-rows.jsonl"),
              "rule": "eligible, no sibling or duplicate, identity<0.3, frozen logistic score below selected cut",
              "variants": output_rows}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(output, flush=True)


def score(dataset, output, tier="mmseqs30", feature_source="series9"):
    from collections import defaultdict
    from certify_four import ec4, hit_panel
    from certify_split import f1, records
    from crossfit_accept import held_families
    from crossfit_engine import transfer
    from crossfit_retrieve import DATA

    if dataset == "orig-go":
        crossfit_original.adapt("go")
        data = Path("data/prepared/deepfri/go")
        paths = crossfit_original.paths("orig-go", "test")
        paths["profile"] = (RAW / ("profile-factors-series11-phmmer-fresh-test-hits.tsv" if feature_source == "fresh"
                                     else "profile-factors-series11-phmmer-test-hits.tsv") if tier == "mmseqs30"
                            else RAW / "profile-factors-series11-pdb-go-test-hits.tsv")
        report = json.loads(paths["profile"].with_suffix(".json").read_text())
    else:
        data = DATA["ec"]
        from certify_split import hit_files
        paths = hit_files(RAW, "old", "ec", "test")
        paths["profile"] = RAW / "profile-ec-test-hits.tsv"
        report = json.loads((RAW / "profile-ec-test-timing.json").read_text())
        if report["queries"] != len(records(data / "test.jsonl")):
            raise ValueError("EC profile test search did not cover entire cohort")
    selected = json.loads((RAW / f"{stem(dataset, tier, feature_source)}-selected.json").read_text())
    queries = set().union(*(set(cell["ids"]) for tasks in selected["variants"].values() for cell in tasks.values()))
    if dataset == "orig-go":
        needed = set().union(*(set(cell["ids"]) for variant in ("phmmer_only", "both")
                               for cell in selected["variants"][variant].values()))
        if set(report["query_ids"]) != needed:
            raise ValueError("phmmer test query coverage differs from frozen selection")
        with paths["profile"].open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != report["output_sha256"]:
                raise ValueError("phmmer test hit SHA differs from query receipt")
        selected_path = RAW / f"{stem(dataset, tier, feature_source)}-selected.json"
        with selected_path.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != report["selected_sha256"]:
                raise ValueError("phmmer selected SHA differs from query receipt")
        if tier == "pdb":
            if report.get("filter") != "exclude same-PDB hits before top 300":
                raise ValueError("PDB test search did not attest same-PDB exclusion")
            with paths["profile"].open() as stream:
                if any(q.split("-", 1)[0] == t.split("-", 1)[0]
                       for q, t, _ in (line.rstrip("\n").split("\t") for line in stream)):
                    raise ValueError("same-PDB hit in PDB test search")
    train = {r["id"]: r["labels"] for r in records(data / "train.jsonl")}
    truth = {r["id"]: r["labels"] for r in records(data / "test.jsonl") if r["id"] in queries}
    if set(truth) != queries:
        raise ValueError("selected IDs outside test set")
    panels = {route: hit_panel(path, route, train, truth, DATASETS[dataset])[1]
              for route, path in paths.items() if route in ("mmseqs", "foldseek", "profile")}
    sibling_labels = defaultdict(list)
    for name, labels in train.items():
        sibling_labels[name.split("-", 1)[0]].append(labels)
    family = held_families(dataset, "test")
    variants = {}
    ledger = []
    for name, tasks in selected["variants"].items():
        routes, rules = VARIANTS[name]
        variants[name] = {}
        for task, cell in tasks.items():
            answered = 0
            for query in cell["ids"]:
                true = truth[query][task]
                predictions = []
                for route in routes:
                    hits = panels[route][task][query]
                    if hits:
                        ranked = [frozenset(train[target][task]) for _, target in hits]
                        weights = [value for value, _ in hits]
                        predictions.extend(transfer(ranked, weights, rule) for rule in rules)
                predictions.extend(labels[task] for labels in sibling_labels.get(query.split("-", 1)[0], ())
                                   if labels[task])
                if task == "ec":
                    leak = bool(ec4(true)) and any(ec4(pred) & ec4(true) for pred in predictions)
                else:
                    leak = bool(true) and any(f1(pred, true) >= .5 for pred in predictions)
                answered += bool(leak)
                ledger.append({"variant": name, "task": task, "id": query, "family": family[query],
                               "answered": bool(leak)})
            variants[name][task] = {"n": cell["n"], "answered": answered,
                                    "families": len({family[q] for q in cell["ids"]}),
                                    "rate": answered / cell["n"] if cell["n"] else None}
            print(dataset, name, task, cell["n"], answered, flush=True)
    result = {"dataset": dataset, "tier": tier, "feature_source": feature_source,
              "selected": str(RAW / f"{stem(dataset, tier, feature_source)}-selected.json"),
              "test_hits": {k: str(v) for k, v in paths.items() if k in ("mmseqs", "foldseek", "profile")},
              "phmmer_query_receipt": report, "variants": variants}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    output.with_name(output.stem + "-chains.jsonl").write_text("".join(json.dumps(r) + "\n" for r in ledger))
    print(output, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("smoke", "calibrate", "select", "score"))
    parser.add_argument("--dataset", choices=tuple(DATASETS), required=True)
    parser.add_argument("--tier", choices=("mmseqs30", "pdb"), default="mmseqs30")
    parser.add_argument("--feature-source", choices=("series9", "fresh"), default="series9")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.feature_source == "fresh" and args.tier != "mmseqs30":
        parser.error("fresh feature source is only defined for the family tier")
    if args.action == "smoke":
        smoke(args.dataset)
    elif args.action == "calibrate":
        output = args.output or RAW / f"{stem(args.dataset, args.tier, args.feature_source)}.json"
        calibrate(args.dataset, output, args.tier, args.feature_source)
    elif args.action == "select":
        output = args.output or RAW / f"{stem(args.dataset, args.tier, args.feature_source)}-selected.json"
        select(args.dataset, output, args.tier, args.feature_source)
    else:
        output = args.output or RAW / f"{stem(args.dataset, args.tier, args.feature_source)}-scored.json"
        score(args.dataset, output, args.tier, args.feature_source)


if __name__ == "__main__":
    main()
