"""Paired chain bootstrap and clean-set sensitivity for saved predictions.

The held-out labels only score frozen predictions. One bootstrap draw per dataset
is shared by every method and every GO ontology; subset masks follow that draw.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau

from cascade import chain_stats
from review_paths import audit_stem, dataset_dir


TASKS = (("ec", "ec", "EC"), ("go", "bp", "GO-BP"), ("go", "mf", "GO-MF"), ("go", "cc", "GO-CC"))
DEFAULT_PUBLISHED = {"ESM-S 650M": "published-esm-s"}
BASELINES = ("Naive", "BLAST", "DiamondScore")
PAIRS = ("MMseqs2", "Foldseek", "folddisco_motif", "multimodal_consensus_retrieval")


def load_audit(path):
    with path.open() as handle:
        return {row["id"]: row for row in csv.DictReader(handle, delimiter="\t")}


def load_stats(path, ids, terms):
    index = {term: i for i, term in enumerate(terms)}
    pos = {name: i for i, name in enumerate(ids)}
    probs = np.zeros((len(ids), len(terms)), dtype=np.float32)
    truth = np.zeros_like(probs, dtype=np.uint8)
    seen = set()
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            i = pos[row["id"]]
            if i in seen:
                raise ValueError(f"duplicate prediction ID {row['id']} in {path}")
            seen.add(i)
            for term, value in row["scores"].items():
                probs[i, index[term]] = value
            for term in row["true"]:
                truth[i, index[term]] = 1
    if len(seen) != len(ids):
        raise ValueError(f"{path}: {len(seen)} predictions, expected {len(ids)}")
    return chain_stats(probs, truth)


def score(stats, weights):
    precision_ratio, active, recall_ratio = stats
    n = weights.sum()
    active_count = weights @ active
    precision = np.divide(weights @ precision_ratio, active_count,
                          out=np.zeros(precision_ratio.shape[1]), where=active_count > 0)
    recall = (weights @ recall_ratio) / n
    return float(np.max(2 * precision * recall / np.maximum(precision + recall, 1e-8)))


def pct(values):
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def method_sources(rows, dataset, task, name, published):
    if name in published:
        source = published[name]
        if isinstance(source, dict):
            source = source[dataset]
        return f"{source}-{dataset}-{task}"
    if name in BASELINES:
        return f"review-{name.lower()}-{dataset}-{task}"
    return rows[name]["tasks"]["EC" if dataset == "ec" else f"GO-{task.upper()}"]["file"] + f"-{dataset}-{task}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/prepared"))
    parser.add_argument("--audit-root", type=Path, default=Path("results/raw"))
    parser.add_argument("--pred-root", type=Path, default=Path("results/raw"))
    parser.add_argument("--rows-file", type=Path, default=Path("results/raw/hard-set-metrics.json"))
    parser.add_argument("--output", type=Path, default=Path("results/raw/review-intervals.json"))
    parser.add_argument("--replicates", type=int, default=500)
    parser.add_argument("--seed", type=int, default=261011)
    parser.add_argument("--published", action="append", metavar="NAME=PREFIX",
                        help="add a complete same-chain family; use EC_PREFIX|GO_PREFIX when stems differ")
    args = parser.parse_args()
    published = dict(DEFAULT_PUBLISHED)
    for item in args.published or []:
        name, prefix = item.split("=", 1)
        if not name or not prefix:
            raise ValueError(f"invalid --published {item!r}")
        published[name] = ({"ec": prefix.split("|", 1)[0], "go": prefix.split("|", 1)[1]}
                           if "|" in prefix else prefix)
    rows = {row["label"].replace("*", ""): row for row in json.loads(args.rows_file.read_text())["rows"]
            if row["rule"] == "ann" and not row["prefix"].startswith("published-")
            and all(name in row["tasks"] for _, _, name in TASKS)}
    names = list(rows) + list(published) + list(BASELINES)
    audits, ids, subsets, scores = {}, {}, {}, {}
    for dataset in ("ec", "go"):
        audit = load_audit(args.audit_root / f"{audit_stem(args.data_root, dataset)}-chains.tsv")
        report = json.loads((args.audit_root / f"{audit_stem(args.data_root, dataset)}.json").read_text())
        strong = float(report["calibration"]["strong_bits"])
        data = dataset_dir(args.data_root, dataset)
        with (data / "test.jsonl").open() as handle:
            ids[dataset] = [json.loads(line)["id"] for line in handle]
        audits[dataset] = [audit[name] for name in ids[dataset]]
        subsets[dataset] = {
            "full": np.ones(len(ids[dataset]), dtype=np.float32),
            "official_lt30": np.array([row["official_lt30"] == "True" for row in audits[dataset]], dtype=np.float32),
            "hard": np.array([row["hard"] == "True" for row in audits[dataset]], dtype=np.float32),
        }
        for identity in (0.25, 0.30, 0.35):
            for bits_scale in (0.8, 1.0, 1.2):
                key = f"identity<{identity:.2f},bits<{bits_scale:.1f}x"
                subsets[dataset][key] = np.array([
                    float(float(row["max_identity"]) < identity
                          and float(row["top_bits"]) < strong * bits_scale
                          and row["sibling"] != "True" and row["duplicate"] != "True")
                    for row in audits[dataset]], dtype=np.float32)
        classes = json.loads((data / "classes.json").read_text())
        for _, task, label in [item for item in TASKS if item[0] == dataset]:
            for name in names:
                source = method_sources(rows, dataset, task, name, published)
                path = args.pred_root / f"{source}-predictions.jsonl"
                scores[name, label] = load_stats(path, ids[dataset], classes[task])
                print(f"loaded {name} {label}", flush=True)

    primary = ("full", "official_lt30", "hard")
    point = {subset: {name: {} for name in names} for subset in primary}
    sensitivity = {}
    for subset in list(subsets["ec"]):
        cells = {}
        for name in names:
            cells[name] = {label: score(scores[name, label], subsets[dataset][subset])
                           for dataset, _, label in TASKS}
        if subset in primary:
            point[subset] = cells
        else:
            sensitivity[subset] = {"ec_n": int(subsets["ec"][subset].sum()),
                                   "go_n": int(subsets["go"][subset].sum()), "scores": cells}

    def rank_tau(cells):
        ranked = list(rows)
        full = point["full"]
        full_mean = [np.mean(list(full[name].values())) for name in ranked]
        cell_mean = [np.mean(list(cells[name].values())) for name in ranked]
        return float(kendalltau(full_mean, cell_mean).statistic)

    for subset, entry in sensitivity.items():
        entry["tau_vs_full"] = rank_tau(entry["scores"])
        entry["winner"] = max(rows, key=lambda name: np.mean(list(entry["scores"][name].values())))

    rng = np.random.default_rng(args.seed)
    draws = {subset: {name: {label: [] for _, _, label in TASKS} for name in names} for subset in primary}
    tau_draws = {subset: [] for subset in ("official_lt30", "hard")}
    for repeat in range(args.replicates):
        counts = {dataset: np.bincount(rng.integers(0, len(ids[dataset]), len(ids[dataset])),
                                       minlength=len(ids[dataset])).astype(np.float32)
                  for dataset in ("ec", "go")}
        for subset in primary:
            for dataset, _, label in TASKS:
                weights = counts[dataset] * subsets[dataset][subset]
                for name in names:
                    draws[subset][name][label].append(score(scores[name, label], weights))
        for subset in tau_draws:
            ranked = list(rows)
            a = [np.mean([draws["full"][name][label][-1] for _, _, label in TASKS]) for name in ranked]
            b = [np.mean([draws[subset][name][label][-1] for _, _, label in TASKS]) for name in ranked]
            tau_draws[subset].append(float(kendalltau(a, b).statistic))
        if (repeat + 1) % 50 == 0:
            print(f"bootstrap {repeat + 1}/{args.replicates}", flush=True)

    intervals = {subset: {name: {label: pct(draws[subset][name][label]) for _, _, label in TASKS}
                          for name in names} for subset in primary}
    pairs = {}
    for published_name in published:
        pairs[published_name] = {}
        for subset in primary:
            pairs[published_name][subset] = {}
            for name in PAIRS:
                pairs[published_name][subset][name] = {label: {
                    "delta": point[subset][published_name][label] - point[subset][name][label],
                    "ci95": pct(np.array(draws[subset][published_name][label]) - np.array(draws[subset][name][label]))
                } for _, _, label in TASKS}
    result = {"bootstrap": {"replicates": args.replicates, "seed": args.seed,
                            "scheme": "resample full-test chains within EC and GO; same draw for all methods and GO terms; subset by frozen audit mask"},
              "counts": {dataset: {key: int(value.sum()) for key, value in subset.items()}
                         for dataset, subset in subsets.items()},
              "methods": names, "published": published, "point": point, "ci95": intervals,
              "paired_published_minus_retrieval": pairs,
              "paired_esm_s_minus_retrieval": pairs["ESM-S 650M"],
              "tau": {subset: {"point": rank_tau(point[subset]), "ci95": pct(tau_draws[subset])}
                      for subset in tau_draws}, "sensitivity": sensitivity}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(args.output, flush=True)


if __name__ == "__main__":
    main()
