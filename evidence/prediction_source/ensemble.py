"""Blend retrieval routes, and optionally same-entry sibling chains, with weights picked on valid.

Each route votes with its own valid-selected k, power and score column (from
results/raw/<route>-<dataset>-metrics.json). Every subset of routes gets the
simplex weights with the best valid Fmax; the test set is scored once per subset.
Sibling chains: train chains from the same PDB entry as the query (the part of
the chain name before "-"). That uses the benchmark's naming, so it is reported
as its own row, never as the main result.
"""
import argparse
import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from evaluate import records, task_hits, truth, vote
from metrics import fmax, fmax_fast, micro_aupr

BETAS = (0, 0.1, 0.25, 0.5, 1, 2, 4)


def simplex(n, step):
    units = round(1 / step)
    for cut in itertools.product(range(units + 1), repeat=n - 1):
        if sum(cut) <= units:
            yield np.array(cut + (units - sum(cut),)) / units


def pick_route(spec, dataset, task):
    """a/b/... names variants of one route; the one with the best valid Fmax for this label set wins."""
    return max(spec.split("/"), key=lambda route: json.loads(
        Path(f"results/raw/{route}-{dataset}-metrics.json").read_text())[task]["valid_fmax"])


def route_probs(route, dataset, task, splits, train, class_index):
    chosen = json.loads(Path(f"results/raw/{route}-{dataset}-metrics.json").read_text())[task]
    labels = {row["id"]: row["labels"][task] for row in train}
    out = {}
    for split, rows in splits.items():
        found = task_hits(Path(f"results/raw/{chosen.get('hits', route)}-{dataset}-{split}-hits.tsv"), train, task,
                          chosen["score_column"], chosen.get("annotated_only", False))
        out[split], _ = vote(rows, labels, class_index, found, chosen["k"], chosen["weight_power"])
    return out, chosen["valid_fmax"]


def siblings(rows, by_entry, class_index, task, annotated=False):
    probs = np.zeros((len(rows), len(class_index)), dtype=np.float32)
    for i, row in enumerate(rows):
        mates = [m for m in by_entry.get(row["id"].split("-")[0], []) if m["labels"][task] or not annotated]
        for mate in mates:
            for term in mate["labels"][task]:
                probs[i, class_index[term]] += 1 / len(mates)
    return probs


def save(name, dataset, task, rows, terms, probs):
    with Path(f"results/raw/{name}-{dataset}-{task}-predictions.jsonl").open("w") as file:
        for row, scores in zip(rows, probs):
            file.write(json.dumps({"id": row["id"], "true": row["labels"][task],
                                   "scores": {terms[i]: round(float(scores[i]), 6) for i in np.flatnonzero(scores)}}) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=("ec", "go"))
    parser.add_argument("--routes", nargs="+", default=["mmseqs", "foldseek"],
                        help="route names; a/b means pick a or b per task by valid Fmax")
    parser.add_argument("--extra", nargs="*", default=[], help="routes allowed only in the all-routes blend")
    parser.add_argument("--name", default="ens")
    parser.add_argument("--data-root", type=Path, default=Path("data/prepared"))
    parser.add_argument("--step", type=float, default=0.1)
    parser.add_argument("--annotated", action="store_true",
                        help="select on valid chains annotated for the label set; siblings only from annotated chains")
    args = parser.parse_args()
    data = args.data_root / args.dataset
    train, valid, test = (records(data / f"{s}.jsonl") for s in ("train", "valid", "test"))
    by_entry = defaultdict(list)
    for row in train:
        by_entry[row["id"].split("-")[0]].append(row)
    output = {}
    for task, terms in json.loads((data / "classes.json").read_text()).items():
        class_index = {t: i for i, t in enumerate(terms)}
        selection = [row for row in valid if row["labels"][task] or not args.annotated]
        yv, yt = truth(selection, class_index, task), truth(test, class_index, task)
        probs, picked = {}, {}
        for spec in args.routes + args.extra:
            short, route = spec.split("/")[0], pick_route(spec, args.dataset, task)
            picked[short] = route
            probs[short], _ = route_probs(route, args.dataset, task, {"valid": selection, "test": test}, train, class_index)
        base = [r.split("/")[0] for r in args.routes]
        subsets = [c for n in range(2, len(base) + 1) for c in itertools.combinations(base, n)]
        if args.extra:
            subsets.append(tuple(base) + tuple(r.split("/")[0] for r in args.extra))
        task_out = {"picked_routes": picked, "blends": {}}
        mates_v = siblings(selection, by_entry, class_index, task, args.annotated)
        mates_t = siblings(test, by_entry, class_index, task, args.annotated)
        task_out["chains_with_siblings"] = {"valid": int((mates_v.sum(1) > 0).sum()), "test": int((mates_t.sum(1) > 0).sum()),
                                            "valid_n": len(selection), "test_n": len(test)}
        for subset in subsets:
            stack_v = np.stack([probs[r]["valid"] for r in subset])
            weights = max(simplex(len(subset), args.step),
                          key=lambda w: fmax_fast(np.tensordot(w, stack_v, 1), yv))
            ev = np.tensordot(weights, stack_v, 1)
            et = np.tensordot(weights, np.stack([probs[r]["test"] for r in subset]), 1)
            name = f"{args.name}-" + "+".join(subset)
            entry = {"weights": dict(zip(subset, map(float, weights))), "valid_fmax": fmax_fast(ev, yv),
                     "test_fmax": fmax(et, yt), "test_aupr_micro": micro_aupr(et, yt)}
            save(name, args.dataset, task, test, terms, et)
            task_out["blends"][name] = entry
            print(f"{args.dataset}-{task} {name} w={entry['weights']} valid {entry['valid_fmax']:.4f} "
                  f"test {entry['test_fmax']:.4f} AUPR {entry['test_aupr_micro']:.4f}", flush=True)
            if subset == subsets[-1]:
                beta = max(BETAS, key=lambda b: fmax_fast((ev + b * mates_v) / (1 + b), yv))
                st = (et + beta * mates_t) / (1 + beta)
                sib = {"weights": entry["weights"], "beta": beta,
                       "valid_fmax": fmax_fast((ev + beta * mates_v) / (1 + beta), yv),
                       "test_fmax": fmax(st, yt), "test_aupr_micro": micro_aupr(st, yt)}
                save(name + "+siblings", args.dataset, task, test, terms, st)
                task_out["blends"][name + "+siblings"] = sib
                print(f"{args.dataset}-{task} {name}+siblings beta={beta} valid {sib['valid_fmax']:.4f} "
                      f"test {sib['test_fmax']:.4f} AUPR {sib['test_aupr_micro']:.4f}", flush=True)
        output[task] = task_out
    Path(f"results/raw/{args.name}-{args.dataset}-metrics.json").write_text(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
