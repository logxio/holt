"""Learned inference layer over retrieval neighbours, trained only on the validation split.

For every (query, candidate term) it builds features from each route's neighbour
list: the route's own vote (valid-selected k and power), the best relative score
and the best rank among neighbours carrying the term, the share of the top 10
carrying it, and the query's absolute top score; plus the term's train
frequency. Candidates are the terms any top-50 neighbour carries plus the 20
most frequent train terms. A gradient-boosted
classifier picks its settings by 5-fold query-grouped cross-validation Fmax on
valid, is refit on all of valid, and scores the test set once.
"""
import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold

from ensemble import pick_route, siblings
from evaluate import records, task_hits, truth, vote
from metrics import fmax, fmax_fast, micro_aupr

DEPTH = 50
FREQUENT = 20
CONFIGS = ({"learning_rate": 0.1, "max_iter": 200, "max_leaf_nodes": 31},
           {"learning_rate": 0.05, "max_iter": 400, "max_leaf_nodes": 31},
           {"learning_rate": 0.05, "max_iter": 400, "max_leaf_nodes": 63, "l2_regularization": 1.0})


def route_features(route, dataset, task, rows, split, train, class_index):
    chosen = json.loads(Path(f"results/raw/{route}-{dataset}-metrics.json").read_text())[task]
    labels = {row["id"]: row["labels"][task] for row in train}
    found = task_hits(Path(f"results/raw/{chosen.get('hits', route)}-{dataset}-{split}-hits.tsv"), train, task,
                      chosen["score_column"], chosen.get("annotated_only", False))
    probs, _ = vote(rows, labels, class_index, found, chosen["k"], chosen["weight_power"])
    shape = (len(rows), len(class_index))
    best_score, best_rank, top10 = (np.zeros(shape, dtype=np.float32) for _ in range(3))
    top = np.zeros(len(rows), dtype=np.float32)
    for i, row in enumerate(rows):
        neighbors = found.get(row["id"], [])[:DEPTH]
        if not neighbors:
            continue
        top[i] = neighbors[0][1]
        for rank, (target, score) in enumerate(neighbors, 1):
            columns = [class_index[t] for t in labels[target]]
            relative = max(score, 0) / max(neighbors[0][1], 1e-9)
            best_score[i, columns] = np.maximum(best_score[i, columns], relative)
            best_rank[i, columns] = np.maximum(best_rank[i, columns], 1 / rank)
            if rank <= 10:
                top10[i, columns] += 1 / min(10, len(neighbors))
    return [probs, best_score, best_rank, top10], top


def table(per_route, tops, prior, mates=None):
    """Rows for every (query, term) that a top-50 neighbour carries, plus the 20 most frequent terms."""
    support = np.zeros_like(per_route[0][0], dtype=bool)
    support[:, np.argsort(-prior)[:FREQUENT]] = True
    for blocks in per_route:
        support |= blocks[2] > 0
    queries, terms = np.nonzero(support)
    columns = [block[queries, terms] for blocks in per_route for block in blocks]
    columns += [top[queries] for top in tops]
    columns.append(np.log10(prior[terms] + 1e-6))
    if mates is not None:
        columns.append(mates[queries, terms])
    return np.column_stack(columns).astype(np.float32), queries, terms


def dense(scores, queries, terms, shape):
    out = np.zeros(shape, dtype=np.float32)
    out[queries, terms] = scores
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=("ec", "go"))
    parser.add_argument("--routes", nargs="+", required=True, help="route names; a/b means pick a or b per task by valid Fmax")
    parser.add_argument("--siblings", action="store_true", help="add the same-entry sibling vote as a feature")
    parser.add_argument("--name", default="stack")
    parser.add_argument("--configs-from", help="metrics JSON of an earlier run: cross-validate only the setting it picked "
                                               "per label set (same seed, so the same model and test scores)")
    parser.add_argument("--annotated", action="store_true",
                        help="train only on valid chains annotated for the label set; siblings only from annotated chains")
    parser.add_argument("--data-root", type=Path, default=Path("data/prepared"))
    parser.add_argument("--full-precision-output", action="store_true",
                        help="preserve float32 prediction scores for rank-sensitive micro AUPR")
    args = parser.parse_args()
    data = args.data_root / args.dataset
    train, valid, test = (records(data / f"{s}.jsonl") for s in ("train", "valid", "test"))
    by_entry = defaultdict(list)
    for row in train:
        by_entry[row["id"].split("-")[0]].append(row)
    name = args.name + ("+siblings" if args.siblings else "")
    output = {}
    for task, terms in json.loads((data / "classes.json").read_text()).items():
        class_index = {t: i for i, t in enumerate(terms)}
        labelled = [row for row in train if row["labels"][task] or not args.annotated]
        prior = truth(labelled, class_index, task).mean(0)
        selection = [row for row in valid if row["labels"][task] or not args.annotated]
        split_tables, seconds, cpu = {}, {}, {}
        for split, rows in (("valid", selection), ("test", test)):
            start, cpu_start = time.perf_counter(), time.process_time()
            built = [route_features(pick_route(r, args.dataset, task), args.dataset, task, rows, split, train, class_index)
                     for r in args.routes]
            mates = siblings(rows, by_entry, class_index, task, args.annotated) if args.siblings else None
            split_tables[split] = table([b[0] for b in built], [b[1] for b in built], prior, mates)
            seconds[split], cpu[split] = time.perf_counter() - start, time.process_time() - cpu_start
        yv, yt = truth(selection, class_index, task), truth(test, class_index, task)
        xv, qv, tv = split_tables["valid"]
        target = yv[qv, tv]
        best = None
        configs = [json.loads(Path(args.configs_from).read_text())[task]["config"]] if args.configs_from else CONFIGS
        for config in configs:
            oof = np.zeros(len(target), dtype=np.float32)
            for fit, held in GroupKFold(5).split(xv, target, qv):
                model = HistGradientBoostingClassifier(random_state=0, **config).fit(xv[fit], target[fit])
                oof[held] = model.predict_proba(xv[held])[:, 1]
            score = fmax_fast(dense(oof, qv, tv, yv.shape), yv)
            print(f"{args.dataset}-{task} {name} cv {config}: {score:.4f}", flush=True)
            if best is None or score > best[0]:
                best = (score, config, oof)
        model = HistGradientBoostingClassifier(random_state=0, **best[1]).fit(xv, target)
        xt, qt, tt = split_tables["test"]
        start, cpu_start = time.perf_counter(), time.process_time()
        pt = dense(model.predict_proba(xt)[:, 1], qt, tt, yt.shape)
        seconds["test_predict"], cpu["test_predict"] = time.perf_counter() - start, time.process_time() - cpu_start
        entry = {"routes": [pick_route(r, args.dataset, task) for r in args.routes], "siblings": args.siblings, "config": best[1], "valid_cv_fmax": best[0],
                 "candidate_rows": {"valid": int(len(qv)), "test": int(len(qt))},
                 "candidate_recall_test": float(yt[qt, tt].sum() / max(yt.sum(), 1)),
                 "test_fmax": fmax(pt, yt), "test_aupr_micro": micro_aupr(pt, yt),
                 "test_feature_seconds": seconds["test"], "test_predict_seconds": seconds["test_predict"],
                 # Wall time stretches on a shared machine, CPU time does not: features are built on one
                 # thread, the classifier predicts on OMP_NUM_THREADS.
                 "test_feature_cpu_seconds": cpu["test"], "test_predict_cpu_seconds": cpu["test_predict"],
                 "threads": int(os.environ.get("OMP_NUM_THREADS", os.cpu_count()))}
        output[task] = entry
        # Out-of-fold valid scores of the chosen setting: the cascade picks its thresholds on these.
        with Path(f"results/raw/{name}-{args.dataset}-{task}-valid-oof.jsonl").open("w") as file:
            for row, scores in zip(selection, dense(best[2], qv, tv, yv.shape)):
                file.write(json.dumps({"id": row["id"], "true": row["labels"][task],
                                       "scores": {terms[i]: (float(scores[i]) if args.full_precision_output else round(float(scores[i]), 6))
                                                  for i in np.flatnonzero(scores)}}) + "\n")
        with Path(f"results/raw/{name}-{args.dataset}-{task}-predictions.jsonl").open("w") as file:
            for row, scores in zip(test, pt):
                file.write(json.dumps({"id": row["id"], "true": row["labels"][task],
                                       "scores": {terms[i]: (float(scores[i]) if args.full_precision_output else round(float(scores[i]), 6))
                                                  for i in np.flatnonzero(scores)}}) + "\n")
        print(f"{args.dataset}-{task} {name}: cv {best[0]:.4f} test {entry['test_fmax']:.4f} "
              f"AUPR {entry['test_aupr_micro']:.4f}", flush=True)
    Path(f"results/raw/{name}-{args.dataset}-metrics.json").write_text(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
