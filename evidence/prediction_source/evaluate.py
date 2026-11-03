"""Tune neighbor voting on valid, then evaluate each held-out MotiL task once."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from metrics import fmax, fmax_fast, micro_aupr


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def clean_id(value):
    name = Path(value).name.split(".pdb")[0].split("_A")[0]
    if name.startswith("X") and len(name) > 1 and len(name[1:]) % 2 == 0:
        try:
            decoded = bytes.fromhex(name[1:]).decode("ascii")
            if "-" in decoded and decoded.encode("ascii").hex().upper() == name[1:]:
                return decoded
        except (ValueError, UnicodeDecodeError):
            pass
    return name


def score_columns(path):
    """Hit files carry one or more score columns after query and target."""
    with path.open() as file:
        return range(2, len(file.readline().rstrip("\n").split("\t")))


def hits(path, train_ids, column=2):
    result = defaultdict(dict)
    with path.open() as file:
        for line in file:
            row = line.rstrip().split("\t")
            if len(row) <= column:
                raise ValueError(f"bad hit line in {path}: {line[:100]}")
            query, target = clean_id(row[0]), clean_id(row[1])
            if target not in train_ids or query == target:
                continue
            score = float(row[column])
            if score > result[query].get(target, float("-inf")):
                result[query][target] = score
    return {query: sorted(targets.items(), key=lambda pair: pair[1], reverse=True)[:100]
            for query, targets in result.items()}


def task_hits(path, train, task, column, annotated):
    """Neighbours restricted to train chains; with annotated, only chains labelled for this task vote."""
    return hits(path, {row["id"] for row in train if row["labels"][task] or not annotated}, column)


def vote(rows, train_labels, class_index, retrieval, k, power):
    probabilities = np.zeros((len(rows), len(class_index)), dtype=np.float32)
    no_hits = 0
    for row_index, row in enumerate(rows):
        neighbors = retrieval.get(row["id"], [])[:k]
        if not neighbors:
            no_hits += 1
            continue
        scores = np.array([max(score, 0.0) for _, score in neighbors], dtype=np.float64)
        if power == 0 or scores.max() == 0:
            weights = np.ones(len(neighbors))
        else:
            weights = (scores / scores.max()) ** power
        weights /= weights.sum()
        for (target, _), weight in zip(neighbors, weights):
            for term in train_labels[target]:
                probabilities[row_index, class_index[term]] += weight
    return probabilities, no_hits


def truth(rows, class_index, task):
    y = np.zeros((len(rows), len(class_index)), dtype=np.uint8)
    for i, row in enumerate(rows):
        for term in row["labels"][task]:
            y[i, class_index[term]] = 1
    return y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("method", help="prefix of results/raw/<method>-<dataset>-{valid,test}-hits.tsv")
    parser.add_argument("dataset", choices=("ec", "go"))
    parser.add_argument("--data-root", type=Path, default=Path("data/prepared"),
                        help="prepared split root; original DeepFRI data lives in data/prepared/deepfri")
    parser.add_argument("--annotated", action="store_true",
                        help="per label set, only train chains annotated for it vote and only annotated valid chains "
                             "select k/power (every test chain is annotated); writes <method>-ann-* files")
    args = parser.parse_args()
    data = args.data_root / args.dataset
    train, valid, test = (records(data / f"{split}.jsonl") for split in ("train", "valid", "test"))
    class_names = json.loads((data / "classes.json").read_text())
    valid_path = Path(f"results/raw/{args.method}-{args.dataset}-valid-hits.tsv")
    test_path = Path(f"results/raw/{args.method}-{args.dataset}-test-hits.tsv")
    name = args.method + ("-ann" if args.annotated else "")
    output = {}
    for task, terms in class_names.items():
        class_index = {name: index for index, name in enumerate(terms)}
        train_labels = {row["id"]: row["labels"][task] for row in train}
        selection = [row for row in valid if row["labels"][task] or not args.annotated]
        valid_truth = truth(selection, class_index, task)
        best = (-1.0, None, None, None)
        for column in score_columns(valid_path):
            valid_hits = task_hits(valid_path, train, task, column, args.annotated)
            for k in (1, 3, 5, 10, 20, 50):
                for power in (0, 1, 2):
                    probs, _ = vote(selection, train_labels, class_index, valid_hits, k, power)
                    value = fmax_fast(probs, valid_truth)
                    if value > best[0]:
                        best = (value, k, power, column)
        _, k, power, column = best
        test_hits = task_hits(test_path, train, task, column, args.annotated)
        final_probs, no_hits = vote(test, train_labels, class_index, test_hits, k, power)
        final_truth = truth(test, class_index, task)
        official_fmax = fmax(final_probs, final_truth)
        final_aupr = micro_aupr(final_probs, final_truth)
        output[task] = {"hits": args.method, "annotated_only": args.annotated, "valid_count": len(selection),
                        "valid_fmax": best[0], "k": k, "weight_power": power, "score_column": column,
                        "test_fmax": official_fmax, "test_aupr_micro": final_aupr,
                        "test_count": len(test), "no_hit_count": no_hits}
        raw_path = Path(f"results/raw/{name}-{args.dataset}-{task}-predictions.jsonl")
        with raw_path.open("w") as file:
            for row, scores in zip(test, final_probs):
                predicted = {terms[i]: float(scores[i]) for i in np.flatnonzero(scores)}
                file.write(json.dumps({"id": row["id"], "true": row["labels"][task],
                                       "scores": predicted, "neighbors": test_hits.get(row["id"], [])[:k]}) + "\n")
        print(f"{name} {args.dataset}-{task}: valid {best[0]:.4f}, test Fmax {official_fmax:.4f}, AUPR {final_aupr:.4f}, k={k}, power={power}, column={column}, no hits={no_hits}", flush=True)
    path = Path(f"results/raw/{name}-{args.dataset}-metrics.json")
    path.write_text(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
