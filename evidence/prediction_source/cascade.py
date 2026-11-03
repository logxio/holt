"""Cascade: a chain stops at the cheapest layer that is sure of it, the rest move on.

Layer 1  MMseqs2 vote (missing-label rule, k and power from valid).
Layer 2  MMseqs2 + Foldseek blend (weights from valid, ens-ann).
Layer 3  inference layer over four routes (stack4-ann; on valid its out-of-fold scores), or
         with --final four the four-route blend (weights from valid, ens-ann).
A chain stops at layer 1 when its MMseqs2 top bits >= a1 and the vote concentration
(largest label score; mean over BP/MF/CC for GO) >= c1, else at layer 2 when its Foldseek
top bits >= a2 and the blend concentration >= c2, else it goes to layer 3. One decision per
chain covers all GO label sets. Every (a1, c1, a2, c2) on the grid is scored on valid (Fmax on
the chains annotated per set; GO objective is the mean of BP, MF, CC) with its mean cost per
chain from results/logs/latency-*.json; the valid Pareto front is then scored once on test.
--tag sf reads the EC sequence routes run on the restored residue letters (mmseqssf,
esm2sf-650m, esmcsf-600m, enssf-ann blends); the per-chain timings are the same tools on the
same database sizes, so the released-letter latency files are reused.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from ensemble import route_probs
from evaluate import hits, records, truth
from metrics import fmax, fmax_fast

THRESHOLDS = np.arange(0, 1, 0.01)
CONCENTRATION = (0.0, 0.3, 0.5, 0.7, 0.9, 1.0)
PERCENTILES = (0, 10, 20, 30, 40, 50, 60, 70, 80, 90)


def load_jsonl_scores(path, rows, class_index):
    found = {}
    for line in path.read_text().splitlines():
        row = json.loads(line)
        found[row["id"]] = row["scores"]
    probs = np.zeros((len(rows), len(class_index)), dtype=np.float32)
    for i, row in enumerate(rows):
        for term, score in found[row["id"]].items():
            probs[i, class_index[term]] = score
    return probs


def chain_stats(probs, labels):
    """Per chain and threshold: correct/predicted, whether anything is predicted, correct/true."""
    correct = np.zeros((len(probs), len(THRESHOLDS)), dtype=np.float32)
    predicted = np.zeros_like(correct)
    for j, t in enumerate(THRESHOLDS):
        pred = probs > t
        predicted[:, j] = pred.sum(1)
        correct[:, j] = np.logical_and(pred, labels).sum(1)
    true = labels.sum(1).astype(np.float32)
    return (np.divide(correct, predicted, out=np.zeros_like(correct), where=predicted > 0),
            predicted > 0, correct / np.maximum(true, 1)[:, None])


def fmax_routed(stats, route):
    """Fmax of the mixed prediction where chain i takes layer route[i]; same convention as fmax_fast.
    stats holds the three chain_stats arrays stacked over layers, each (layers, chains, thresholds)."""
    rows = np.arange(len(route))
    precision_ratio, active, recall_ratio = (block[route, rows] for block in stats)
    n_active = active.sum(0)
    precision = np.divide(precision_ratio.sum(0), n_active, out=np.zeros(len(THRESHOLDS)), where=n_active > 0)
    recall = recall_ratio.mean(0)
    return float((2 * precision * recall / np.maximum(precision + recall, 1e-8)).max())


def top_bits(path, train, rows):
    found = hits(path, {row["id"] for row in train}, 2)
    return np.array([found[row["id"]][0][1] if found.get(row["id"]) else 0.0 for row in rows])


def routing(features, policy):
    (s1, c1, s2, c2), (a1, b1, a2, b2) = features, policy
    stop1 = (s1 >= a1) & (c1 >= b1)
    stop2 = ~stop1 & (s2 >= a2) & (c2 >= b2)
    return np.where(stop1, 0, np.where(stop2, 1, 2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=("ec", "go"))
    parser.add_argument("--final", default="stack4-ann",
                        help="layer-3 prediction prefix (needs -valid-oof files), or 'four' for the four-route blend")
    parser.add_argument("--tolerance", type=float, default=0.005,
                        help="operating point 'chosen': cheapest valid-front policy within this of layer 3 everywhere; "
                             "'matched' is the cheapest one at least as good as layer 3 everywhere on valid")
    parser.add_argument("--tag", default="", help="'sf': the EC sequence routes on restored residue letters")
    args = parser.parse_args()
    data = Path("data/prepared") / args.dataset
    train, valid, test = (records(data / f"{s}.jsonl") for s in ("train", "valid", "test"))
    classes = json.loads((data / "classes.json").read_text())
    sequence, blend_name = f"mmseqs{args.tag}", f"ens{args.tag}-ann"
    routes = (f"{sequence}-ann", "foldseek-ann", f"esm2{args.tag}-650m-ann", f"esmc{args.tag}-600m-ann")
    blends = json.loads(Path(f"results/raw/{blend_name}-{args.dataset}-metrics.json").read_text())
    timing = {name: json.loads(Path(f"results/logs/latency-{name}-{args.dataset}.json").read_text())
              for name in ("mmseqs", "foldseek", "esm2-650m", "esmc-600m")}
    four = f"{blend_name}-" + "+".join(routes)
    # The Mac was shared (load average 150-290 while timing), which stretches wall time; CPU time is not
    # stretched, so a search layer costs its batch CPU time spread over its 10 threads. On the cloud
    # Threadripper this matched the measured 10-thread wall time (MMseqs2 0.89 vs 0.97, Foldseek 11.3 vs 11.5 ms).
    idle = {tool: timing[tool]["batch_cpu_ms_per_chain"] / timing[tool]["threads"] for tool in ("mmseqs", "foldseek")}
    per_chain = {"mmseqs": idle["mmseqs"], "foldseek": idle["foldseek"],
                 "esm2": timing["esm2-650m"]["batch_ms_per_chain"], "esmc": timing["esmc-600m"]["batch_ms_per_chain"]}
    if args.final != "four":
        # Measured wall time of the inference layer; a GO chain pays for all three label sets.
        final_metrics = json.loads(Path(f"results/raw/{args.final}-{args.dataset}-metrics.json").read_text())
        per_chain["stack"] = sum(1000 * (m["test_feature_seconds"] + m["test_predict_seconds"])
                                 / len(test) for m in final_metrics.values())
    layer_cost = np.array([per_chain["mmseqs"], per_chain["mmseqs"] + per_chain["foldseek"],
                           sum(per_chain.values())])
    splits = {"valid": valid, "test": test}
    features, stats, mixed, truths, annotated = {}, {}, {}, {}, {}
    concentration = {split: [] for split in splits}
    for task, terms in classes.items():
        class_index = {t: i for i, t in enumerate(terms)}
        layer1, _ = route_probs(routes[0], args.dataset, task, splits, train, class_index)
        blend = blends[task]["blends"][f"{blend_name}-{routes[0]}+foldseek-ann"]["weights"]
        foldseek, _ = route_probs("foldseek-ann", args.dataset, task, splits, train, class_index)
        layer2 = {s: blend[routes[0]] * layer1[s] + blend["foldseek-ann"] * foldseek[s] for s in splits}
        for split in splits:
            concentration[split].append((layer1[split].max(1), layer2[split].max(1)))
        keep = np.array([bool(row["labels"][task]) for row in valid])
        annotated[task] = keep
        if args.final == "four":
            weights = blends[task]["blends"][four]["weights"]
            picked = blends[task]["picked_routes"]
            layer3 = {s: 0 for s in splits}
            for route in routes:
                probs, _ = route_probs(picked[route], args.dataset, task, splits, train, class_index)
                layer3 = {s: layer3[s] + weights[route] * probs[s] for s in splits}
            layer3["valid"] = layer3["valid"][keep]
        else:
            layer3 = {"valid": load_jsonl_scores(Path(f"results/raw/{args.final}-{args.dataset}-{task}-valid-oof.jsonl"),
                                                 [r for r, k in zip(valid, keep) if k], class_index),
                      "test": load_jsonl_scores(Path(f"results/raw/{args.final}-{args.dataset}-{task}-predictions.jsonl"),
                                                test, class_index)}
        truths[task] = {"valid": truth([r for r, k in zip(valid, keep) if k], class_index, task),
                        "test": truth(test, class_index, task)}
        mixed[task] = {"valid": [layer1["valid"][keep], layer2["valid"][keep], layer3["valid"]],
                       "test": [layer1["test"], layer2["test"], layer3["test"]]}
        stats[task] = {}
        for split in splits:
            per_layer = [chain_stats(p, truths[task][split]) for p in mixed[task][split]]
            stats[task][split] = tuple(np.stack([layer[k] for layer in per_layer]) for k in range(3))
        print(f"{args.dataset}-{task}: layers ready", flush=True)
    for split, rows in splits.items():
        c1 = np.mean([c[0] for c in concentration[split]], axis=0)
        c2 = np.mean([c[1] for c in concentration[split]], axis=0)
        features[split] = (top_bits(Path(f"results/raw/{sequence}-{args.dataset}-{split}-hits.tsv"), train, rows), c1,
                           top_bits(Path(f"results/raw/foldseek-{args.dataset}-{split}-hits.tsv"), train, rows), c2)

    def score(split, policy):
        route = routing(features[split], policy)
        per_task = {}
        for task in classes:
            r = route[annotated[task]] if split == "valid" else route
            per_task[task] = fmax_routed(stats[task][split], r)
        return per_task, float(layer_cost[route].mean()), np.bincount(route, minlength=3) / len(route)

    grid1 = sorted(set(np.percentile(features["valid"][0], PERCENTILES).tolist())) + [np.inf]
    grid2 = sorted(set(np.percentile(features["valid"][2], PERCENTILES).tolist())) + [np.inf]
    candidates = []
    for a1 in grid1:
        for b1 in CONCENTRATION:
            if a1 == np.inf and b1 > 0:
                continue
            for a2 in grid2:
                for b2 in CONCENTRATION:
                    if a2 == np.inf and b2 > 0:
                        continue
                    policy = (a1, b1, a2, b2)
                    per_task, cost, _ = score("valid", policy)
                    candidates.append((cost, float(np.mean(list(per_task.values()))), policy))
    candidates.sort(key=lambda c: (c[0], -c[1]))
    front, best = [], -1.0
    for cost, objective, policy in candidates:
        if objective > best + 1e-9:
            front.append((cost, objective, policy))
            best = objective
    everything = (np.inf, 0.0, np.inf, 0.0)
    full_objective = float(np.mean(list(score("valid", everything)[0].values())))
    chosen = min((c for c in front if c[1] >= full_objective - args.tolerance), key=lambda c: c[0])
    matched = min((c for c in front if c[1] >= full_objective), key=lambda c: c[0], default=front[-1])
    output = {"per_chain_ms": per_chain, "layer_cost_ms": layer_cost.tolist(),
              "loaded_wall_ms": {tool: timing[tool]["batch_ms_per_chain"] for tool in timing},
              "load_average": {tool: timing[tool].get("load_average") for tool in timing}, "valid_candidates": len(candidates),
              "full_layer3_valid_objective": full_objective, "tolerance": args.tolerance, "front": []}
    for cost, objective, policy in front:
        test_tasks, test_cost, shares = score("test", policy)
        output["front"].append({"policy": [float(x) for x in policy], "valid_objective": objective,
                                "valid_cost_ms": cost, "test_cost_ms": test_cost, "test_shares": shares.tolist(),
                                "test_fmax": test_tasks, "chosen": policy == chosen[2], "matched": policy == matched[2]})
    named = {"layer1_only": (-np.inf, 0.0, np.inf, 0.0), "layer2_only": (np.inf, 0.0, -np.inf, 0.0),
             "layer3_everywhere": everything, "chosen": chosen[2], "matched": matched[2]}
    output["named"] = {}
    for name, policy in named.items():
        route = routing(features["test"], policy)
        official = {}
        for task in classes:
            probs = np.stack(mixed[task]["test"])[route, np.arange(len(route))]
            official[task] = fmax(probs, truths[task]["test"])
            if abs(official[task] - fmax_fast(probs, truths[task]["test"])) > 1e-6:
                raise AssertionError("fmax_fast disagrees with MotiL's Fmax")
        output["named"][name] = {"policy": [float(x) for x in policy], "test_fmax": official,
                                 "test_cost_ms": float(layer_cost[route].mean()),
                                 "test_shares": (np.bincount(route, minlength=3) / len(route)).tolist(),
                                 "valid_objective": float(np.mean(list(score("valid", policy)[0].values())))}
        print(f"{args.dataset} {name}: {output['named'][name]}", flush=True)
    name = args.final + (args.tag if args.final == "four" else "")
    Path(f"results/raw/cascade-{name}-{args.dataset}.json").write_text(json.dumps(output, indent=2))
    print(f"{args.dataset}: front {len(front)} points of {len(candidates)}", flush=True)


if __name__ == "__main__":
    main()
