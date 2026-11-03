"""Apply frozen training-crossfit HoLT-hard rules to DeepFRI validation and test.

Nothing here feeds back into a rule. Validation and test labels give measured
copying rates; test labels also score ESM-S, DeepFRI CNN, Naive and the
four-route retrieval ensemble on each certified test set.

The held-out copying event follows the certification's adversary: ``four``
copies from the four routes and same-PDB siblings, ``homology`` from MMseqs2,
Foldseek and siblings. Under ``homology`` the four-route event on the same
test sets is reported beside it as ``test_four``.

The certificate bounds the population risk R(H) of the certified rule; the
rate on a finite test set scatters around it. Each measured rate therefore
carries its own intervals: exact (Clopper-Pearson) over chains, and bootstrap
over PDB entries and over test families (same entry or same 30% cluster), and
``p_at_alpha`` = P(Bin(n, alpha) >= k), how surprising k answered chains would
be if R(H) were exactly alpha.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.stats import beta, binom

from cascade import chain_stats
from certify_four import build_rows, ec4, hit_panel, vote
from certify_split import TASKS, f1, hit_files, kind, records, wilson
from crossfit_certify import ADVERSARIES, copy_score, fixed_sequence, naive_prediction, prefix, stratum
from crossfit_engine import BASE_RULES, RULES, transfer
from crossfit_retrieve import DATA
from review_intervals import score


RAW = Path("results/raw")
FOUR = "ens-ann-mmseqs-ann+foldseek-ann+esm2-650m-ann+esmc-600m-ann"
# The same four routes for EC on the restored residue letters (scripts/run_seqfix_ec.sh).
FOUR_EC = "enssf-ann-mmseqssf-ann+foldseek-ann+esm2sf-650m-ann+esmcsf-600m-ann"
REPLICATES = 500
ALPHAS = (.05, .1, .15, .2, .3, .5)
VARIANTS = ("pooled", "bonferroni", "mondrian", "cluster")
SCREENS = ("learned", "similarity")


def held_families(dataset, part):
    """Families of held-out chains: components of same PDB entry and same 30% cluster of that split."""
    members = []
    with (RAW / f"certify-mmseqs30-old-{dataset}-{part}_cluster.tsv").open() as handle:
        for line in handle:
            members.append(line.rstrip("\n").split("\t")[:2])
    names = sorted({x for pair in members for x in pair})
    index = {name: i for i, name in enumerate(names)}
    first = {}
    rows, cols = [], []
    for rep, member in members:
        rows.append(index[rep])
        cols.append(index[member])
    for name in names:
        pdb = name.split("-", 1)[0]
        rows.append(index[name])
        cols.append(index[first.setdefault(pdb, name)])
    graph = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(names), len(names)))
    _, label = connected_components(graph, directed=False)
    return {name: int(label[index[name]]) for name in names}


def group_interval(rows, key, seed, group):
    if not rows:
        return [None, None]
    groups = defaultdict(list)
    for row in rows:
        groups[group(row)].append(bool(row[key]))
    counts = np.array([len(v) for v in groups.values()])
    leaks = np.array([sum(v) for v in groups.values()])
    rng = np.random.default_rng(seed)
    draw = rng.integers(0, len(counts), (REPLICATES, len(counts)))
    rates = leaks[draw].sum(1) / counts[draw].sum(1)
    return [float(x) for x in np.percentile(rates, (2.5, 97.5))]


def pdb_interval(rows, key, seed):
    return group_interval(rows, key, seed, lambda row: row["pdb"])


def exact95(k, n):
    if not n:
        return [None, None]
    lower = float(beta.ppf(.025, k, n - k + 1)) if k else 0.
    upper = float(beta.ppf(.975, k + 1, n - k)) if k < n else 1.
    return [lower, upper]


def risk(rows, key, seed, alpha=None):
    n = len(rows)
    k = sum(bool(row[key]) for row in rows)
    out = {"n": n, "leaks": k, "rate": k / n if n else None,
           "wilson95": wilson(k, n), "exact95": exact95(k, n), "pdb_ci95": pdb_interval(rows, key, seed),
           "families": len({row["family"] for row in rows}) if rows and "family" in rows[0] else None,
           "family_ci95": group_interval(rows, key, seed, lambda row: row["family"])
           if rows and "family" in rows[0] else None}
    if alpha is not None and n:
        out["p_at_alpha"] = float(binom.sf(k - 1, n, alpha))
    return out


def homology_events(data, dataset, part, rows, naive, adversary="homology"):
    """Replace the held-out copying events by the homology attacker's: top-1 and
    strict top-5 majority of MMseqs2 and Foldseek, and same-PDB siblings. The
    four-route events stay under ``four_`` keys. The ``profile`` adversary adds HMMER
    phmmer hits (results/raw/profile-<dataset>-<part>-hits.tsv) and the rules bits5,
    union3 and union5 on every attacking route."""
    train = {r["id"]: r["labels"] for r in records(data / "train.jsonl")}
    truth = {r["id"]: r["labels"] for r in records(data / f"{part}.jsonl")}
    tasks = TASKS[kind(dataset)]
    paths = hit_files(RAW, "old", dataset, part) | {"profile": RAW / f"profile-{dataset}-{part}-hits.tsv"}
    routes = ADVERSARIES[adversary]
    rules = RULES.get(adversary, BASE_RULES)
    panel = {route: hit_panel(paths[route], route, train, truth, tasks)[1] for route in routes}
    by_task = {task: {name: labels[task] for name, labels in train.items()} for task in tasks}
    siblings = defaultdict(list)
    for name, labels in train.items():
        siblings[name.split("-", 1)[0]].append(labels)
    keys = {"ec": ("leak_ec_1.0",)} | {task: (f"leak_{task}_0.5", f"leak_{task}_relative") for task in ("bp", "mf", "cc")}
    for row in rows:
        for task in tasks:
            true = truth[row["id"]][task]
            for key in keys[task]:
                row[f"four_{key}"] = row.get(key, False)
            attacks = []
            for route in routes:
                hits = panel[route][task][row["id"]]
                if hits and rules == BASE_RULES:
                    attacks += [train[hits[0][1]][task], vote(hits, by_task[task])]
                elif hits:
                    ranked = [frozenset(train[t][task]) for _, t in hits]
                    attacks += [transfer(ranked, [v for v, _ in hits], rule) for rule in rules]
            attacks += [labels[task] for labels in siblings.get(row["pdb"], ()) if labels[task]]
            if task == "ec":
                row["leak_ec_1.0"] = bool(true) and any(ec4(pred) & ec4(true) for pred in attacks)
                continue
            base = f1(naive[task], true) if true else 0.
            values = [f1(pred, true) for pred in attacks]
            row[f"leak_{task}_0.5"] = bool(true) and any(value >= .5 for value in values)
            row[f"leak_{task}_relative"] = bool(true) and any(value >= .5 and value > base for value in values)


def inside(row, cut, coords):
    return cut is not None and not (row["sibling"] or row["duplicate"]) and all(
        row[name] < bound for name, bound in zip(coords, cut))


def choose(trace, alpha):
    """Largest path point whose simultaneous upper bound is within alpha."""
    passing = [i for i, point in enumerate(trace) if point["n"] and point["ucb"] <= alpha]
    return max(passing) if passing else None


def select(found, alpha):
    """The certified candidate of one frozen test at alpha, by the test it was run with."""
    if found.get("test") == "fixed_sequence":
        return fixed_sequence(found["trace"], alpha, found["level"])
    return choose(found["trace"], alpha)


def rule(screen, variant, task, alpha):
    """Return a predicate for one frozen certification at one alpha."""
    coords = screen["coords"]
    if variant != "mondrian":
        trace = screen[variant]["trace"]
        i = select(screen[variant], alpha)
        cut = trace[i]["cut"] if i is not None else None
        return (lambda row: inside(row, cut, coords)), {"q": Q_of(trace, i), "cut": cut}
    edges = screen["mondrian"]["edges"]
    cuts = {}
    for layer in screen["mondrian"]["strata"]:
        trace = layer.get("trace") or []
        i = select(layer, alpha) if trace else None
        cuts[layer["stratum"]] = trace[i]["cut"] if i is not None else None
    return (lambda row: inside(row, cuts.get(stratum(row, task, edges)), coords)), {
        "edges": edges, "cuts": {str(k): v for k, v in cuts.items()}}


def Q_of(trace, i):
    return trace[i]["q"] if i is not None else None


class Models:
    """Per-chain Fmax statistics, loaded once per task for all test chains."""

    def __init__(self, dataset, task, ids):
        self.ids = ids
        self.pos = {name: i for i, name in enumerate(ids)}
        terms = json.loads((Path("data/prepared") / dataset / "classes.json").read_text())[task]
        self.sources = {"esm_s": f"published-esm-s-{dataset}-{task}",
                        "cnn": ("published-deepfri-cnn-restored-ec-ec" if dataset == "ec"
                                else f"published-deepfri-cnn-standard-go-{task}"),
                        "naive": f"review-naive-{dataset}-{task}",
                        "four": f"{FOUR_EC if dataset == 'ec' else FOUR}-{dataset}-{task}"}
        self.stats = {name: self.load(RAW / f"{stem}-predictions.jsonl", terms)
                      for name, stem in self.sources.items()}
        # Alphas and variants often select the same chains; same chains and seed give the same draws.
        self.done = {}

    def load(self, path, terms):
        index = {term: i for i, term in enumerate(terms)}
        probs = np.zeros((len(self.ids), len(terms)), dtype=np.float32)
        truth = np.zeros_like(probs, dtype=np.uint8)
        seen = set()
        with path.open() as handle:
            for line in handle:
                row = json.loads(line)
                i = self.pos.get(row["id"])
                if i is None:
                    continue
                seen.add(i)
                for term, value in row["scores"].items():
                    probs[i, index[term]] = value
                for term in row["true"]:
                    truth[i, index[term]] = 1
        if len(seen) != len(self.ids):
            raise ValueError(f"missing {len(self.ids) - len(seen)} predictions: {path}")
        return chain_stats(probs, truth)

    def evaluate(self, rows, seed):
        if not rows:
            return None
        key = (tuple(row["id"] for row in rows), seed)
        if key not in self.done:
            self.done[key] = self.bootstrap(rows, seed)
        return self.done[key]

    def bootstrap(self, rows, seed):
        picked = np.array([self.pos[row["id"]] for row in rows])
        groups = list(dict.fromkeys(row["pdb"] for row in rows))
        index = {name: i for i, name in enumerate(groups)}
        keys = np.array([index[row["pdb"]] for row in rows])
        def fmax(name, w):
            weights = np.zeros(len(self.ids), dtype=np.float32)
            np.add.at(weights, picked, w)
            return score(self.stats[name], weights)
        unit = np.ones(len(rows), dtype=np.float32)
        points = {name: fmax(name, unit) for name in self.sources}
        rng = np.random.default_rng(seed)
        samples = defaultdict(list)
        for _ in range(REPLICATES):
            counts = np.bincount(rng.integers(0, len(groups), len(groups)), minlength=len(groups))
            w = counts[keys].astype(np.float32)
            values = {name: fmax(name, w) for name in self.sources}
            for name, value in values.items():
                samples[name].append(value)
            for name in ("esm_s", "cnn", "four"):
                samples[f"{name}_minus_naive"].append(values[name] - values["naive"])
            samples["esm_s_minus_cnn"].append(values["esm_s"] - values["cnn"])
        pct = lambda values: [float(x) for x in np.percentile(values, (2.5, 97.5))]
        diffs = {"esm_s_minus_naive": ("esm_s", "naive"), "cnn_minus_naive": ("cnn", "naive"),
                 "four_minus_naive": ("four", "naive"), "esm_s_minus_cnn": ("esm_s", "cnn")}
        return {"n": len(rows), "pdb_groups": len(groups), "sources": self.sources,
                "fmax": {name: {"point": points[name], "pdb_ci95": pct(samples[name])}
                         for name in self.sources},
                "paired_difference": {name: {"point": points[a] - points[b], "pdb_ci95": pct(samples[name])}
                                      for name, (a, b) in diffs.items()}}


def fixed_comparison(dataset, task, eligible, selected, key, seed):
    # ProteinShake: the split's own audit, b* calibrated on its validation set (Table A2); it states no
    # identity bins, so there is no official "<30%" rule.
    audit = f"audit-ps-{dataset[3:]}" if dataset.startswith("ps-") else f"audit-deepfri-{dataset}"
    report = json.loads((RAW / f"{audit}.json").read_text())
    bstar = report["calibration"]["strong_bits"]
    with (RAW / f"{audit}-chains.tsv").open() as handle:
        official = {row["id"]: row["official_lt30"] == "True"
                    for row in csv.DictReader(handle, delimiter="\t")}
    clean = lambda row: not (row["sibling"] or row["duplicate"])
    fixed = {} if dataset.startswith("ps-") else {
        "official_lt30": sorted((r for r in eligible if official[r["id"]]),
                                key=lambda r: (r["identity"], r["bits"], r["id"]))}
    fixed |= {
        "measured_identity_lt30": sorted((r for r in eligible if r["identity"] < .3),
                                         key=lambda r: (r["identity"], r["id"])),
        "identity30_plus_bstar": sorted((r for r in eligible if r["identity"] < .3 and r["bits"] < bstar and clean(r)),
                                        key=lambda r: (max(r["identity"] / .3, r["bits"] / bstar), r["id"])),
    }
    target = len(selected)
    selected_rate = sum(bool(r[key]) for r in selected) / target if target else None
    output = {"bstar": bstar}
    for name, ranked in fixed.items():
        # Ranking by each rule's own strictness; the equal-risk count reads test
        # outcomes and is descriptive only.
        cumulative = np.cumsum([bool(r[key]) for r in ranked])
        same_risk = max((i for i, k in enumerate(cumulative, 1)
                         if selected_rate is not None and k / i <= selected_rate), default=0)
        output[name] = {"available_n": len(ranked), "full": risk(ranked, key, seed),
                        "matched": risk(ranked[:target], key, seed) if target else None,
                        "equal_or_lower_empirical_risk_n": same_risk,
                        "matching_complete": len(ranked) >= target}
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=list(DATA))
    parser.add_argument("--tag", default="full")
    parser.add_argument("--adversary", default="four", choices=ADVERSARIES)
    parser.add_argument("--design", default="representatives", choices=("representatives", "split"))
    args = parser.parse_args()
    dataset = args.dataset
    data = DATA[dataset]
    train = records(data / "train.jsonl")
    tasks = TASKS[kind(dataset)]
    # Published model predictions exist for DeepFRI's test set only; other datasets report risk alone.
    scored = not dataset.startswith("ps-")
    naive = {task: naive_prediction(train, task)[2] for task in tasks if task != "ec"}
    # The profile route was searched for test chains only (results/对手-261013.md).
    parts = ("test",) if args.adversary == "profile" else ("valid", "test")
    held = {part: build_rows(data, dataset, part, naive) for part in parts}
    if args.adversary in ("homology", "profile"):
        for part, rows in held.items():
            homology_events(data, dataset, part, rows, naive, args.adversary)
    for part, rows in held.items():
        family = held_families(dataset, part)
        for row in rows:
            row["family"] = family[row["id"]]
    full_ec = {part: {r["id"] for r in records(data / f"{part}.jsonl") if ec4(r["labels"].get("ec", ()))}
               for part in held} if kind(dataset) == "ec" else {}
    stem = prefix(dataset, args.adversary, args.design)
    source = f"{stem}-{args.tag}-certification.json"
    certification = json.loads((RAW / source).read_text())
    output = {"dataset": dataset, "source": source, "adversary": args.adversary,
              "attack_routes": ADVERSARIES[args.adversary] + ("pdb_sibling",),
              "alphas": ALPHAS, "replicates": REPLICATES, "eligible": {}, "tiers": {}}
    models = {}
    chain_rows = []
    for tier, detail in certification["tiers"].items():
        output["tiers"][tier] = {}
        for task_event, selection in detail["tasks"].items():
            task, event = task_event.split("_", 1)
            key = f"leak_{task}_relative" if event == "relative" else (
                "leak_ec_1.0" if task == "ec" else f"leak_{task}_0.5")
            eligible = {part: [r for r in rows if r[f"labels_{task}"]
                               and (task != "ec" or r["id"] in full_ec[part])]
                        for part, rows in held.items()}
            output["eligible"][task] = {part: len(rows) for part, rows in eligible.items()}
            if scored and task not in models:
                models[task] = Models(dataset, task, [r["id"] for r in eligible["test"]])
            seed = 261012 + sum(map(ord, tier + task_event))
            model = selection["learned"]["model"]
            for rows in eligible.values():
                for r in rows:
                    r[f"score_{task_event}"] = copy_score(r, task, model)
            result = {}
            for screen_name, variant in ((a, b) for a in SCREENS for b in VARIANTS):
                if variant not in selection[screen_name]:
                    continue
                result.setdefault(screen_name, {})[variant] = {}
                for alpha in ALPHAS:
                    accept, frozen = rule(selection[screen_name], variant, task, alpha)
                    chosen = {part: [r for r in rows if accept(r)] for part, rows in eligible.items()}
                    entry = {"rule": frozen, "valid": risk(chosen["valid"], key, seed, alpha) if "valid" in chosen
                             else None, "test": risk(chosen["test"], key, seed, alpha)}
                    if args.adversary in ("homology", "profile"):
                        entry["test_four"] = risk(chosen["test"], f"four_{key}", seed)
                    if chosen["test"]:
                        if scored:
                            entry["models"] = models[task].evaluate(chosen["test"], seed)
                        entry["fixed"] = fixed_comparison(dataset, task, eligible["test"],
                                                          chosen["test"], key, seed)
                    for part, rows in chosen.items():
                        chain_rows += [{"tier": tier, "screen": screen_name, "variant": variant,
                                        "alpha": alpha, "task": task, "event": event,
                                        "part": part, "id": r["id"], "pdb": r["pdb"],
                                        "leak": int(bool(r[key]))} for r in rows]
                    result[screen_name][variant][str(alpha)] = entry
                print(tier, task_event, screen_name, variant,
                      {a: ((result[screen_name][variant][str(a)]["valid"] or {}).get("n"),
                           result[screen_name][variant][str(a)]["test"]["n"],
                           result[screen_name][variant][str(a)]["test"]["rate"]) for a in ALPHAS}, flush=True)
            output["tiers"][tier][task_event] = result
    if args.adversary == "profile":
        # Every eligible test chain's event under the profile adversary, beside the homology one, so that any
        # earlier certified list can be checked against the stronger routes.
        with (RAW / f"{stem}-test-events.tsv").open("w") as handle:
            handle.write("id\ttask\tleak\thomology_leak\n")
            homology = {part: build_rows(data, dataset, part, naive) for part in ("test",)}
            homology_events(data, dataset, "test", homology["test"], naive, "homology")
            before = {r["id"]: r for r in homology["test"]}
            for task in tasks:
                key = "leak_ec_1.0" if task == "ec" else f"leak_{task}_0.5"
                for r in held["test"]:
                    if r[f"labels_{task}"]:
                        handle.write(f"{r['id']}\t{task}\t{int(bool(r[key]))}\t{int(bool(before[r['id']][key]))}\n")
    path = RAW / f"{stem}-acceptance.json"
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    chains = RAW / f"{stem}-acceptance-chains.tsv"
    with chains.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("tier", "screen", "variant", "alpha", "task", "event", "part", "id", "pdb", "leak"),
                                delimiter="\t")
        writer.writeheader()
        writer.writerows(chain_rows)
    print(path, flush=True)


if __name__ == "__main__":
    main()
