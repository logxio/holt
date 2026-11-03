"""HoLT-hard certification on training chains whose copying events are independent (HoLT-2).

Training chains fall into families: connected components of same-PDB-entry and
same-MMseqs2-30%-cluster links. Copying events of two calibration chains are
dependent when one reads the other's labels, as a top-5 hit of a route, or when
they share a family. HoLT-2 therefore calibrates on representatives
(``--design representatives``, the default): per tier and label set, at most
one chain per family, and no representative among another's top-5 annotated
hits on any route, chosen from hit lists alone. Each representative retrieves
the whole training set outside its own family (tier ``mmseqs30``) or PDB entry
(tier ``pdb``), as a test chain retrieves the training set; given every
sequence and every other label, the representatives' events are independent,
and the exact binomial test holds as stated in results/方法-理论-261013.tex.
Design chains retrieve the training set without the representatives, so the
learned score and the path never read a representative's label.

``--design split`` is the other exact route: a hash puts 40% of families in
calibration, one representative each, retrieving only the other 60%, which is
also the design set. Its library is 60% of the one test chains search.

Tests (protocol results/raw/crossfit-protocol-v2.json): fixed sequence along
the 12-point design-quantile path at delta/4, four label sets in the family,
from the first candidate that could pass with no event at alpha = 0.05 (one
label-free start for every alpha; primary, key ``pooled``); Bonferroni at
delta/(4x12) as in HoLT-1 (``bonferroni``); Mondrian strata on the label-free
vote size; in the split design's ``mmseqs30`` tier, every chain of a
calibration family with the family counted once (``cluster``). The relative GO events are tested at the same
levels as diagnostics outside the family.

The previous design, every training chain a calibration query and chains as
units (HoLT-1), is archived in results/raw/crossfit-loo/ and rerun on
training-internal splits by crossfit_validity.py.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import binom
from sklearn.linear_model import LogisticRegression

from certify_split import cp_upper, records
from crossfit_engine import ADVERSARIES, BASE_RULES, ROUTES, RULES, Pool, features, unit_hash
from crossfit_retrieve import DATA


RAW = Path("results/raw")
PROTOCOL = json.loads((RAW / "crossfit-protocol-v2.json").read_text())
Q = tuple(PROTOCOL["path"]["design_quantiles"])
COORDS = ("identity", "bits", "esm2", "esmc")
EVENTS = (("ec", "exact"), ("bp", "raw"), ("bp", "relative"),
          ("mf", "raw"), ("mf", "relative"), ("cc", "raw"), ("cc", "relative"))
FAMILY = tuple(PROTOCOL["events"]["family_of_label_sets"])
TIERS = ("pdb", "mmseqs30")
FEATURES = tuple(PROTOCOL["score"]["features"])
CALIBRATION = PROTOCOL["roles"]["calibration_fraction_split"]
STRATA = 3
DELTA = PROTOCOL["levels"]["delta"]
ALPHA = PROTOCOL["levels"]["primary_alpha"]
PER_TEST = {"pooled": DELTA / len(FAMILY),
            "bonferroni": DELTA / (len(FAMILY) * len(Q)),
            "mondrian": DELTA / (len(FAMILY) * STRATA),
            "cluster": DELTA / len(FAMILY)}
# The fixed sequence starts, for every alpha, at the first candidate that could pass with no event at the
# smallest reported level, so its start never moves with alpha.
START_ALPHA = min(PROTOCOL["levels"]["alpha"])
ALPHA_GRID = np.round(np.arange(.005, .9995, .001), 3)


def assert_tail_position(n: int, alpha: float, level: float) -> float | None:
    """Check the finite-domain and binomial-tail premise used by the risk bound."""
    if not isinstance(n, int) or n < 0 or n > 20_000:
        raise ValueError(f"representative count outside theorem domain: N={n}")
    if not .005 <= alpha <= .5:
        raise ValueError(f"alpha outside theorem domain: {alpha}")
    if not 0 < level <= .025:
        raise ValueError(f"test level outside theorem domain: {level}")
    if n == 0:
        return None
    margin = float(binom.cdf(math.floor(n * alpha), n, alpha)) - level
    if margin <= 0:
        raise ValueError(f"binomial tail position fails: N={n}, alpha={alpha}, gamma={level}")
    return margin


def naive_prediction(train, task):
    """Training term frequencies thresholded for the best Fmax on PDB-hashed design chains (Naive)."""
    from collections import Counter
    annotated = [row for row in train if row["labels"][task]]
    count = Counter(x for row in annotated for x in set(row["labels"][task]))
    frequencies = {x: n / len(annotated) for x, n in count.items()}
    held = [row for row in annotated if pdb_design(row["id"])]
    best = (-1., None, set())
    for t in np.arange(0, 1, .01):
        pred = {x for x, p in frequencies.items() if p > t}
        if not pred:
            continue
        precision = sum(len(pred & set(row["labels"][task])) / len(pred) for row in held) / len(held)
        recall = sum(len(pred & set(row["labels"][task])) / len(row["labels"][task]) for row in held) / len(held)
        metric = 2 * precision * recall / max(precision + recall, 1e-8)
        if metric > best[0]:
            best = (metric, float(t), pred)
    return best


def pdb_design(name):
    pdb = name.split("-", 1)[0]
    return int.from_bytes(hashlib.sha256(pdb.encode()).digest()[:4], "big") / 2**32 < .2


def prefix(dataset, adversary, design="representatives"):
    return (f"crossfit-{dataset}" + ("" if adversary == "four" else f"-{adversary}")
            + ("" if design == "representatives" else f"-{design}"))


def cluster_map(dataset):
    mapping = {}
    with (RAW / f"crossfit-{dataset}-mmseqs30_cluster.tsv").open() as handle:
        for line in handle:
            rep, member = line.rstrip("\n").split("\t")[:2]
            mapping[member] = rep
    return mapping


def hit_panel(dataset, tag, route, tier, train_labels, clusters, tasks):
    """List-based panel of the leave-one-out design (kept for crossfit_routes.py)."""
    name = f"crossfit-{dataset}-{tag}-{route}"
    if tier == "mmseqs30" and route in ("esm2-650m", "esmc-600m"):
        name += "-cluster"
    scores = defaultdict(float)
    top = {task: defaultdict(list) for task in tasks}
    with (RAW / f"{name}-hits.tsv").open() as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            q, target = fields[:2]
            if q not in train_labels or target not in train_labels:
                continue
            if q.split("-", 1)[0] == target.split("-", 1)[0]:
                continue
            if tier == "mmseqs30" and clusters[q] == clusters[target]:
                continue
            rank = float(fields[6] if route == "mmseqs" else fields[2])
            value = float(fields[2]) if route == "mmseqs" else rank
            scores[q] = max(scores[q], value)
            for task in tasks:
                if not train_labels[target][task]:
                    continue
                hits = top[task][q]
                hits.append((rank, target))
                hits.sort(reverse=True)
                del hits[5:]
    return scores, top


def learn(design_rows, task, key):
    """HoLT Learn step: logistic copy-risk score fitted on design chains only."""
    x = np.array([features(row, task) for row in design_rows], dtype=float)
    y = np.array([bool(row[key]) for row in design_rows])
    mean, scale = x.mean(0), x.std(0) + 1e-12
    fit = LogisticRegression(C=1., max_iter=2000).fit((x - mean) / scale, y)
    return {"features": FEATURES, "mean": mean.tolist(), "scale": scale.tolist(),
            "coef": fit.coef_[0].tolist(), "intercept": float(fit.intercept_[0]),
            "design_n": len(y), "design_events": int(y.sum())}


def copy_score(row, task, model):
    """Linear predictor of the learned score; larger means easier to copy."""
    x = (np.array(features(row, task)) - np.array(model["mean"])) / np.array(model["scale"])
    return float(x @ np.array(model["coef"]) + model["intercept"])


def quantile_path(rows, coords):
    values = {name: np.array([row[name] for row in rows]) for name in coords}
    return [tuple(math.nextafter(float(np.percentile(values[name], q)), math.inf)
                  if q == 100 else float(np.percentile(values[name], q)) for name in coords)
            for q in Q]


def inside(row, cut, coords):
    return all(row[name] < bound for name, bound in zip(coords, cut))


def strata_edges(design_rows, task):
    """Design tertile edges of the vote-size proxy; tied edges merge strata."""
    values = [row[f"vote_size_{task}"] for row in design_rows]
    return sorted({float(np.percentile(values, q)) for q in (100 / 3, 200 / 3)})


def stratum(row, task, edges):
    return sum(row[f"vote_size_{task}"] > edge for edge in edges)


def zero_min(alpha, level):
    """Fewest units a candidate needs to pass with no event: (1 - alpha)^n <= level."""
    return math.ceil(math.log(level) / math.log(1 - alpha))


def fixed_sequence(trace, alpha, level):
    """Largest candidate of the fixed-sequence test at alpha: start at the first candidate large enough to
    pass with zero events at the smallest reported alpha (a label-free choice, the same for every alpha),
    continue while each passes, stop at the first failure."""
    need = zero_min(START_ALPHA, level)
    start = next((i for i, point in enumerate(trace) if point["n"] >= need), None)
    chosen = None
    if start is not None:
        for i in range(start, len(trace)):
            if trace[i]["ucb"] > alpha:
                break
            chosen = i
    return chosen


def bonferroni(trace, alpha):
    passing = [i for i, point in enumerate(trace) if point["n"] and point["ucb"] <= alpha]
    return max(passing) if passing else None


def test_path(calibration, cuts, coords, key, level, sequential, unit=None):
    """Counts along the path, one binomial test per candidate. ``unit`` groups chains (a unit is answered when
    any of its selected chains is)."""
    trace = []
    for i, cut in enumerate(cuts):
        picked = [row for row in calibration if inside(row, cut, coords)]
        if unit is None:
            n, k = len(picked), sum(bool(row[key]) for row in picked)
        else:
            groups = defaultdict(bool)
            for row in picked:
                groups[row[unit]] |= bool(row[key])
            n, k = len(groups), sum(groups.values())
        for alpha in PROTOCOL["levels"]["alpha"]:
            assert_tail_position(int(n), float(alpha), float(level))
        trace.append({"q": Q[i], "cut": cut, "n": n, "leaks": k, "rate": k / n if n else None,
                      "p": float(binom.cdf(k, n, ALPHA)) if n else 1., "ucb": cp_upper(k, n, level)})
    choose = (lambda a: fixed_sequence(trace, a, level)) if sequential else (lambda a: bonferroni(trace, a))
    chosen = choose(ALPHA)
    for i, point in enumerate(trace):
        # Under fixed sequence a candidate is certified when it lies in the passing run that ends at the choice.
        point["certified"] = point["ucb"] <= ALPHA and (not sequential or (chosen is not None and i <= chosen))
    reachable = [a for a in ALPHA_GRID if choose(float(a)) is not None]
    return {"test": "fixed_sequence" if sequential else "bonferroni", "level": level,
            "selected_index": chosen, "selected_q": Q[chosen] if chosen is not None else None,
            "selected_cut": cuts[chosen] if chosen is not None else None, "trace": trace,
            "min_p": min(point["p"] for point in trace),
            # The smallest alpha at which this test certifies a non-empty candidate.
            "min_ucb": float(reachable[0]) if reachable else 1.0,
            "min_bound": min(point["ucb"] for point in trace)}


def certify_screen(design_rows, calibration, members, task, key, coords):
    cuts = quantile_path(design_rows, coords)
    out = {"coords": coords,
           "pooled": test_path(calibration, cuts, coords, key, PER_TEST["pooled"], True),
           "bonferroni": test_path(calibration, cuts, coords, key, PER_TEST["bonferroni"], False)}
    if members is not None:
        out["cluster"] = test_path(members, cuts, coords, key, PER_TEST["cluster"], True, unit="family")
    edges = strata_edges(design_rows, task)
    layers = []
    for s in range(len(edges) + 1):
        d = [row for row in design_rows if stratum(row, task, edges) == s]
        c = [row for row in calibration if stratum(row, task, edges) == s]
        result = test_path(c, quantile_path(d, coords), coords, key, PER_TEST["mondrian"], True) if d else None
        layers.append({"stratum": s, "design_n": len(d), "calibration_n": len(c), **(result or {})})
    out["mondrian"] = {"edges": edges, "proxy": f"vote_size_{task}", "strata": layers}
    return out


def certify(design_rows, calibration, members, task, event):
    key = f"leak_{task}_{event}"
    model = learn(design_rows, task, key)
    name = f"score_{task}_{event}"
    for row in [*design_rows, *calibration, *(members or [])]:
        row[name] = copy_score(row, task, model)
    return {"design_n": len(design_rows), "calibration_n": len(calibration),
            "calibration_clusters": len({row["family"] for row in calibration}),
            "in_family": f"{task}_{event}" in FAMILY,
            "learned": {"model": model, **certify_screen(design_rows, calibration, members, task, key, (name,))},
            "similarity": certify_screen(design_rows, calibration, members, task, key, COORDS)}


KEYS = {"mmseqs30": "family", "pdb": "pdb"}


def sources(pool, tier, library=None):
    """Per label set, the chains whose labels a chain's features and copying event read: the top five annotated
    hits of every route, retrieved from the library outside the chain's own family or entry."""
    n = len(pool.ids)
    library = np.ones(n, bool) if library is None else library
    _, top = pool.panels(np.arange(n), library, getattr(pool, KEYS[tier]))
    out = {}
    for task in pool.tasks:
        stacked = np.concatenate([top[route][task] for route in pool.routes], axis=1)
        out[task] = [set(int(x) for x in row if x >= 0) for row in stacked]
    return out


class Representatives:
    """HoLT-2: calibration chains that never read each other's labels.

    Per tier and label set, at most one eligible chain per family is a representative, and no representative is a
    copy source (a top-5 annotated hit of any route) of another. The set is grown greedily from the hit lists
    alone, chains with the fewest conflicts first. Representatives retrieve the whole training set outside their
    own family (mmseqs30) or entry (pdb), as test chains retrieve it; design chains retrieve it without the
    representatives, so the learned score never reads a representative's label."""

    name = "representatives"

    def __init__(self, pool, priority="degree", chains=None, tiers=TIERS):
        self.pool = pool
        self.priority = priority
        n = len(pool.ids)
        self.chains = np.ones(n, bool) if chains is None else chains
        rep_hash = np.array([unit_hash("rep:" + name) for name in pool.ids])
        self.sources, self.representatives, self.stats = {}, {}, {}
        for tier in tiers:
            self.sources[tier] = sources(pool, tier, self.chains)
            if priority == "hardness":
                best = pool.panels(np.arange(n), np.ones(n, bool), getattr(pool, KEYS[tier]))[0]
                # Label-free hardness: the mean percentile of a chain's best score on the four routes.
                hardness = np.mean([np.argsort(np.argsort(best[route])) / n for route in ROUTES], axis=0)
            self.representatives[tier], self.stats[tier] = {}, {}
            for task in pool.tasks:
                eligible = pool.eligible[task] & self.chains
                src = self.sources[tier][task]
                degree = np.zeros(n, int)
                for i in np.flatnonzero(eligible):
                    for s in src[i]:
                        if eligible[s]:
                            degree[s] += 1
                            degree[i] += 1
                rank = (hardness if priority == "hardness" else degree if priority == "degree"
                        else np.zeros(n))  # "random": hash order alone
                order = sorted(np.flatnonzero(eligible), key=lambda i: (rank[i], rep_hash[i]))
                chosen = np.zeros(n, bool)
                blocked = np.zeros(n, bool)
                taken = set()
                for i in order:
                    if pool.family[i] in taken or blocked[i] or any(chosen[s] for s in src[i]):
                        continue
                    taken.add(pool.family[i])
                    chosen[i] = True
                    for s in src[i]:
                        blocked[s] = True
                reps = np.flatnonzero(chosen)
                if any(chosen[s] for i in reps for s in src[i]):
                    raise AssertionError(f"{tier} {task}: a representative reads another representative's label")
                self.representatives[tier][task] = reps
                self.stats[tier][task] = {"eligible_chains": int(eligible.sum()),
                                          "eligible_families": int(len(set(pool.family[eligible]))),
                                          "representatives": int(len(reps))}

    def rows(self, tier, naive, attackers):
        pool = self.pool
        key = getattr(pool, KEYS[tier])
        everything = self.chains
        reps = np.unique(np.concatenate(list(self.representatives[tier].values())))
        calibration = {row["index"]: row for row in pool.rows(reps, everything, key, naive, attackers)}
        design, cal = {}, {}
        for task in pool.tasks:
            chosen = self.representatives[tier][task]
            library = everything.copy()
            library[chosen] = False
            queries = np.flatnonzero(library & pool.eligible[task])
            design[task] = pool.rows(queries, library, key, naive, attackers, (task,))
            cal[task] = [calibration[i] for i in chosen]
        return design, cal, None

    def describe(self, tier):
        return {"representatives": self.stats[tier], "library": "every training chain outside the query's own "
                + ("family" if tier == "mmseqs30" else "PDB entry"), "library_fraction": 1.0}


class Split:
    """Disjoint reference library: 40% of families calibrate, one representative chain each, and retrieve only
    the other 60%, which is also the design set."""

    name = "split"

    def __init__(self, pool, chains=None, salt=""):
        self.pool = pool
        n = len(pool.ids)
        self.chains = np.ones(n, bool) if chains is None else chains
        names = pool.family_name
        self.calibration = np.array([unit_hash(names[i], salt) < CALIBRATION for i in range(n)]) & self.chains
        self.design = self.chains & ~self.calibration
        rep_hash = np.array([unit_hash("rep:" + name) for name in pool.ids])
        self.representatives = {}
        for task in pool.tasks:
            best = {}
            for i in np.flatnonzero(self.calibration & pool.eligible[task]):
                f = pool.family[i]
                if f not in best or rep_hash[i] < rep_hash[best[f]]:
                    best[f] = i
            self.representatives[task] = np.array(sorted(best.values()), dtype=np.int64)
        held_entries = np.zeros(pool.pdb.max() + 1, bool)
        for reps in self.representatives.values():
            held_entries[pool.pdb[reps]] = True
        self.library = {"mmseqs30": self.design.copy(),
                        "pdb": self.chains & ~held_entries[pool.pdb]}

    def rows(self, tier, naive, attackers, members=True):
        pool, library = self.pool, self.library[tier]
        key = getattr(pool, KEYS[tier])
        everyone = pool.rows(np.flatnonzero(self.design), library, key, naive, attackers)
        reps = np.unique(np.concatenate(list(self.representatives.values())))
        calibration = {row["index"]: row for row in pool.rows(reps, library, key, naive, attackers)}
        if tier == "mmseqs30" and members:
            members = pool.rows(np.flatnonzero(self.calibration), library, key, naive, attackers)
        design = {task: [row for row in everyone if pool.eligible[task][row["index"]]] for task in pool.tasks}
        cal = {task: [calibration[i] for i in self.representatives[task]] for task in pool.tasks}
        mem = ({task: [row for row in members if pool.eligible[task][row["index"]]] for task in pool.tasks}
               if tier == "mmseqs30" and members else None)
        return design, cal, mem

    def describe(self, tier):
        pool = self.pool
        return {"families": {"calibration": int(len(np.unique(pool.family[self.calibration]))),
                             "library": int(len(np.unique(pool.family[self.design])))},
                "representatives": {task: int(len(v)) for task, v in self.representatives.items()},
                "library_chains": int(self.library[tier].sum()),
                "library_fraction": float(self.library[tier].sum() / len(pool.ids))}


def run(pool, design_of, tier, naive, attackers):
    design, calibration, members = design_of.rows(tier, naive, attackers)
    tasks = {}
    for task, event in EVENTS:
        if task not in pool.tasks:
            continue
        tasks[f"{task}_{event}"] = certify(design[task], calibration[task],
                                           members[task] if members is not None else None, task, event)
    return design, calibration, members, tasks


def library_curve(pool, split, tier, naive, attackers, tasks, design, fractions=(.5, .75, 1.)):
    """Split design, design chains only: leak rate on the frozen path when the library shrinks to a fraction of
    its families. No calibration label is read; this shows which way library size moves the risk."""
    out = {}
    base = split.library[tier]
    for fraction in fractions:
        keep = np.array([unit_hash(name, "library:") < fraction for name in pool.family_name])
        rows = None if fraction == 1. else pool.rows(np.flatnonzero(split.design), base & keep,
                                                     getattr(pool, KEYS[tier]), naive, attackers)
        out[str(fraction)] = {}
        for name, result in tasks.items():
            task, event = name.split("_", 1)
            if f"{task}_{event}" not in FAMILY:
                continue
            key = f"leak_{task}_{event}"
            model = result["learned"]["model"]
            d = design[task] if rows is None else [row for row in rows if pool.eligible[task][row["index"]]]
            cuts = [point["cut"] for point in result["learned"]["pooled"]["trace"]]
            scores = np.array([copy_score(row, task, model) for row in d])
            leaks = np.array([bool(row[key]) for row in d])
            out[str(fraction)][name] = [{"q": Q[i], "n": int((scores < cut[0]).sum()),
                                         "leaks": int(leaks[scores < cut[0]].sum())}
                                        for i, cut in enumerate(cuts)]
    return out


def write_chains(path, design, calibration, members):
    """One line per chain and role; each label set's columns come from that label set's rows, and
    ``rep_<task>`` marks the representatives of each label set."""
    tasks = list(design)
    belongs = lambda column, task: column.endswith(f"_{task}") or f"_{task}_" in column
    merged = {}
    for role, groups in (("design", design), ("calibration", calibration), ("member", members or {})):
        for task, group in groups.items():
            for row in group:
                entry = merged.setdefault((role, row["index"]), {"role": role})
                entry.update({k: v for k, v in row.items()
                              if belongs(k, task) or not any(belongs(k, t) for t in tasks)})
                if role == "calibration":
                    entry[f"rep_{task}"] = True
    columns = []
    for entry in merged.values():
        for column in entry:
            if column not in columns:
                columns.append(column)
    with path.open("w") as handle:
        handle.write("\t".join(columns) + "\n")
        for entry in merged.values():
            handle.write("\t".join(str(entry.get(column, "")) for column in columns) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=list(DATA))
    parser.add_argument("--tag", default="full")
    parser.add_argument("--tiers", nargs="+", default=list(TIERS), choices=TIERS)
    parser.add_argument("--adversary", default="four", choices=ADVERSARIES)
    parser.add_argument("--design", default="representatives", choices=("representatives", "split"))
    args = parser.parse_args()
    if args.adversary == "profile" and "pdb" in args.tiers:
        if args.tiers != list(TIERS):
            parser.error("the profile hits drop the query's own 30% cluster: tier mmseqs30 only")
        args.tiers = ["mmseqs30"]
    started = time.time()
    pool = Pool(args.dataset, extra=("profile",) if args.adversary == "profile" else ())
    pool.rules = RULES.get(args.adversary, BASE_RULES)
    train = records(DATA[args.dataset] / "train.jsonl")
    naive_full = {task: naive_prediction(train, task) for task in pool.tasks if task != "ec"}
    naive = {task: value[2] for task, value in naive_full.items()}
    design_of = Representatives(pool) if args.design == "representatives" else Split(pool)
    print("roles", round(time.time() - started), "s", flush=True)
    stem = prefix(args.dataset, args.adversary, args.design)
    result = {"dataset": args.dataset, "tag": args.tag, "design": f"HoLT-2 {args.design}", "protocol": PROTOCOL,
              "adversary": args.adversary, "attack_routes": ADVERSARIES[args.adversary], "rules": pool.rules,
              "alpha": ALPHA, "delta": DELTA, "path_quantiles": Q, "events_in_family": len(FAMILY),
              "family_of_label_sets": FAMILY, "per_test": PER_TEST,
              "zero_leak_min_n": {name: zero_min(ALPHA, level) for name, level in PER_TEST.items()},
              "families": {"total": int(pool.family.max() + 1), "largest": int(np.bincount(pool.family).max())},
              "tiers": {}}
    for tier in args.tiers:
        design, calibration, members, tasks = run(pool, design_of, tier, naive, ADVERSARIES[args.adversary])
        entry = {"naive": {task: {"design_fmax": val[0], "threshold": val[1], "label_count": len(val[2])}
                           for task, val in naive_full.items()},
                 # Tier counts cover every training chain: calibration chains of any label set, and the rest.
                 "calibration_n": int(len({row["index"] for rows in calibration.values() for row in rows})),
                 "design_n": int(len(pool.ids) - len({row["index"] for rows in calibration.values() for row in rows})),
                 **design_of.describe(tier), "tasks": tasks}
        if args.design == "split":
            entry["design_library_curve"] = library_curve(pool, design_of, tier, naive,
                                                          ADVERSARIES[args.adversary], tasks, design)
        result["tiers"][tier] = entry
        table = RAW / f"{stem}-{tier}-{args.tag}-chains.tsv"
        write_chains(table, design, calibration, members)
        print(table, round(time.time() - started), "s", flush=True)
    output = RAW / f"{stem}-{args.tag}-certification.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(output, round(time.time() - started), "s", flush=True)


if __name__ == "__main__":
    main()
