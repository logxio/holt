"""HoLT training calibration with family representatives and exact binomial tests.

Representatives are selected from hit lists with one chain per family and no
representative in another's top five annotated hits. Design chains exclude the
representatives, so the score and path do not read their labels. The fixed
sequence starts from a label-free path size and pays for every label set and
protocol compared on the same calibration labels.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import binom

from ._common import cp_upper
from ._engine import ROUTES, features, unit_hash


PROTOCOL = json.loads(Path(__file__).with_name("protocol.json").read_text())
Q = tuple(PROTOCOL["path"]["design_quantiles"])
COORDS = ("identity", "bits", "esm2", "esmc")
EVENTS = (("ec", "exact"), ("bp", "raw"), ("mf", "raw"), ("cc", "raw"))
FAMILY = tuple(PROTOCOL["events"]["family_of_label_sets"])
TIERS = ("pdb", "mmseqs30")
FEATURES = tuple(PROTOCOL["score"]["features"])
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
    """Check the finite binomial-tail premise for every calibrated candidate."""
    if not isinstance(n, int) or n < 0:
        raise ValueError(f"invalid representative count: N={n}")
    if not 0 < alpha <= .5:
        raise ValueError(f"alpha outside Lemma 3 domain: {alpha}")
    if not 0 < level <= .25:
        raise ValueError(f"test level outside Lemma 3 domain: {level}")
    if n == 0:
        return None
    margin = float(binom.cdf(math.floor(n * alpha), n, alpha)) - level
    if margin <= 0:
        raise ValueError(f"binomial tail position fails: N={n}, alpha={alpha}, level={level}")
    return margin


def learn(design_rows, task, key):
    """HoLT Learn step: logistic copy-risk score fitted on design chains only."""
    x = np.array([features(row, task) for row in design_rows], dtype=float)
    y = np.array([bool(row[key]) for row in design_rows], dtype=float)
    if not len(y) or len(set(y)) != 2:
        raise ValueError("design rows must contain both copy-event classes")
    mean, scale = x.mean(0), x.std(0) + 1e-12
    x = (x - mean) / scale
    n = len(y)

    def objective(parameters):
        weights, intercept = parameters[:-1], parameters[-1]
        # L-BFGS-B can probe very large trial vectors before backing off.
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            margin = x @ weights + intercept
            probability = expit(margin)
            loss = (np.logaddexp(0, margin).sum() - y @ margin + .5 * (weights @ weights)) / n
            gradient = np.r_[x.T @ (probability - y) + weights, (probability - y).sum()] / n
        return loss, gradient

    # This is the C=1 binary logistic objective used by the frozen fit. An
    # explicit average-loss stopping rule keeps the result independent of
    # scikit-learn's version-specific solver scaling.
    fit = minimize(objective, np.zeros(x.shape[1] + 1), method="L-BFGS-B", jac=True,
                   options={"gtol": 1e-4, "ftol": 2.220446049250313e-9, "maxiter": 2000})
    if not fit.success:
        raise RuntimeError(f"logistic fit did not converge: {fit.message}")
    if not np.all(np.isfinite(fit.x)):
        raise RuntimeError("logistic fit returned non-finite coefficients")
    return {"features": FEATURES, "mean": mean.tolist(), "scale": scale.tolist(),
            "coef": fit.x[:-1].tolist(), "intercept": float(fit.x[-1]),
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
        tail_margins = [assert_tail_position(int(n), float(a), float(level))
                        for a in (*PROTOCOL["levels"]["alpha"], ALPHA)]
        trace.append({"q": Q[i], "cut": cut, "n": n, "leaks": k, "rate": k / n if n else None,
                      "min_tail_margin": min((m for m in tail_margins if m is not None), default=None),
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
    # Maximum MMseqs2 identity, at any coverage, guards all three roles.
    # Filter before fitting the score or fixing the quantile path.
    design_rows = [row.copy() for row in design_rows if row["identity"] < .3]
    calibration = [row.copy() for row in calibration if row["identity"] < .3]
    members = [row.copy() for row in members if row["identity"] < .3] if members is not None else None
    key = f"leak_{task}_{event}"
    if not design_rows or len({bool(row[key]) for row in design_rows}) < 2:
        return {"unavailable": "guarded design lacks two outcome classes",
                "design_n": len(design_rows), "calibration_n": len(calibration)}
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
    """HoLT: calibration chains that never read each other's labels.

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


def run(pool, design_of, tier, naive, attackers):
    design, calibration, members = design_of.rows(tier, naive, attackers)
    tasks = {}
    for task, event in EVENTS:
        if task not in pool.tasks:
            continue
        tasks[f"{task}_{event}"] = certify(design[task], calibration[task],
                                           members[task] if members is not None else None, task, event)
    return design, calibration, members, tasks
