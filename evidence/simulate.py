"""Conditional outer calibration simulation of the frozen HoLT fixed sequence.

Every one of 10,000 outer trials draws independent design and calibration
samples, learns the logistic score, constructs the design quantile path, and
runs the fixed-sequence exact test. The population has a uniform covariate
and analytic risk, so every candidate's risk is known without test outcomes.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.stats import beta, binom
from sklearn.linear_model import LogisticRegression

from holt import _certify as cc
from holt._common import cp_upper


ALPHA = 0.1
GAMMA = cc.DELTA / len(cc.FAMILY)
HIGH = (ALPHA - .3 * .04) / .7
SCENARIOS = {
    "homogeneous_boundary": (lambda x: np.full_like(x, ALPHA), lambda x: ALPHA * x),
    "heterogeneous_boundary": (lambda x: ALPHA + .12 * (x - .5),
                               lambda x: ALPHA * x + .06 * (x * x - x)),
    "heterogeneous_all_above": (lambda x: ALPHA + .004 + .004 * (x - .5),
                                lambda x: (ALPHA + .004) * x + .002 * (x * x - x)),
    "all_near_above": (lambda x: np.full_like(x, ALPHA + .002),
                       lambda x: (ALPHA + .002) * x),
    "mixed_below_and_above": (lambda x: np.where(x < .3, .04, HIGH),
                              lambda x: .04 * np.minimum(x, .3) + HIGH * np.maximum(x - .3, 0)),
}


def exact_ci(k: int, n: int) -> list[float]:
    return [float(beta.ppf(.025, k, n - k + 1)) if k else 0.,
            float(beta.ppf(.975, k + 1, n - k)) if k < n else 1.]


def validate_equivalence():
    rng = np.random.default_rng(42)
    for n in rng.integers(72, 801, 100):
        k = int(rng.integers(0, n + 1))
        if (cp_upper(k, int(n), GAMMA) <= ALPHA) != (binom.cdf(k, int(n), ALPHA) <= GAMMA):
            raise AssertionError("exact lower tail differs from frozen UCB test")
    x = np.linspace(0., 1., 200001)
    for name, (pi, integral) in SCENARIOS.items():
        if abs(np.trapezoid(pi(x), x) - integral(1.)) >= 1e-6:
            raise AssertionError(f"analytic population risk integral differs: {name}")


def one(pi, integral, rng: np.random.Generator, trials: int, design_n: int,
        calibration_n: int) -> tuple[dict, np.ndarray]:
    design_x = rng.random((trials, design_n), dtype=np.float32)
    design_y = rng.random((trials, design_n), dtype=np.float32) < pi(design_x)
    # For one feature, the sign of the fitted penalized logistic coefficient
    # equals the design sample's x/y covariance sign. Its magnitude and
    # intercept cancel from every score-quantile membership comparison.
    covariance = ((design_x - design_x.mean(1, keepdims=True)) * design_y).sum(1)
    positive = covariance >= 0
    for i in range(min(4, trials)):
        fitted = LogisticRegression(C=1., max_iter=1000, tol=1e-12).fit(design_x[i, :, None], design_y[i])
        if (fitted.coef_[0, 0] >= 0) != bool(positive[i]):
            raise AssertionError("analytic logistic ordering differs from fitted score")
    qs = np.asarray(cc.Q) / 100
    positive_cut = np.quantile(design_x, qs, axis=1).T
    negative_cut = np.quantile(design_x, 1 - qs, axis=1).T
    positive_cut[:, -1] = np.nextafter(positive_cut[:, -1], np.inf)
    negative_cut[:, -1] = np.nextafter(negative_cut[:, -1], -np.inf)
    cuts = np.where(positive[:, None], positive_cut, negative_cut)
    risks = np.where(positive[:, None],
                     integral(cuts) / np.maximum(cuts, 1e-12),
                     (integral(1.) - integral(cuts)) / np.maximum(1. - cuts, 1e-12))
    x = rng.random((trials, calibration_n), dtype=np.float32)
    events = rng.random((trials, calibration_n), dtype=np.float32) < pi(x)
    n = np.empty((trials, len(cc.Q)), dtype=np.int32)
    k = np.empty_like(n)
    for j in range(len(cc.Q)):
        member = np.where(positive[:, None], x < cuts[:, j, None], x > cuts[:, j, None])
        n[:, j] = member.sum(axis=1)
        k[:, j] = (member & events).sum(axis=1)
    # In crossfit_certify, UCB <= alpha iff the exact lower tail <= gamma.
    passes = binom.cdf(k, n, ALPHA) <= GAMMA
    start_n = cc.zero_min(cc.START_ALPHA, GAMMA)
    starts = np.where(n >= start_n, np.arange(len(cc.Q)), len(cc.Q)).min(axis=1)
    indices = np.arange(len(cc.Q))[None, :]
    in_run = (indices >= starts[:, None]) & passes
    fail = (indices >= starts[:, None]) & ~passes
    first_fail = np.where(fail, indices, len(cc.Q)).min(axis=1)
    selected = np.where(in_run & (indices < first_fail[:, None]), indices, -1).max(axis=1)
    for i in range(min(5, trials)):
        trace = [{"n": int(n[i, j]), "ucb": 0. if passes[i, j] else 1.} for j in range(len(cc.Q))]
        if cc.fixed_sequence(trace, ALPHA, GAMMA) != (int(selected[i]) if selected[i] >= 0 else None):
            raise AssertionError("vectorized fixed sequence differs from frozen implementation")
    false = (selected >= 0) & (risks[np.arange(trials), np.maximum(selected, 0)] >= ALPHA - 1e-12)
    detail = {"design_events_total": int(design_y.sum()),
            "score_slope_positive_trials": int(positive.sum()),
            "path_quantiles": list(cc.Q),
            "candidate_risk_min": float(risks.min()), "candidate_risk_max": float(risks.max()),
            "start_n": start_n, "certifications": int((selected >= 0).sum()),
            "false_certifications": int(false.sum()), "false_rate": float(false.mean()),
            "false_ci95_exact": exact_ci(int(false.sum()), trials),
            "chosen_counts": np.bincount(selected + 1, minlength=13).tolist()}
    return detail, false


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=10000)
    parser.add_argument("--design-n", type=int, default=1200)
    parser.add_argument("--calibration-n", type=int, default=800)
    parser.add_argument("--seed", type=int, default=26101911)
    parser.add_argument("--output", type=Path, default=Path("synthetic-calibration-rerun.json"))
    args = parser.parse_args()
    if min(args.trials, args.design_n, args.calibration_n) < 1:
        parser.error("sample sizes must be positive")
    validate_equivalence()
    started = time.perf_counter()
    rng = np.random.default_rng(args.seed)
    scenarios = {}
    for name, (pi, integral) in SCENARIOS.items():
        outcomes = {task: one(pi, integral, rng, args.trials, args.design_n, args.calibration_n)
                    for task in ("EC", "BP", "MF", "CC")}
        tasks = {task: outcome[0] for task, outcome in outcomes.items()}
        any_false = np.logical_or.reduce([outcome[1] for outcome in outcomes.values()])
        k = int(any_false.sum())
        scenarios[name] = {"pi_min": float(pi(np.array([0., 1.])).min()),
                           "pi_mean": float(integral(1.)),
                           "pi_max": float(pi(np.array([0., 1.])).max()), "tasks": tasks,
                           "familywise_false_trials": k,
                           "familywise_false_rate": k / args.trials,
                           "familywise_ci95_exact": exact_ci(k, args.trials)}
    result = {"alpha": ALPHA, "delta": cc.DELTA, "label_sets": list(cc.FAMILY),
              "gamma": GAMMA, "seed": args.seed, "trials_per_scenario": args.trials,
              "design_n": args.design_n, "calibration_n": args.calibration_n,
              "risk_definition": "population mean of known Bernoulli pi over each nested score candidate; equality counted as false for conservative boundary diagnostic",
              "independence": "fresh independent Bernoulli design and calibration representatives for every trial and label set",
              "score_reduction": "one-feature penalized logistic score; sign of fitted slope equals design x/y covariance sign, while intercept and slope magnitude cancel from nested quantile membership",
              "scenarios": scenarios, "seconds": time.perf_counter() - started}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(args.output, result["seconds"])


if __name__ == "__main__":
    main()
