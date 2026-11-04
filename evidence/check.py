"""Recompute every number of the README's paper-to-code map from the TSVs beside this file (Python 3 standard library)."""
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent
PERIODS = ((2020, 2021), (2022, 2023), (2024, 2024), (2025, 2026))


def rows(name):
    return list(csv.DictReader((HERE / name).open(), delimiter="\t"))


def q90(n, alpha):
    """Smallest k with P(Bin(n, alpha) <= k) >= 0.9."""
    total, k = 0.0, 0
    while True:
        total += math.exp(math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
                          + k * math.log(alpha) + (n - k) * math.log1p(-alpha))
        if total >= 0.9 - 1e-12:
            return k
        k += 1


def above(chains, alpha):
    return bool(chains) and sum(c["homology_answered"] == "1" for c in chains) > q90(len(chains), alpha)


def pct(k, n, digits=1):
    """Percent rounded half up, as printed in the paper (477/720 = 66.25% prints 66.3%)."""
    return f"{k}/{n:,} ({math.floor(10 ** (digits + 2) * k / n + .5) / 10 ** digits:.{digits}f}%)"


# Sections 1 and 4.1, Figs. 1b and 3: the stated "<30%" EC bin.
audit = rows("bin-audit-ec.tsv")
complete = [r for r in audit if r["complete_ec"] == "1"]
print("[1] stated <30% EC bin: training hit >=30% identity over >=50% of the chain:",
      pct(sum(float(r["identity_query50"]) >= 0.3 for r in audit), len(audit)),
      "| homology answers chains with a complete EC number:",
      pct(sum(r["homology_answered"] == "1" for r in complete), len(complete)))

# Section 3.2, Theorem 1 and Appendix A: known-risk calibrations, and the delta/16 receipt.
edge = next(r for r in rows("synthetic-calibration.tsv") if r["scenario"] == "homogeneous_boundary")
print("[2] risk exactly alpha in every candidate: false certification",
      pct(int(edge["false_certifications"]), int(edge["trials"]), 2),
      f"[{100 * float(edge['ci95_low']):.2f}, {100 * float(edge['ci95_high']):.2f}]")
chains = rows("certified-chains.tsv")
cells = defaultdict(list)
for c in chains:
    cells[c["bound"], c["cohort"], c["task"], c["tier"], c["alpha"]].append(c)
for (bound, cohort, task, tier, alpha), kept in sorted(cells.items()):
    if bound == "delta16" and kept:
        print(f"    delta/16 {cohort} {task} {tier} alpha={alpha}: keeps {len(kept):,} of",
              f"{len(cells['delta4', cohort, task, tier, alpha]):,}")
for r in rows("ablation.tsv"):
    if r["variant"] == "delta16":
        full = next(x for x in rows("ablation.tsv") if x["variant"] == "full" and x["task"] == r["task"]
                    and x["tier"] == r["tier"] and x["alpha"] == r["alpha"])
        print(f"    delta/16 deepfri {r['task']} {r['tier']} alpha={r['alpha']}: keeps {r['certified']} of {full['certified']}")

# Sections 4.2 and 4.5, Table 1, Figs. 4 and 6: run `holt reproduce`.
print("[3] Table 1, Figs. 4 and 6 on the certified sets: run `holt reproduce` (DeepFRI 93 Fmax and 161 paired",
      "differences, post-2020 56 Fmax and 105 paired differences, each with its 2,000-draw family-block 95% interval)")

# Section 4.3 and Appendix Table A4: one component removed at a time.
ablation = rows("ablation.tsv")
removed = {(x["task"], x["tier"], x["alpha"]) for x in ablation if x["variant"] == "identity"}
held = {(r["task"], r["tier"], r["alpha"]) for r in ablation
        if r["variant"] == "full" and int(r["certified"]) and (r["task"], r["tier"], r["alpha"]) in removed}
for variant in ("identity", "no_guard", "hash_order"):
    left = sum(int(r["certified"]) > 0 for r in ablation
               if r["variant"] == variant and (r["task"], r["tier"], r["alpha"]) in held)
    print(f"[4] {variant}: certifies {left} of the {len(held)} cells the protocol certifies")

# Section 4.4, Fig. 5 and Appendix Table A1, with each run's level and variant.
runs = rows("transfer-runs.tsv")
for variant, alpha, tier in sorted({(r["variant"], r["alpha"], r["tier"]) for r in runs}):
    mine = [r for r in runs if (r["variant"], r["alpha"], r["tier"]) == (variant, alpha, tier)]
    certifying = [r for r in mine if int(r["certified"])]
    tail = sum(int(r["answered"]) > q90(int(r["certified"]), float(alpha)) for r in certifying)
    print(f"[5] {variant} {tier} alpha={alpha}: runs above the 90% binomial tail {tail}/{len(certifying)}")

# Section 4.5, Fig. 6, Appendix Table A10: chains released after May 2020, family tier, by initial release year.
for task, tier, alpha in (("bp", "mmseqs30", "0.1"), ("cc", "mmseqs30", "0.1"), ("mf", "mmseqs30", "0.15"),
                          ("bp", "pdb", "0.1"), ("mf", "pdb", "0.15")):
    kept = cells["delta4", "temporal-go", task, tier, alpha]
    seqs = {c["sequence"] for c in kept}
    hit = {c["sequence"] for c in kept if c["homology_answered"] == "1"}
    first = {}
    for c in sorted(kept, key=lambda c: (c["release_date"], c["id"])):
        first.setdefault(c["family"], c)
    exceed = {"chain": 0, "family": 0, "family_once": 0}
    for lo, hi in PERIODS:
        period = [c for c in kept if lo <= int(c["release_date"][:4]) <= hi]
        once = {}
        for c in sorted(period, key=lambda c: c["id"]):
            once.setdefault(c["family"], c)
        exceed["chain"] += above(period, float(alpha))
        exceed["family"] += above(list(once.values()), float(alpha))
        exceed["family_once"] += above([c for c in first.values() if lo <= int(c["release_date"][:4]) <= hi], float(alpha))
    print(f"[6] post-2020 {task} {tier} alpha={alpha}: {len(kept):,} certified, sequences answered {len(hit)}/{len(seqs)};",
          f"period cells above the 90% tail: chain by chain {exceed['chain']}/4, one chain per family {exceed['family']}/4,",
          f"each family once {exceed['family_once']}/4")

if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from verify import main as verify_all
verify_all()
