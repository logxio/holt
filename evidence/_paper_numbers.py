"""Every number in the paper, from the result files, in one place.

Reads results/raw/ (the audit of the DeepFRI release, the release scores with family-block intervals, the HoLT
certification and acceptance runs on the release, the post-2020 and ProteinShake runs) and the prepared sequence files,
and writes generated/numbers.tex (one macro per number used in the prose) and the bodies of the tables. Re-run after
the result files change; the text, the tables and abstract.txt follow. Scores printed in other papers are constants
below, each with the table it was copied from. Every sentence that states a comparison has its condition checked here.
"""
import csv
import hashlib
import json
import math
import re
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import numpy as np
from scipy.stats import beta, binom

HERE = Path(__file__).resolve().parent
ROOT = next(parent for parent in HERE.parents if (parent / ".git").exists())
RAW = ROOT / "results" / "raw"
OUT = HERE / "generated"

# ---------------------------------------------------------------------------
# Sources. Every number and every figure follows the names in this block.
# ---------------------------------------------------------------------------
# The DeepFRI release searched against itself: sizes, sequence and structure relatives, sibling chains, labels, the
# stated "<30%" bin, the strong-hit threshold b* and the alphabet.
AUDIT = {"EC": "deepfri-original-ec", "GO": "deepfri-original-go"}
STRUCT_AUDIT = AUDIT  # figures/ read the structure checks under this name; the release audit carries them
CALIBRATION = "fig-calibration/review-calibration"  # validation pairs behind b* on the release (scripts/fig_data.py calibration)
PREPARED = {"EC": ROOT / "data/prepared/deepfri/ec", "GO": ROOT / "data/prepared/deepfri/go"}
# The stated "<30%" bin by highest identity to training under four coverage rules, with the chains each rule moves
# below 30% (scripts/fig_data.py). The text counts only alignments covering at least half of the test chain.
COVERAGE = "fig-identity-coverage.json"
COVERAGE_RULE = "query50"
COVERAGE_RULES = (("any", "all alignments"), ("query50", "alignments covering $\\geq$50\\% of the test chain"),
                  ("both50", "alignments covering $\\geq$50\\% of both chains"), ("residues50", "alignments of $\\geq$50 residues"))
# The specimen of Figs. 1, 2 and A3: a test chain of the stated bin and its training homolog (scripts/fig_data.py).
SPECIMEN = "fig-specimen-5YTU-A.json"
# The EC archive redistributed with CDConv, and the archive runs (keys of the table-one migration).
RELEASED_EC = ROOT / "data/prepared/ec"
LETTERS = "ec-letters-trace.json"
ARCHIVE_SCORES, ARCHIVE_KEY = "hard-set-metrics.json", "ec_corrupted_redistributed_archive"
ESM_S_RUN = "published-esm-s-original-metrics.json"
# Release scores: Fmax and AUPR per method on the 95% set, the stated bin and DeepFRI HoLT-hard, Fmax intervals and
# paired differences over test families (scripts/paper_intervals.py), checked against the per-list scoring of the same
# certified lists by scripts/review_transfer_paired.py.
INTERVALS = "paper-intervals.json"
RECERTIFIED = "crossfit-repair-s13-comparison/original-paired.json"  # the 95% set and the stated bin, scored in its own pass
PREDICTIONS = {"Mmseqs": "mmseqs-deepfri-ann", "Foldseek": "foldseek-deepfri-ann", "EsmS": "published-esm-s-original",
               "Gcn": "published-deepfri-gcn-original"}
# HoLT on the release: the guarded protocol, each candidate tested at delta / |O| = delta / 4, as frozen and applied once
# (Series13 and Series14) and re-read by Series15 (scripts/review_transfer_selection.py, route historical): the freeze
# (per cohort, label set, tier and level: score, path, trace, start and selected candidate), its one application to the
# cached test features of DeepFRI's test set and the post-2020 chains (every certified chain with its family and whether
# homology answers it), and every method on each certified set with paired intervals over its families
# (ec/go/temporal-go-paired*.json). The same files hold route bonferroni4, the four combinations of guard and
# representative order counted as candidates and tested at delta / 16 (RECEIPT, Appendix A's one robustness sentence).
# The guarded candidate (GUARD, scripts/review_transfer_guard_freeze.py), its deployment record (DEPLOYED: the evaluated
# rows every application reads, and the sets Appendix C's event diagnostics recalibrate), and the guard's freeze record.
S15 = "crossfit-repair-s15"
ROUTE = "historical"
RECEIPT = "bonferroni4"
FREEZE = f"{S15}/freeze.json"
APPLIED = f"{S15}/acceptance.json"
APPLIED_CHAINS = f"{S15}/acceptance-chains.tsv"
PAIRED_TEMPORAL = f"{S15}/temporal-go-paired-final.json"
DEPLOYMENTS = f"{S15}/per-deployment.json"  # every route on the 100 pseudo-deployments per tier and the stress split
MF15_DEPLOYMENTS = f"{S15}/mf15-deployments-summary.json"  # the same runs read at alpha = 0.15 for MF
YEARS = f"{S15}/year-deployments-complete.json"  # the post-2020 sets split by initial release year
YEARS_ONCE = f"{S15}/year-one-family.json"  # the same periods with each family kept once, at its earliest release
GUARD = {key: f"crossfit-{key.lower()}-guard-candidate-frozen.json" for key in ("EC", "GO")}
GUARD_FREEZE = "crossfit-repair-s13-deployment-freeze.json"
DEPLOYED = {key: f"crossfit-repair-s13-{key.lower()}-acceptance.json" for key in ("EC", "GO")}
EVALUATED = {key: f"crossfit-repair-s13-{key.lower()}-evaluated-rows.jsonl" for key in ("EC", "GO")}
# The unguarded protocol, reported beside it: certification (families, path, protocol levels) and acceptance (the frozen
# rules on the test set, fixed rules released whole) against the homology adversary.
CERTIFIED = {key: f"crossfit-orig-{key.lower()}-homology-full-certification.json" for key in ("EC", "GO")}
ACCEPTED = {key: f"crossfit-orig-{key.lower()}-homology-acceptance.json" for key in ("EC", "GO")}
# The guarded protocol on the diagnostics first read without the guard (scripts/review_guarded_replays.py and
# review_guarded_events.py): phmmer added to the adversary (training side only), the GO event read two other ways, the
# fixed rules cut to the size of each certified set in their own order of strictness, and every ProteinShake split.
GUARDED_PROFILE = {key: f"guarded-series9/profile-{key.lower()}.json" for key in ("EC", "GO")}
GUARDED_EVENTS = "guarded-series9/event-go.json"
# Component ablations (Section 4.3, Appendix Table A-ablation): each component removed alone, fitted on the full training
# set and applied once to the test set, and run on the 100 pseudo-deployments per tier of seeds 100-199 (no guard,
# representatives in hash order, identity or Foldseek bits alone as the score); the same pass tests every label set at
# alpha = 0.1 and 0.15 at delta / 16, Appendix A's receipt for EC at 0.15 (scripts/review_ablation_*.py).
ABLATION = "review-ablation-s10r2-summary.json"
GUARDED_FIXED = "guarded-series9/fixed-original.json"
GUARDED_SHAKE = "guarded-series9/proteinshake-{dataset}-{split}.json"
HOLT_TIER, ENTRY_TIER = "mmseqs30", "pdb"  # family held out (the protocol's statement); PDB entry held out
# phmmer added alone (with the routes' original votes), the extra votes alone (no phmmer) and both, each a guarded
# protocol at delta / 4 with its own score and path on task-specific design features; an audit run after the test results
# were read (scripts/review_profile_factors_series11.py score). The GO family tier reads the features recomputed per label
# set ("fresh"); the entry tiers compute theirs per label set from their own representatives.
PROFILE_FACTORS = {("EC", HOLT_TIER): "profile-factors-series11-ec-scored.json",
                   ("EC", ENTRY_TIER): "profile-factors-series11-ec-pdb-scored.json",
                   ("GO", HOLT_TIER): "profile-factors-series11-orig-go-fresh-scored.json",
                   ("GO", ENTRY_TIER): "profile-factors-series11-orig-go-pdb-scored.json"}
# Known risk (Section 3.2, Appendix A): the fixed-sequence test on synthetic calibrations with independent units whose
# copying probabilities are known along a label-free score, five settings (scripts/review_synthetic_outer.py).
SYNTHETIC = "synthetic-outer-series11.json"
# The post-2020 sets with one chain per unique sequence, paired intervals over the families left, and the BP chains'
# terms and correct answers on both cohorts (Table 3, Section 4.6; scripts/review_temporal_dedup_series11.py).
DEDUP = "temporal-dedup-series11.json"
# Same-host query times (Table 1, Appendix E): 20 EC test chains of the release, five cold and five warm runs per route
# (scripts/review_timing_end_to_end.py); the host's cores as results/补实验-261012.md section 5 records them.
TIMING = "review-timing-end-to-end.json"
TIMING_CORES = 10
# Transfer to unseen chains (Section 4.4, Table 2, Appendix A): the guarded candidate chosen on exploration seeds 0-9
# (review_transfer_repair.py pilot), preregistered, and confirmed on 100 new pseudo-deployments per tier and on the stress
# split (seeds 100-199, review_transfer_repair_gate.py); the unguarded protocol's 100 runs per tier
# (review_transfer_original.py, review_transfer_summary.py) and stress split (--multifamily), and one stress-split BP
# chain in both libraries (review_transfer_case.py).
GATE_PROTOCOL = "crossfit-repair-confirmation-protocol.json"
GATE = "crossfit-repair-confirmation-gate.json"
TRANSFER = {key: f"crossfit-{key.lower()}-original-transfer-summary-0-100.json" for key in ("EC", "GO")}
STRESS = {key: f"crossfit-{key.lower()}-original-multifamily-summary-0-100.json" for key in ("EC", "GO")}
STRESS_CASE = "crossfit-go-original-multifamily-case-5ZZ8-r.json"
# The SIFTS file behind the post-2020 labels, as read by scripts/review_event_sensitivity.py.
EVENTS = "event-sensitivity-series1.json"
# Fig. 2: the chains each route answers, stage by stage, on the release's 95% set and stated bin (scripts/fig_data.py).
WATERFALL = "fig-waterfall-summary.json"
# Chains released after 2020: the cohort searched against the release's GO training chains (the unguarded application
# holds its sizes and phmmer's) and the cohort stages; the frozen rules' application to it is APPLIED, its scores
# PAIRED_TEMPORAL.
TEMPORAL_COHORT = "certify-temporal-orig-go-acceptance.json"
TEMPORAL_STAGE = "review-temporal-stage.json"
# Series15's method names in its paired files, by the stems of the tables.
S15_NAMES = {"Naive": "Naive", "Blast": "BLAST", "Diamond": "DiamondScore", "Mmseqs": "MMseqs2", "Foldseek": "Foldseek",
             "Folddisco": "Folddisco", "Four": "Four-route", "EsmS": "ESM-S", "Gcn": "GCN", "Cnn": "CNN"}
# ProteinShake: the two-route lower bounds on every split (the guarded protocol on each split is GUARDED_SHAKE).
SHAKE_BOUNDS = "certify-proteinshake-series10.json"
# The public repository (Code and data availability): every certified set and abstaining cell with its certificate, the
# per-model scores and paired differences that `holt reproduce` rebuilds on DeepFRI's sets.
REPO = ROOT / "holt/certified_sets"
# Papers that report EC or GO scores on this benchmark, as cited in the introduction.
BENCHMARK_USERS = ("lmgvp", "gearnet", "cdconv", "promptprotein", "esmgearnet", "protir", "esms", "saprot", "motil", "dpfunc")

SHAKE = (("ps-ec-random", "EC random"), ("ps-ec-sequence", "EC seq.\\ 70\\%"), ("ps-ec-structure", "EC struct.\\ 70\\%"),
         ("ps-go-random", "GO random"), ("ps-go-sequence", "GO seq.\\ 70\\%"), ("ps-go-structure", "GO struct.\\ 70\\%"))
TASK_KEYS = (("EC", "ec", "ec"), ("BP", "go", "bp"), ("MF", "go", "mf"), ("CC", "go", "cc"))  # (short, dataset, label set)
EVENT = {"ec": "ec_exact", "bp": "bp_raw", "mf": "mf_raw", "cc": "cc_raw"}
SUBSETS = (("Full", "full"), ("Bin", "official_lt30"), ("Holt", "holt_hard"))
HARD_MF = "mmseqs30@0.15"  # Table 1's hard column for MF, which abstains at alpha = 0.1: the family tier at 0.15
# The other certified cells of the release (Section 4.3, Table 3): PDB entry held out at alpha = 0.1 (EC and BP), and MF
# at alpha = 0.15 in both tiers.
CELL_SUBSETS = (("Entry", "pdb@0.1"), ("MFFifteen", "mmseqs30@0.15"), ("MFFifteenEntry", "pdb@0.15"))
HOMOLOGY_ROUTES = (("Blast", "B"), ("Diamond", "D"), ("Mmseqs", "M"), ("Foldseek", "F"))  # with the letter of Table 3

# Table 1; a method with no prediction for a label set (GearNet-Edge on GO) prints --.
TABLE_ROWS = (  # (stem, label, mark for trained on this split); \yr prints the year of the method's paper in refs.bib
    ("Four", "Four-route transfer", ""),
    ("Mmseqs", "\\quad MMseqs2 (seq.) \\yr{2017}", ""),
    ("Foldseek", "\\quad Foldseek (3D) \\yr{2024}", ""),
    ("MmFs", "\\quad MMseqs2 + Foldseek", ""),
    ("EsmTwo", "\\quad ESM-2 (PLM) \\yr{2023}", ""),
    ("EsmC", "\\quad ESM~C (PLM) \\yr{2024}", ""),
    ("Stack", "\\quad Stacked combiner$^\\dagger$", ""),
    ("Folddisco", "Folddisco \\yr{2026}~\\cite{folddisco}", ""),
    ("EsmS", "ESM-S 650M \\yr{2024}~\\cite{esms}", ""),
    ("Gcn", "DeepFRI GCN \\yr{2021}~\\cite{deepfri}", "$^\\ast$"),
    ("Cnn", "DeepFRI CNN \\yr{2021}~\\cite{deepfri}", "$^\\ast$"),
    ("GearnetEdge", "GearNet-Edge \\yr{2023}~\\cite{gearnet}", "$^\\ast$"),
    ("Naive", "Naive \\yr{2013}~\\cite{cafaone}", ""),
    ("Blast", "BLAST \\yr{1997}~\\cite{blast,cafaone}", ""),
    ("Diamond", "DiamondScore \\yr{2020}~\\cite{deepgoplus}", ""),
)
TABLE_PAIRS = (("Four-EsmS", "Four routes $-$ ESM-S"), ("Mmseqs-EsmS", "MMseqs2 $-$ ESM-S"), ("EsmS-Naive", "ESM-S $-$ Naive"),
               ("Gcn-Naive", "DeepFRI GCN $-$ Naive"))
# Table 3: every certified set, on DeepFRI's test set and after 2020, with ESM-S and each trained model against Naive and
# the strongest homology route. Rows: (set, label set, tier, alpha); MF, which abstains at alpha = 0.1 in both tiers,
# closes each group at 0.15.
CERT_ROWS = (("deepfri", "ec", HOLT_TIER, "0.1"), ("deepfri", "ec", ENTRY_TIER, "0.1"),
             ("deepfri", "bp", HOLT_TIER, "0.1"), ("deepfri", "cc", HOLT_TIER, "0.1"), ("deepfri", "bp", ENTRY_TIER, "0.1"),
             ("deepfri", "mf", HOLT_TIER, "0.15"),
             ("temporal", "bp", HOLT_TIER, "0.1"), ("temporal", "cc", HOLT_TIER, "0.1"), ("temporal", "bp", ENTRY_TIER, "0.1"),
             ("temporal", "mf", HOLT_TIER, "0.15"))

# Scores printed in other papers: (95% cutoff, "<30%" bin or None), each EC, GO-BP, GO-MF, GO-CC.
PUBLISHED = (
    ("Prompt", "PromptProtein", "protir", "T3", (88.8, 49.5, 67.7, 55.1), (76.5, 43.9, 57.7, 53.2)),
    ("EsmGearnet", "ESM-GearNet + SiamDiff", "esmgearnet", "T3", (89.7, 50.0, 68.2, 50.5), None),
    ("EsmGearnetMvc", "ESM-GearNet + Multiview Contrast", "esmgearnet", "T3", (89.6, 51.4, 68.3, 49.7), None),
    ("Protir", "ProtIR + CDConv", "protir", "T5", (88.5, 50.3, 67.8, 49.9), (76.9, 43.4, 56.7, 44.7)),
    ("EsmFt", "ESM-2 650M, fine-tuned", "protir", "T3", (87.7, 48.4, 66.1, 53.5), (76.3, 42.3, 56.3, 49.7)),
    ("Motil", "MotiL", "motil", "T3", (87.9, 44.4, 64.0, 48.9), None),
    ("GearnetMvc", "GearNet-Edge, Multiview Contrast", "gearnet", "T2, T6", (87.4, 49.0, 65.4, 48.8), (74.4, 43.6, 53.3, 45.9)),
    ("Cdconv", "CDConv", "protir", "T3", (82.0, 45.3, 65.4, 47.9), (63.4, 38.1, 53.3, 42.8)),
    ("Deepfri", "DeepFRI", "gearnet", "T2, T6", (63.1, 39.9, 46.5, 46.0), (47.0, 36.1, 37.4, 44.0)),
    ("ProtirMmseqs", "MMseqs retriever", "protir", "T3", (88.7, 44.4, 64.7, 33.2), (78.1, 32.3, 50.2, 23.7)),
    ("ProtirFoldseek", "Foldseek retriever", "protir", "T3", (88.5, 43.5, 65.1, 31.2), (78.1, 32.8, 52.5, 24.5)),
)
TRAINED = tuple(stem for stem, *_ in PUBLISHED if stem not in ("ProtirMmseqs", "ProtirFoldseek"))  # rows that are trained models
# EC Fmax printed only for the "<30%" bin: GearNet appendix Table 6 (printed p. 22), first of the five cutoffs.
PRINTED_BIN_EC = (("GearnetEdge", 62.5), ("EsmOneB", 73.7))
# AUPR GearNet prints for DeepFRI on the 95% set (appendix Table 7, p. 23), which Appendix E attributes to the official
# GCN checkpoint; with the Fmax row above, the twelve printed DeepFRI cells Section 4.2 sets against the release's CNN.
PRINTED_DEEPFRI_AUPR = (54.7, 28.2, 46.2, 36.3)

macros = {}
shares = {}  # certified share of a GO label set's chains: BP and CC, DeepFRI's test set and post-2020


def put(name, value):
    if not name.isalpha():
        raise ValueError(f"macro name {name!r} must be letters only")
    if name in macros:
        raise ValueError(f"macro {name} defined twice")
    macros[name] = str(value)


def load(name):
    return json.loads((RAW / name).read_text())


def pct(share, digits=1):
    return f"{100 * share:.{digits}f}"


def pct_of(n, total, digits=1):
    """n of total as a percentage, from the exact fraction, halves rounded up (477 of 720 is 66.3)."""
    return str((Decimal(100 * n) / Decimal(total)).quantize(Decimal(1).scaleb(-digits), rounding=ROUND_HALF_UP))


def word(n):
    """A count below ten, written out for the prose."""
    return ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine")[n]


def count(n):
    return f"{n:,}".replace(",", "{,}")


def level(alpha):
    """A smallest certifiable level, rounded up so that the printed level certifies."""
    return f"{math.ceil(round(alpha * 100, 6)) / 100:.2f}"


def signed(value, digits=1):
    return f"{value:+.{digits}f}".replace("-", "$-$")


def interval(low, high, digits=1):
    return "[" + ", ".join(f"{v:.{digits}f}".replace("-", "$-$") for v in (low, high)) + "]"


def chain_rows(audit):
    with (RAW / f"audit-{audit}-chains.tsv").open() as file:
        return list(csv.DictReader(file, delimiter="\t"))


def tsv(name):
    with (RAW / name).open() as file:
        return list(csv.DictReader(file, delimiter="\t"))


def fasta(path):
    sequences, name = {}, None
    with path.open() as file:
        for line in file:
            line = line.strip()
            if line.startswith(">"):
                name = line[1:].split()[0]
                sequences[name] = []
            elif name:
                sequences[name].append(line)
    return {name: "".join(parts) for name, parts in sequences.items()}


def records(path):
    with path.open() as file:
        return [json.loads(line) for line in file]


def check(claims, where):
    failed = [claim for claim, holds in claims.items() if not holds]
    if failed:
        raise AssertionError(f"{where} states what the result files no longer show: " + "; ".join(failed))


def audit_numbers():
    """Checks that read sequences, structures, chain names and labels: the release."""
    for key, audit in AUDIT.items():
        r = load(f"audit-{audit}.json")
        n = r["sizes"]["test"]
        for split in ("train", "valid", "test"):
            put(f"{key}{split.capitalize()}N", count(r["sizes"][split]))
        stated = [row for row in chain_rows(audit) if row["official_lt30"] == "True"]
        if len(stated) != r["official_lt30"]["n"]:
            raise AssertionError(f"{audit}: per-chain table and report disagree on the stated bin")
        put(f"{key}BinN", count(len(stated)))
        put(f"{key}BinPct", pct(len(stated) / n))
        calibration = r["calibration"]
        put(f"{key}StrongBits", f"{calibration['strong_bits']:.0f}")
        put(f"{key}TransferF", f"{calibration['f1_at_30pct_identity']:.2f}")
        put(f"{key}HardN", count(r["hard"]["n"]))
        put(f"{key}HardPct", pct(r["hard"]["share"]))
        put(f"{key}FoundPct", pct(1 - r["hard"]["share"]))
        put(f"{key}BinStrongN", count(r["official_lt30_but_strong_structure"]["n"]))
        put(f"{key}NearPct", pct(r["identity_by_coverage"]["ge_30"]["coverage_ge_80"]["share"]))  # Appendix Table A2, C1
    go = load(f"audit-{AUDIT['GO']}.json")
    for task in ("mf", "bp", "cc"):
        put(f"GOMiss{task.upper()}", pct(go["missing_labels"][task]["train_missing_share"], 0))
        put(f"GODensRatio{task.upper()}", f"{go['label_density'][task]['test_over_train']:.1f}")
    shake = [load(f"audit-{name}.json") for name, _ in SHAKE]
    put("ShakeHardLo", pct(min(r["hard"]["share"] for r in shake)))
    put("ShakeHardHi", pct(max(r["hard"]["share"] for r in shake)))
    put("ShakeECBrokenPct", pct(load("audit-ps-ec-sequence.json")["declared_cap_broken"]["share"]))


def coverage_numbers():
    """The stated "<30%" bin by highest identity to training, counting only alignments that cover at least half of the
    test chain; Appendix Table A-coverage with four rules; and what the rule moves below 30%."""
    report = load(COVERAGE)
    for key, dataset in (("EC", "ec"), ("GO", "go")):
        d = report["datasets"][dataset]
        with (RAW / f"fig-identity-coverage-{dataset}-chains.tsv").open() as file:
            identity = [float(row[f"identity_{COVERAGE_RULE}"]) for row in csv.DictReader(file, delimiter="\t")]
        stated = [row for row in chain_rows(AUDIT[key]) if row["official_lt30"] == "True"]
        if not len(identity) == d["n"] == len(stated):
            raise AssertionError(f"{COVERAGE} and the audit disagree on the stated {key} bin")
        for name, floor, field in (("Leak", 0.30, "ge30"), ("IdThirtyFive", 0.35, "ge35"),  # 35%: top of Rost's twilight zone
                                   ("IdFifty", 0.50, None), ("NinetyFive", 0.95, None)):
            hits = sum(value >= floor for value in identity)
            if field and hits != d[COVERAGE_RULE][field]["n"]:
                raise AssertionError(f"{key}: {hits} chains at {floor:.0%} or more, {COVERAGE} has {d[COVERAGE_RULE][field]['n']}")
            put(f"{key}Bin{name}N", count(hits))
            put(f"{key}Bin{name}Pct", pct_of(hits, len(identity)))
        dropped = d["dropped_below_30"]
        put(f"{key}DropN", count(dropped["n"]))
        put(f"{key}DropTagN", count(dropped["by_cause"]["tag"]))
        put(f"{key}DropPartN", count(dropped["by_cause"]["partial_chain_50_residues_or_more"]))
        hits = sorted(d[rule]["ge30"]["n"] for rule, _ in COVERAGE_RULES)
        put(f"{key}BinRuleLo", pct_of(hits[0], d["n"]))
        put(f"{key}BinRuleHi", pct_of(hits[-1], d["n"]))
    ec, go = (report["datasets"][k][COVERAGE_RULE]["ge30"]["share"] for k in ("ec", "go"))
    if abs(ec - 2 / 3) > 0.01 or abs(go - 1 / 2) > 0.01:
        raise AssertionError("the introduction and Fig. 1 say two in three EC and half the GO chains of the bin")
    body = []
    for rule, label in COVERAGE_RULES:
        cells = []
        for dataset in ("ec", "go"):
            d = report["datasets"][dataset]
            for field in ("ge30", "ge35"):
                text = pct_of(d[rule][field]["n"], d["n"])
                cells.append((f"\\textbf{{{text}}}" if rule == COVERAGE_RULE else text) + f" ({count(d[rule][field]['n'])})")
        body.append((f"\\textbf{{{label}}}" if rule == COVERAGE_RULE else label) + " & " + " & ".join(cells) + " \\\\")
    (OUT / "tab_coverage.tex").write_text("\n".join(body) + "\n")


def example_numbers():
    """The specimen: a test chain of the stated "<30%" bin, the training chain it aligns to on the release, the two
    substitutions between them, and the EC number MMseqs2, Foldseek, ESM-S and DeepFRI's GCN give it."""
    s = load(SPECIMEN)
    test, train, hit = s["chain"], s["neighbour"], s["identity"]["release"]
    if hit["query"] != test or hit["target"] != train:
        raise AssertionError(f"{SPECIMEN}: the release alignment is not {test} against {train}")
    if {row["id"]: row for row in chain_rows(AUDIT["EC"])}[test]["official_lt30"] != "True":
        raise AssertionError(f"{test} is no longer in the stated EC bin")
    # The alignment has no gap: the test chain after its tag and linker against the first copy in the training chain.
    query = s["sequences"]["deepfri_release"][len(s["sequences"]["release_prefix_before_coordinates"]):]
    target = fasta(PREPARED["EC"] / "train.fasta")[train][:len(query)]
    changes = [f"{a}{i}{b}" for i, (a, b) in enumerate(zip(query, target)) if a != b]  # residue numbers of mature SOD1
    if len(query) != int(hit["alnlen"]) or abs((len(query) - len(changes)) / len(query) - float(hit["fident"])) > 5e-4:
        raise AssertionError(f"{SPECIMEN}: the ungapped reading of the {test} alignment no longer matches MMseqs2")
    truth = next(json.loads(line) for line in (PREPARED["EC"] / "test.jsonl").open() if f'"{test}"' in line)["labels"]["ec"]
    number = max(truth, key=lambda t: ("-" not in t, t))
    for stem, prefix in PREDICTIONS.items():
        row = next(json.loads(line) for line in (RAW / f"{prefix}-ec-ec-predictions.jsonl").open() if f'"{test}"' in line)
        top = max(row["scores"].values())
        if row["scores"].get(number) != top:
            raise AssertionError(f"Section 1 says {stem} gives {test} its EC number {number}")
    put("ExampleTest", test)
    put("ExampleTrain", train)
    put("ExampleIdentity", pct(float(hit["fident"])))
    put("ExampleColumns", str(hit["alnlen"]))
    put("ExampleChanges", " and ".join(changes))
    put("ExampleEC", number)


def audit_table():
    final = {key: load(f"audit-{audit}.json") for key, audit in AUDIT.items()}
    shake = [load(f"audit-{name}.json") for name, _ in SHAKE]

    def cell(entry, with_n=False):
        if entry is None:
            return "--"
        return f"{pct(entry['share'])}" + (f" ({count(entry['n'])})" if with_n else "")

    coverage = load(COVERAGE)["datasets"]

    def stated(r):
        if r.get("official_lt30"):  # the headline rule of Section 4.1
            d = coverage[r["name"].rsplit("-", 1)[1]]
            return pct_of(d[COVERAGE_RULE]["ge30"]["n"], d["n"])
        return cell(r.get("declared_cap_broken"))

    def hard(r):
        return f"\\textbf{{{pct(r['hard']['share'])}}} ({count(r['hard']['n'])})"
    lines = [
        ("Test chains", lambda r: count(r["sizes"]["test"])),
        ("Training chains", lambda r: count(r["sizes"]["train"])),
        ("C1\\quad Sequence relative, $\\geq$30\\% identity", lambda r: cell(r["near_relatives"]["identity_ge_30"])),
        ("\\qquad\\ $\\geq$30\\%, alignment covers $\\geq$80\\% of both", lambda r: cell(r["identity_by_coverage"]["ge_30"]["coverage_ge_80"])),
        ("\\qquad\\ $\\geq$50\\% identity", lambda r: cell(r["near_relatives"]["identity_ge_50"])),
        ("\\qquad\\ $\\geq$95\\% identity", lambda r: cell(r["near_relatives"]["identity_ge_95"])),
        ("C2\\quad Strong-hit threshold $b^\\ast$ (Foldseek bits)", lambda r: f"{r['calibration']['strong_bits']:.0f}"),
        ("\\qquad\\ Structural relative, bits $\\geq b^\\ast$", lambda r: cell(r["near_relatives"].get("bits_ge_strong"))),
        ("\\qquad\\ of these, $<$30\\% identity", lambda r: cell(r["near_relatives"].get("strong_structure_below_30pct_identity"))),
        ("C3\\quad Sibling chain in training", lambda r: cell(r["siblings"]) if "deepfri" in r["name"] else "n/a"),
        ("\\qquad\\ Identical sequence in training", lambda r: cell(r["duplicates"])),
        ("C5\\quad Stated ceiling exceeded$^\\dagger$", stated),
        ("\\textbf{Hard set of fixed thresholds}", hard),
    ]
    body = []
    for title, getter in lines:
        cells = [getter(final[key]) for key in ("EC", "GO")] + [getter(r) for r in shake]
        body.append(title + " & " + " & ".join(cells) + " \\\\")
        if title == "Training chains":
            body.append("\\midrule")
    (OUT / "tab_audit.tex").write_text("\n".join(body) + "\n")


def integrity_numbers():
    """Check C6 on the archive redistributed with CDConv: its letters, its lengths, and ESM-S and MMseqs2 on it."""
    trace = load(LETTERS)["fasta"]
    source = trace["deepfri"]["all"]
    put("SourceECChains", count(source["chains"]))
    if trace["torchdrug"]["all"]["alphabet"] != source["alphabet"] or not all(load(LETTERS)["cdconv_equals_motil"].values()):
        raise AssertionError("the letter trace no longer supports the text of Section 4.5")
    package = load(f"audit-{AUDIT['EC']}.json")["alphabet"]["package"]["shares"]
    substitution = {"L": "I", "E": "K", "Y": "R"}
    letters, residues, chains, other, same_length = {}, 0, 0, 0, {}
    for split in ("train", "valid", "test"):
        released, original = fasta(RELEASED_EC / f"{split}.fasta"), fasta(PREPARED["EC"] / f"{split}.fasta")
        same_length[split] = 0
        for name, sequence in released.items():
            chains += 1
            residues += len(sequence)
            for letter in sequence:
                letters[letter] = letters.get(letter, 0) + 1
            full = original.get(name)
            if full is None or len(full) != len(sequence):
                continue
            if full != sequence and all(a == b or substitution.get(a) == b for a, b in zip(full, sequence)):
                same_length[split] += 1
            elif full != sequence:
                other += 1
    if any(letters.get(letter) for letter in "LEY") or other > 1:
        raise AssertionError("the released EC package no longer shows the letter pattern described in Section 4.5")
    put("ReleasedECChains", count(chains))
    put("ReleasedECResidues", f"{residues / 1e6:.1f}")
    for letter in "LEY":
        put(f"Source{letter}Pct", pct(package[letter]))
    put("SourceLEYPct", pct(sum(package[letter] for letter in "LEY"), 0))
    for letter in "IKR":  # what the rewritten letters become (Fig. A3b)
        put(f"Source{letter}Pct", pct(package[letter]))
        put(f"Released{letter}Pct", pct(letters[letter] / residues))
    for split in ("train", "valid", "test"):
        put(f"Subst{split.capitalize()}N", count(same_length[split]))
    # Fig. A3a, as fig_package.py picks it: the test chain of unchanged length with the most rewritten letters among
    # its first 46 residues.
    span, best = 46, None
    released, original = fasta(RELEASED_EC / "test.fasta"), fasta(PREPARED["EC"] / "test.fasta")
    for name in sorted(released):
        full = original.get(name)
        if full and len(full) == len(released[name]) and full != released[name]:
            hits = sum(letter in substitution for letter in full[:span])
            if best is None or hits > best[0]:
                best = (hits, name)
    put("PackageChain", best[1])
    put("PackageSpan", str(span))
    put("PackageRewrittenN", str(best[0]))

    run = load(ESM_S_RUN)
    put("EsmSECPrinted", f"{run['paper']['EC']:.1f}")
    archive = load(INTERVALS)["archive"]
    put("ArchiveChainsN", count(archive["chains"]))
    put("ArchiveEsmS", f"{100 * archive['archive']['fmax']:.1f}")
    put("ArchiveEsmSRelease", f"{100 * archive['release']['fmax']:.1f}")
    gain = archive["release_minus_archive"]
    put("ArchiveGain", signed(100 * gain["delta"]))
    put("ArchiveGainCI", interval(*(100 * v for v in gain["ci95"])))
    rows = {r["prefix"]: r["tasks"] for r in load(ARCHIVE_SCORES)["rows"] if r["rule"] == "ann"}
    if abs(rows["published-esm-s"][ARCHIVE_KEY]["full_fmax"] - archive["archive"]["fmax"]) > 0.002:
        raise AssertionError(f"{ARCHIVE_SCORES} and {INTERVALS} score ESM-S on the archive differently")
    put("MmseqsECArchive", f"{100 * rows['mmseqs-ann'][ARCHIVE_KEY]['full_fmax']:.1f}")
    check({"ESM-S gains on the release over the archive, interval above zero": gain["ci95"][0] > 0,
           "ESM-S on the archive within 0.2 of its printed EC score": abs(100 * archive["archive"]["fmax"] - run["paper"]["EC"]) < 0.2},
          "Section 4.5")


    equal = load("equal-length-ec-scores.json")
    put("EqualLengthRelease", f"{100 * equal['scores']['release']['fmax']:.1f}")
    put("EqualLengthArchive", f"{100 * equal['scores']['archive']['fmax']:.1f}")
    put("EqualLengthGain", f"{100 * equal['paired']['delta_fmax']:.1f}")
    put("EqualLengthGainCI", interval(*(100 * v for v in equal['paired']['ci95'])))


def published_numbers():
    best = [max(row[4][i] for row in PUBLISHED if row[0] in TRAINED) for i in range(4)]
    for (short, *_), value in zip(TASK_KEYS, best):
        put(f"BestPrinted{short}", f"{value:.1f}")
    for i, (short, *_) in enumerate(TASK_KEYS):
        put(f"BestPrintedBin{short}", f"{max(row[5][i] for row in PUBLISHED if row[0] in TRAINED and row[5]):.1f}")
    for stem, _, _, _, full, low in PUBLISHED:
        for (short, *_), value in zip(TASK_KEYS, full):
            put(f"{stem}{short}Printed", f"{value:.1f}")
        if low:
            for (short, *_), value in zip(TASK_KEYS, low):
                put(f"{stem}{short}PrintedBin", f"{value:.1f}")
    for stem, value in PRINTED_BIN_EC:
        put(f"{stem}ECPrintedBin", f"{value:.1f}")
    put("UsersN", str(len(BENCHMARK_USERS)))
    put("UsersCite", ",".join(BENCHMARK_USERS))


def release_numbers():
    """Release scores and paired differences over test families (Sections 1, 4.2 and 4.3, Table 1, Fig. 3)."""
    r, recert = load(INTERVALS), load(RECERTIFIED)
    put("IntervalReps", count(r["replicates"]))
    if r["replicates"] != recert["protocol"]["bootstrap"]["replicates"] or r["seed"] != recert["protocol"]["bootstrap"]["seed"]:
        raise AssertionError(f"{INTERVALS} and {RECERTIFIED} resample with different settings")
    for subset, field in SUBSETS + CELL_SUBSETS:
        for stem, tasks in r["point"].get(field, {}).items():
            for short, _, task in TASK_KEYS:
                if task in tasks:
                    put(f"{stem}{short}{subset}", f"{100 * tasks[task]['fmax']:.1f}")
        for pair, tasks in r["paired"].get(field, {}).items():
            for short, _, task in TASK_KEYS:
                if task in tasks:
                    entry = tasks[task]["fmax"]
                    stem = pair.replace("-", "")
                    put(f"{stem}{subset}{short}", signed(100 * entry["delta"]))
                    put(f"{stem}{subset}{short}Abs", f"{abs(100 * entry['delta']):.1f}")
                    put(f"{stem}{subset}{short}CI", interval(*(100 * v for v in entry["ci95"])))
    # One scoring: the per-list file and the intervals file agree on every point and on ESM-S minus each baseline.
    names = {"EsmS": "ESM-S", "Gcn": "DeepFRI GCN", "Cnn": "DeepFRI CNN", "Naive": "Naive", "Blast": "BLAST",
             "Diamond": "DiamondScore", "Mmseqs": "MMseqs2", "Foldseek": "Foldseek", "Folddisco": "Folddisco"}
    labels = {"ec": "EC", "bp": "GO-BP", "mf": "GO-MF", "cc": "GO-CC"}
    for stem, name in names.items():
        for _, _, task in TASK_KEYS:
            entry = recert["tasks"][labels[task]]["methods"][name]
            for subset, field in (("full", "full"), ("official_lt30", "official_lt30")):
                if entry[subset] and abs(entry[subset]["fmax"] - r["point"][field][stem][task]["fmax"]) > 1e-6:
                    raise AssertionError(f"{stem} {task} {subset}: {RECERTIFIED} and {INTERVALS} disagree")
    # The certified sets are checked against Series15's paired files by paper_intervals.py itself.

    def ci(pair, field, task):
        return r["paired"][field][pair][task]["fmax"]["ci95"]

    def point(stem, field, task, metric="fmax"):
        return r["point"][field][stem][task][metric]
    # The bounds that replace "level": the largest endpoint of the paired intervals the text groups together.
    put("FullTieGO", f"{max(abs(100 * v) for s in ('Mmseqs', 'Foldseek') for t in ('bp', 'mf') for v in ci(f'{s}-EsmS', 'full', t)):.1f}")
    put("BinTieGO", f"{max(abs(100 * v) for t in ('bp', 'mf', 'cc') for v in ci('EsmS-Foldseek', 'official_lt30', t)):.1f}")
    leads = [100 * r["paired"]["official_lt30"]["EsmS-Mmseqs"][t]["fmax"]["delta"] for t in ("bp", "mf", "cc")]
    put("EsmSMmseqsBinGOLo", f"{min(leads):.1f}")
    put("EsmSMmseqsBinGOHi", f"{max(leads):.1f}")
    routes = ("Blast", "Diamond", "Mmseqs", "Foldseek")
    put("EsmSRouteHoltMin", f"{min(100 * r['paired']['holt_hard'][f'EsmS-{s}'][t]['fmax']['delta'] for s in routes for t in ('bp', 'cc')):.1f}")
    # The four-route vote against ESM-S: ahead on every label set of the 95% set and the stated bin, level on HoLT-hard.
    four = [100 * r["paired"][f]["Four-EsmS"][t]["fmax"]["delta"] for f in ("full", "official_lt30") for t in ("ec", "bp", "mf", "cc")]
    put("FourEsmSLeadLo", f"{min(four):.1f}")
    put("FourEsmSLeadHi", f"{max(four):.1f}")
    # Fig. 3's Folddisco tray: ESM-S ahead on every label set of the 95% set.
    folddisco = [100 * r["paired"]["full"]["EsmS-Folddisco"][t]["fmax"]["delta"] for t in ("ec", "bp", "mf", "cc")]
    put("EsmSFolddiscoFullLo", f"{min(folddisco):.1f}")
    put("EsmSFolddiscoFullHi", f"{max(folddisco):.1f}")
    # The stacked combiner on the 95% set: the highest score of every re-scored method on EC, BP and CC; on MF the
    # four-route vote is ahead of it by a margin the text prints to two decimals.
    full = r["point"]["full"]
    for _, _, task in TASK_KEYS:
        top = max(full, key=lambda stem: full[stem][task]["fmax"] if task in full[stem] else -1)
        if top != ("Four" if task == "mf" else "Stack"):
            raise AssertionError(f"Sections 1 and 4.2 say the stacked combiner is highest on EC, BP and CC and the four-route vote on MF; {task}: {top}")
    gap = 100 * (full["Four"]["mf"]["fmax"] - full["Stack"]["mf"]["fmax"])
    if not 0 < gap < 0.05:
        raise AssertionError(f"the text says the four-route vote leads the stacked combiner on MF by 0.02 points; now {gap:.3f}")
    put("StackFourMFGap", f"{gap:.2f}")
    # GearNet-Edge, retrained by us, predicts EC alone: the same scoring in both files on the 95% set and the stated bin.
    for subset in ("full", "official_lt30"):
        if abs(recert["tasks"]["EC"]["methods"]["GearNet-Edge"][subset]["fmax"] - point("GearnetEdge", subset, "ec")) > 1e-6:
            raise AssertionError(f"GearNet-Edge EC {subset}: {RECERTIFIED} and {INTERVALS} disagree")
    gearnet = {"fmax": point("GearnetEdge", "official_lt30", "ec")}
    put("GearnetECBin", f"{100 * gearnet['fmax']:.1f}")
    # The DeepFRI row GearNet prints, Fmax on the 95% set and the stated bin and AUPR on the 95% set, against DeepFRI's
    # CNN and GCN re-scored on the release from their released weights.
    printed = {stem: (full, low) for stem, _, _, _, full, low in PUBLISHED}["Deepfri"]
    cells = [(field, task, "fmax", value) for field, row in (("full", printed[0]), ("official_lt30", printed[1]))
             for (_, _, task), value in zip(TASK_KEYS, row)]
    cells += [("full", task, "aupr", value) for (_, _, task), value in zip(TASK_KEYS, PRINTED_DEEPFRI_AUPR)]
    gaps = {stem: [abs(100 * point(stem, field, task, metric) - value) for field, task, metric, value in cells] for stem in ("Cnn", "Gcn")}
    put("PrintedDeepfriCellsN", count(len(cells)))
    put("CnnPrintedGap", f"{max(gaps['Cnn']):.2f}")
    put("GcnPrintedGapMin", f"{min(gaps['Gcn']):.1f}")
    best = [max(row[4][i] for row in PUBLISHED if row[0] in TRAINED) for i in range(4)]
    best_bin = max(row[5][0] for row in PUBLISHED if row[0] in TRAINED and row[5])
    check({
        "MMseqs2 and Foldseek above every printed EC score on the 95% set (MMseqs2) and the stated bin (both)":
            100 * point("Mmseqs", "full", "ec") > best[0] and min(100 * point(s, "official_lt30", "ec") for s in ("Mmseqs", "Foldseek")) > best_bin,
        "MMseqs2 and Foldseek lead ESM-S on EC of the 95% set and the bin, every interval above zero":
            min(ci(f"{s}-EsmS", f, "ec")[0] for s in ("Mmseqs", "Foldseek") for f in ("full", "official_lt30")) > 0,
        "on the stated EC bin MMseqs2 and Foldseek score above GearNet-Edge and DeepFRI's GCN":
            min(point(s, "official_lt30", "ec") for s in ("Mmseqs", "Foldseek")) > max(gearnet["fmax"], point("Gcn", "official_lt30", "ec")),
        "on CC of the 95% set Foldseek leads ESM-S and MMseqs2 trails it":
            ci("Foldseek-EsmS", "full", "cc")[0] > 0 and ci("Mmseqs-EsmS", "full", "cc")[1] < 0,
        "on the stated bin ESM-S leads MMseqs2 on BP, MF and CC, every interval above zero":
            all(ci("EsmS-Mmseqs", "official_lt30", t)[0] > 0 for t in ("bp", "mf", "cc")),
        "DeepFRI's GCN trails ESM-S on EC, BP and MF of the 95% set and the bin":
            all(ci("EsmS-Gcn", f, t)[0] > 0 for f in ("full", "official_lt30") for t in ("ec", "bp", "mf")),
        "on HoLT-hard ESM-S leads Naive, BLAST, DiamondScore, MMseqs2, Foldseek and Folddisco on BP and CC, and on MF at "
        "alpha = 0.15, every interval above zero":
            all(ci(f"EsmS-{s}", f, t)[0] > 0 for s in ("Naive", "Folddisco") + routes
                for f, t in (("holt_hard", "bp"), ("holt_hard", "cc"), ("mmseqs30@0.15", "mf"))),
        "on HoLT-hard DeepFRI's GCN and CNN lead Naive on all four label sets, every interval above zero":
            all(ci(f"{m}-Naive", f, t)[0] > 0 for m in ("Gcn", "Cnn")
                for f, t in (("holt_hard", "ec"), ("holt_hard", "bp"), ("holt_hard", "cc"), ("mmseqs30@0.15", "mf"))),
        "on EC HoLT-hard ESM-S leads MMseqs2 and Folddisco, every interval above zero":
            all(ci(f"EsmS-{s}", "holt_hard", "ec")[0] > 0 for s in ("Mmseqs", "Folddisco")),
        "the four-route vote leads ESM-S on every label set of the 95% set and the bin, every interval above zero":
            all(ci("Four-EsmS", f, t)[0] > 0 for f in ("full", "official_lt30") for t in ("ec", "bp", "mf", "cc")),
        "on HoLT-hard the four-route vote minus ESM-S spans zero on BP and CC, and on MF at alpha = 0.15":
            all(ci("Four-EsmS", f, t)[0] < 0 < ci("Four-EsmS", f, t)[1] for f, t in (("holt_hard", "bp"), ("holt_hard", "cc"), ("mmseqs30@0.15", "mf"))),
        "HoLT-hard holds EC, BP and CC chains and no MF chain at alpha = 0.1":
            not r["counts"]["mf"]["holt_hard"]["chains"] and all(r["counts"][t]["holt_hard"]["chains"] for t in ("ec", "bp", "cc")),
        "DeepFRI's CNN on the release reproduces every printed DeepFRI cell within 0.25 points; DeepFRI's GCN misses each by more than 2":
            max(gaps["Cnn"]) < 0.25 and min(gaps["Gcn"]) > 2,
        "ESM-S leads Folddisco on every label set of the 95% set and the stated bin, every interval above zero":
            all(ci("EsmS-Folddisco", f, t)[0] > 0 for f in ("full", "official_lt30") for t in ("ec", "bp", "mf", "cc")),
        "AUPR puts DeepFRI's GCN ahead of ESM-S on BP and CC of the 95% set, Fmax behind":
            all(point("Gcn", "full", t, "aupr") > point("EsmS", "full", t, "aupr") and point("Gcn", "full", t) < point("EsmS", "full", t) for t in ("bp", "cc")),
    }, "Sections 1, 4.2 and 4.3")


def scored_table(metric, printed_rows):
    """Table 1 (Fmax with intervals and the printed scores) and its AUPR twin in the appendix (points): the release's
    methods on the 95% set, the stated bin and DeepFRI HoLT-hard, best bold and second underlined per column."""
    r = load(INTERVALS)
    cells = {}  # (stem, subset, task index) -> (value, low, high) or "--"
    for stem, *_ in TABLE_ROWS:
        for i, (_, _, task) in enumerate(TASK_KEYS):
            for subset, field in SUBSETS:
                field = HARD_MF if (subset, task) == ("Holt", "mf") else field
                predicts = r["methods"].get(stem)
                if not r["counts"][task][field]["chains"] or (isinstance(predicts, dict) and task not in predicts):
                    cells[stem, subset, i] = "--"
                elif stem not in r["point"][field] or task not in r["point"][field][stem]:
                    raise ValueError(f"Table 1: {INTERVALS} scores no {stem} on {task} {field}")
                else:
                    value = 100 * r["point"][field][stem][task][metric]
                    low, high = (100 * v for v in r["ci95"][field][stem][task]["fmax"]) if metric == "fmax" else (None, None)
                    cells[stem, subset, i] = (value, low, high)
    rows = TABLE_ROWS
    if printed_rows:
        printed = {stem: (full, low) for stem, _, _, _, full, low in PUBLISHED}
        best = {"Full": [max(printed[s][0][i] for s in TRAINED) for i in range(4)],
                "Bin": [max(printed[s][1][i] for s in TRAINED if printed[s][1]) for i in range(4)]}
        for i in range(4):
            for subset, _ in SUBSETS:
                cells["Deepfri", subset, i] = (printed["Deepfri"][0 if subset == "Full" else 1][i], None, None) if subset != "Holt" else "--"
                cells["Best", subset, i] = (best[subset][i], None, None) if subset != "Holt" else "--"
        rows = rows + (("Deepfri", "DeepFRI, printed~\\cite{gearnet}", "$^\\ast$"), ("Best", "Best printed model$^{\\ddagger}$", ""))
    order = {}
    for subset, _ in SUBSETS:
        for i in range(4):
            values = sorted({round(e[0], 1) for (s, sub, j), e in cells.items() if sub == subset and j == i and isinstance(e, tuple)},
                            reverse=True)
            order[subset, i] = values[:2]

    def render(entry, subset, i):
        if entry == "--":
            return "--"
        value, low, high = entry
        text = f"{value:.1f}"
        if round(value, 1) == order[subset, i][0]:
            text = f"\\textbf{{{text}}}"
        elif len(order[subset, i]) > 1 and round(value, 1) == order[subset, i][1]:
            text = f"\\underline{{{text}}}"
        if low is not None:
            text = f"${text}^{{+{high - value:.1f}}}_{{-{value - low:.1f}}}$".replace("\\textbf", "\\mathbf")
        return text

    body = ["\\rowgroup{Re-scored on the same test chains}"]
    for stem, label, trained in rows:
        if stem == "Deepfri":
            body += ["\\midrule", "\\rowgroup{Printed in the cited papers}"]
        if stem in ("EsmS", "Naive"):
            body.append("\\midrule")
        line = [render(cells[stem, subset, i], subset, i) for i in range(4) for subset, _ in SUBSETS]
        body.append(f"{label}{trained} & " + " & ".join(line) + " \\\\")
    if printed_rows:
        body += ["\\midrule", "\\rowgroup{Paired differences on the same chains}"]
        for pair, label in TABLE_PAIRS:
            line = []
            for _, _, task in TASK_KEYS:
                for subset, field in SUBSETS:
                    field = HARD_MF if (subset, task) == ("Holt", "mf") else field
                    entry = r["paired"].get(field, {}).get(pair, {}).get(task)
                    if entry is None:
                        line.append("--")
                        continue
                    delta, low, high = (100 * v for v in (entry["fmax"]["delta"], *entry["fmax"]["ci95"]))
                    line.append(f"${delta:+.1f}^{{+{high - delta:.1f}}}_{{-{delta - low:.1f}}}$")
            body.append(f"{label} & " + " & ".join(line) + " \\\\")
    if printed_rows:
        # Give rows touching rules extra upper/lower clearance without breaking colour bands.
        data = [i for i, line in enumerate(body) if " & " in line]
        for i in data:
            before = i == 0 or " & " not in body[i - 1]
            after = i == len(body) - 1 or " & " not in body[i + 1]
            if before or after:
                body[i] = r"\tbrulestrut " + body[i]
    return body


def main_table():
    """Table 1 and Appendix Table A-AUPR."""
    (OUT / "tab_main.tex").write_text("\n".join(scored_table("fmax", True)) + "\n")
    (OUT / "tab_aupr.tex").write_text("\n".join(scored_table("aupr", False)) + "\n")


def applied(route=ROUTE):
    """One route's frozen cells and their one application, {(cohort, task, tier, alpha): cell} each, checked against the
    freeze the application names."""
    run, freeze = load(APPLIED), load(FREEZE)
    if Path(run["freeze"]["path"]).name != Path(FREEZE).name:
        raise AssertionError(f"{APPLIED} applies another freeze than {FREEZE}")

    def own(cells):
        return {tuple(key.split("/")[1:]): cell for key, cell in cells.items() if key.split("/")[0] == route}
    return own(freeze["cells"]), own(run["cells"])


def family_ci(rows, replicates, seed):
    """95% interval of the share homology answers among certified chains, over resampled families (the weights of
    Series15's block_draws: each family drawn with replacement, families sorted)."""
    groups = sorted({row["family"] for row in rows})
    at = {g: i for i, g in enumerate(groups)}
    n = np.bincount([at[row["family"]] for row in rows], minlength=len(groups))
    k = np.bincount([at[row["family"]] for row in rows], weights=[float(row["leak"]) for row in rows], minlength=len(groups))
    rng = np.random.default_rng(seed)
    weights = np.array([np.bincount(rng.integers(0, len(groups), len(groups)), minlength=len(groups)) for _ in range(replicates)])
    return np.quantile((weights @ k) / (weights @ n), [0.025, 0.975])


def zero_min(alpha, level):
    """Fewest representatives a candidate needs to pass with no event: (1 - alpha)^N <= level (crossfit_certify)."""
    return math.ceil(math.log(level) / math.log(1 - alpha))


def holt_numbers():
    """DeepFRI HoLT-hard (Section 4.3, Appendix Tables A-levels and A-fixed): the frozen rules' representatives, path and
    selected candidates in both tiers, its one application to the test set, the smallest certifiable level where a
    label set abstains, MF at alpha = 0.15, the unguarded protocol beside it, phmmer added to the guarded adversary at the
    same level, fixed rules released whole and cut to the size of each certified set, and the four routes on the
    certified chains."""
    cert = {key: load(name) for key, name in CERTIFIED.items()}  # the unguarded run: families and protocol constants
    accept = {key: load(name) for key, name in ACCEPTED.items()}  # the unguarded deployment, and fixed rules released whole
    guard = {key: load(name) for key, name in GUARD.items()}
    frozen, cells = applied()
    freeze = load(FREEZE)
    ec = cert["EC"]
    if not ec["alpha"] == ec["delta"] == 0.1 or ec["path_quantiles"] != cert["GO"]["path_quantiles"]:
        raise AssertionError("the two certification runs no longer share alpha = delta = 0.1 and one path")
    if {accept[k]["adversary"] for k in accept} != {"homology"}:
        raise AssertionError("the unguarded acceptance runs are not the homology adversary")
    for key in ("EC", "GO"):
        c = guard[key]["candidate"]
        if (c["guard"], c["calibration_alpha"], c["target_alpha"], c["delta"]) != ("identity30", 0.1, 0.1, 0.1):
            raise AssertionError(f"{GUARD[key]} is not the identity30 guard at alpha = delta = 0.1")
    if not load(GATE)["test_evaluation_permitted"]:
        raise AssertionError(f"{GATE} did not open the test set")
    alpha = 0.1
    gamma = ec["per_test"]["pooled"]  # the per-candidate level of the fixed sequence, delta / |O|
    if abs(gamma - ec["delta"] / ec["events_in_family"]) > 1e-12:
        raise AssertionError(f"{CERTIFIED['EC']} does not test each candidate at delta / |O|")
    put("HoltAlpha", f"{alpha:g}")
    put("CrossPathN", str(len(ec["path_quantiles"])))
    put("CrossEventsN", word(ec["events_in_family"]))
    put("CrossLevel", f"{gamma:.3f}")
    put("CrossBonfLevel", f"{gamma / len(ec['path_quantiles']):.5f}")
    put("CrossZeroN", str(zero_min(alpha, gamma)))
    put("CrossStartN", str(zero_min(0.05, gamma)))
    put("FreezeDate", load(GUARD_FREEZE)["created_at"][:10])
    # Appendix A's receipt: the four combinations of guard and representative order counted as candidates on the same
    # training labels, each candidate tested at delta / (4 |O|).
    protocols = freeze["multiplicity"]["protocols"]
    receipt = freeze["multiplicity"]["level"]
    if abs(receipt - gamma / len(protocols)) > 1e-12 or {p.split("/")[1] for p in protocols} != {"none", "identity30"} \
            or {p.split("/")[0] for p in protocols} != {"degree", "random"}:
        raise AssertionError(f"{FREEZE}'s {RECEIPT} route is not the four guard x order combinations at delta / 16")
    put("ReceiptK", word(len(protocols)))
    put("ReceiptLevel", f"{receipt:.5f}")
    _, paid_cells = applied(RECEIPT)
    _, mine = applied()
    names = {("ec", "ec", HOLT_TIER, "0.1"): "EC", ("ec", "ec", ENTRY_TIER, "0.1"): "EntryEC",
             ("go", "bp", HOLT_TIER, "0.1"): "BP", ("go", "bp", ENTRY_TIER, "0.1"): "EntryBP", ("go", "cc", HOLT_TIER, "0.1"): "CC",
             ("go", "mf", HOLT_TIER, "0.15"): "MFFifteen", ("go", "mf", ENTRY_TIER, "0.15"): "MFFifteenEntry",
             ("temporal-go", "bp", HOLT_TIER, "0.1"): "TemporalBP", ("temporal-go", "cc", HOLT_TIER, "0.1"): "TemporalCC",
             ("temporal-go", "bp", ENTRY_TIER, "0.1"): "TemporalEntryBP",
             ("temporal-go", "mf", HOLT_TIER, "0.15"): "TemporalMFFifteen", ("temporal-go", "mf", ENTRY_TIER, "0.15"): "TemporalMFFifteenEntry"}
    for key, name in names.items():
        paid, held = paid_cells[key], mine[key]
        if not {r["id"] for r in paid["rows"]} <= {r["id"] for r in held["rows"]} or not held["n"]:
            raise AssertionError(f"Appendix A says every set at delta / 16 is a subset of the one at delta / 4; {key}")
        put(f"Receipt{name}", count(paid["n"]))
    if any(paid_cells[key]["n"] for key in names if key[1] == "ec"):
        raise AssertionError("Appendix A says EC abstains at delta / 16 in both tiers at alpha = 0.1")
    kept = {key for key in names if key[1] != "ec" and paid_cells[key]["n"] == mine[key]["n"]}
    if {names[key] for key in kept} != {"EntryBP", "CC", "MFFifteenEntry", "TemporalCC", "TemporalEntryBP", "TemporalMFFifteenEntry"}:
        raise AssertionError("Appendix A says at delta / 16 the entry tiers of BP and MF and both CC sets keep every chain, "
                             "family-tier BP and MF lose some")
    for key in ("EC", "GO"):
        put(f"CrossFam{key}", count(cert[key]["families"]["total"]))
        put(f"CrossTrain{key}", count(cert[key]["tiers"][HOLT_TIER]["design_n"] + cert[key]["tiers"][HOLT_TIER]["calibration_n"]))

    def cohort(task):
        return "ec" if task == "ec" else "go"

    def rule(tier, task, at="0.1"):
        return frozen[cohort(task), task, tier, at]

    def size(n, k):
        return f"{count(n)} ({count(k)})" if n else "0"

    put("CrossDesignEC", count(rule(HOLT_TIER, "ec")["rule"]["design_n"]))
    design = [rule(HOLT_TIER, t)["rule"]["design_n"] for t in ("bp", "mf", "cc")]
    put("CrossDesignGOLo", count(min(design)))
    put("CrossDesignGOHi", count(max(design)))
    tags = {HOLT_TIER: "Cross", ENTRY_TIER: "Entry"}
    certified = {HOLT_TIER: [], ENTRY_TIER: []}
    floors = {}
    for tier, tag in tags.items():
        for short, _, task in TASK_KEYS:
            for at in ("0.1", "0.15") if task == "mf" else ("0.1",):
                f = rule(tier, task, at)
                trace = f["rule"]["trace"]
                old = guard["EC" if task == "ec" else "GO"]["tiers"][tier][EVENT[task]]["rule"]["trace"]
                if [(r["n"], r["leaks"]) for r in trace] != [(r["n"], r["leaks"]) for r in old] or f["rule"]["level"] != gamma:
                    raise AssertionError(f"{short} {tier}: the frozen rule tests another path than the guard, or at another level")
                if any(abs(r["ucb"] - float(beta.ppf(1 - gamma, r["leaks"] + 1, r["n"] - r["leaks"]))) > 1e-9 for r in trace if r["leaks"] < r["n"]):
                    raise AssertionError(f"{short} {tier}: the upper bounds are not exact at delta / 4")
                j0 = f["start"]
                if j0 != next(i for i, r in enumerate(trace) if r["n"] >= zero_min(0.05, gamma)):
                    raise AssertionError(f"{short} {tier}: the start is not the first candidate able to pass with no event at 0.05")
                passed = [i for i in range(j0, len(trace)) if all(trace[j]["ucb"] <= float(at) for j in range(j0, i + 1))]
                if (passed[-1] if passed else None) != f["index"]:
                    raise AssertionError(f"{short} {tier} {at}: the selected candidate is not the last that passes from the start")
                c = cells[cohort(task), task, tier, at]
                stem = tag + ("Fifteen" if at == "0.15" else "")
                put(f"{stem}Calib{short}", count(f["rule"]["calibration_n"]))
                if f["index"] is None:
                    if c["n"]:
                        raise AssertionError(f"{short} {tier}: an abstaining rule certified test chains")
                    floors[tier, short, at] = trace[j0]["ucb"]
                    put(f"{stem}MinAlpha{short}", level(trace[j0]["ucb"]))
                    continue
                if at == "0.1":
                    certified[tier].append(short)
                chosen = trace[f["index"]]
                if chosen != f["calibration"] or c["n"] == 0:
                    raise AssertionError(f"{short} {tier}: the application applied another candidate than the frozen one")
                put(f"{stem}SelQ{short}", str(chosen["q"]))
                put(f"{stem}SelN{short}", count(chosen["n"]))
                put(f"{stem}SelK{short}", word(chosen["leaks"]) if chosen["leaks"] < 10 else count(chosen["leaks"]))
                put(f"{stem}SelUpper{short}", pct(chosen["ucb"]))
    if certified != {HOLT_TIER: ["EC", "BP", "CC"], ENTRY_TIER: ["EC", "BP"]}:
        raise AssertionError(f"Section 4.3 says the family tier certifies EC, BP and CC and the entry tier EC and BP; now {certified}")
    put("MinAlphaMFLo", level(min(floors[t, "MF", "0.1"] for t in tags)))
    put("MinAlphaMFHi", level(max(floors[t, "MF", "0.1"] for t in tags)))
    for short, _, task in TASK_KEYS:  # the score's largest weights: route agreement and Foldseek bits, in the family tier
        weights = rule(HOLT_TIER, task)["rule"]["model"]
        top = {f for _, f in sorted(zip(weights["coef"], weights["features"]), key=lambda p: -abs(p[0]))[:2]}
        if top != {"panel_consensus", "log1p_bits"}:
            raise AssertionError(f"Section 4.3 says the largest weights fall on route agreement and Foldseek bits; {task}: {top}")

    # The one application on the test set, against the release's scoring of the same lists.
    intervals = load(INTERVALS)
    seed = load(f"{S15}/go-paired.json")["bootstrap"]["seed"]
    eligible = {short: cells[cohort(task), task, HOLT_TIER, "0.1"]["eligible"] for short, _, task in TASK_KEYS}
    put("HoltEligibleTestEC", count(eligible["EC"]))
    put("HoltEligibleTestGO", count(eligible["BP"]))
    if not eligible["BP"] == eligible["MF"] == eligible["CC"] == intervals["counts"]["bp"]["full"]["chains"]:
        raise AssertionError("every GO test chain carries all three ontologies")
    rows = {}
    for key in ("EC", "GO"):
        with (RAW / EVALUATED[key]).open() as file:
            rows.update({(key, row["id"]): row for row in map(json.loads, file)})
    total = 0
    for tier, tag, subset, at, tasks in ((HOLT_TIER, "HoltCert", "holt_hard", "0.1", certified[HOLT_TIER]),
                                         (ENTRY_TIER, "HoltEntry", f"{ENTRY_TIER}@0.1", "0.1", certified[ENTRY_TIER]),
                                         (HOLT_TIER, "MFFifteen", f"{HOLT_TIER}@0.15", "0.15", ["MF"]),
                                         (ENTRY_TIER, "MFFifteenEntry", f"{ENTRY_TIER}@0.15", "0.15", ["MF"])):
        for short in tasks:
            task = short.lower()
            c = cells[cohort(task), task, tier, at]
            counted = intervals["counts"][task][subset]
            if (c["n"], c["families"]) != (counted["chains"], counted["families"]) or c["k"] != sum(r["leak"] for r in c["rows"]):
                raise AssertionError(f"{short} {tier} {at}: {APPLIED} and {INTERVALS} hold different sets")
            stem = tag if tag.startswith("MF") else f"{tag}{short}"
            put(f"{stem}N", count(c["n"]))
            put(f"{stem}K", word(c["k"]) if c["k"] < 10 else count(c["k"]))
            put(f"{stem}Pct", pct_of(c["k"], c["n"]))
            put(f"{stem}CI", interval(*(100 * v for v in family_ci(c["rows"], intervals["replicates"], seed))))
            put(f"{stem}Share", pct_of(c["n"], eligible[short]))
            key = "EC" if task == "ec" else "GO"
            if any(rows[key, r["id"]]["identity"] >= 0.3 for r in c["rows"]):
                raise AssertionError(f"{short}: a certified chain has a training hit at 30% identity or more")
            if tier == HOLT_TIER and at == "0.1":
                total += c["n"]
                shares[short + "Holt"] = c["n"] / eligible[short]
                four = f"four_leak_{task}_1.0" if task == "ec" else f"four_leak_{task}_0.5"
                put(f"HoltCertFour{short}K", count(sum(rows[key, r["id"]][four] for r in c["rows"])))
    put("HoltCertAllN", count(total))
    put("RiskReps", count(intervals["replicates"]))  # family resamples behind every realized-rate interval
    # The confidence of one certified set (one tier, one adversary, one level) and of two together by a union bound.
    put("HoltDelta", f"{ec['delta']:g}")
    put("HoltConf", f"{100 * (1 - ec['delta']):.0f}")
    put("HoltConfJoint", f"{100 * (1 - 2 * ec['delta']):.0f}")

    # The unguarded protocol: what it certified on the test set.
    old = {(t, short): accept["EC" if task == "ec" else "GO"]["tiers"][t][EVENT[task]]["learned"]["pooled"]["0.1"]["test"]
           for t in tags for short, _, task in TASK_KEYS}
    for (t, short), test in old.items():
        if test["n"]:
            put(f"Unguarded{tags[t]}{short}N", count(test["n"]))
            put(f"Unguarded{tags[t]}{short}K", word(test["leaks"]) if test["leaks"] < 10 else count(test["leaks"]))
    # phmmer added to the guarded adversary, family tier, its path re-tested at delta / 4: every label set
    # abstains, read from the training side alone.
    levels = []
    for key, tasks in (("EC", ("ec",)), ("GO", ("bp", "mf", "cc"))):
        profile = load(GUARDED_PROFILE[key])
        if (profile["tier"], profile["alpha"], profile["test_labels_read"]) != (HOLT_TIER, alpha, False) or "phmmer" not in profile["adversary"]:
            raise AssertionError(f"{GUARDED_PROFILE[key]} is not phmmer added to the guarded family tier at alpha = 0.1")
        for task in tasks:
            trace = profile["cells"][task]["trace"]
            ucb = [float(beta.ppf(1 - gamma, r["leaks"] + 1, r["n"] - r["leaks"])) if r["leaks"] < r["n"] else 1.0 for r in trace]
            j0 = next(i for i, r in enumerate(trace) if r["n"] >= zero_min(0.05, gamma))
            if ucb[j0] <= alpha:
                raise AssertionError(f"Section 4.3 says phmmer leaves {task.upper()} without a certified set at alpha = 0.1")
            levels.append(ucb[j0])
    put("ProfileMinAlphaLo", level(min(levels)))
    put("ProfileMinAlphaHi", level(max(levels)))

    # Fixed rules released whole.
    rules = (("Bin", "official_lt30", "stated ``$<$30\\%'' bin"), ("Id", "measured_identity_lt30", "identity $<$30\\%"),
             ("Fixed", "identity30_plus_bstar", "identity $<$30\\% and Foldseek $<b^\\ast$"))
    whole = {}
    for short, _, task in TASK_KEYS:
        found = next(x["fixed"] for x in accept["EC" if task == "ec" else "GO"]["tiers"][HOLT_TIER][EVENT[task]]["learned"]["pooled"].values()
                     if x.get("fixed") and all("full" in x["fixed"].get(rule_, {}) for _, rule_, _ in rules))
        whole[short] = {rule_: found[rule_]["full"] for _, rule_, _ in rules}
        for stem, rule_, _ in rules:
            put(f"HoltAll{stem}{short}Pct", pct_of(whole[short][rule_]["leaks"], whole[short][rule_]["n"]))
    for stem, rule_, _ in rules:
        rates = [whole[s][rule_]["rate"] for s, *_ in TASK_KEYS]
        put(f"HoltAll{stem}Lo", pct(min(rates), 0))
        put(f"HoltAll{stem}Hi", pct(max(rates), 0))
    every = [whole[s][rule_]["rate"] for s, *_ in TASK_KEYS for _, rule_, _ in rules]
    put("HoltAllRulesLo", pct(min(every)))
    put("HoltAllRulesHi", pct(max(every)))
    put("FixedEligibleECN", count(whole["EC"]["identity30_plus_bstar"]["n"]))
    if len({whole[t]["identity30_plus_bstar"]["n"] for t in ("BP", "MF", "CC")}) != 1:
        raise AssertionError("Section 1 counts one set of GO test chains that pass the fixed thresholds")
    put("FixedEligibleGON", count(whole["BP"]["identity30_plus_bstar"]["n"]))
    stated = whole["EC"]["official_lt30"]
    put("BinHomECK", count(stated["leaks"]))
    put("BinHomECN", count(stated["n"]))
    put("BinHomECPct", pct_of(stated["leaks"], stated["n"]))
    for short in ("BP", "MF", "CC"):
        put(f"BinHom{short}Pct", pct_of(whole[short]["official_lt30"]["leaks"], whole[short]["official_lt30"]["n"]))
    body = []
    for stem, rule_, label in rules:
        cells_ = [f"{count(whole[s][rule_]['n'])} & {pct(whole[s][rule_]['rate'])}" for s, *_ in TASK_KEYS]
        body.append(f"{label} & " + " & ".join(cells_) + " \\\\")
    # The same rules cut, in their own order of strictness, to the size of each set HoLT certifies on the test set
    # (the ranking of scripts/review_guarded_replays.py fixed(), on the evaluated rows every application reads).
    bstar = {key: load(f"audit-{AUDIT[key]}.json")["calibration"]["strong_bits"] for key in ("EC", "GO")}

    def cut(task, n):
        key, event = ("EC", "leak_ec_1.0") if task == "ec" else ("GO", f"leak_{task}_0.5")
        pool = [r for (k, _), r in rows.items() if k == key and r[f"labels_{task}"] and (task != "ec" or r["eligible_ec_exact"])]
        identity = sorted((r for r in pool if r["identity"] < .3), key=lambda r: (r["identity"], r["id"]))
        both = sorted((r for r in identity if r["bits"] < bstar[key] and not r["sibling"] and not r["duplicate"]),
                      key=lambda r: (max(r["identity"] / .3, r["bits"] / bstar[key]), r["id"]))
        out = {}
        for name, ranked in (("measured_identity_lt30", identity), ("identity30_plus_bstar", both)):
            if len(ranked) < n:
                raise AssertionError(f"{name} keeps fewer {task} chains than HoLT certifies")
            out[name] = {"n": n, "leaks": sum(bool(r[event]) for r in ranked[:n])}
        return out
    replayed = load(GUARDED_FIXED)["cells"]["cc_mmseqs30_a0.1"]["fixed"]  # the replay's own cut of the unchanged CC set
    if {k: v["matched"]["leaks"] for k, v in replayed.items()} != {k: v["leaks"] for k, v in cut("cc", 64).items()}:
        raise AssertionError(f"the cut here and {GUARDED_FIXED} disagree on the CC set both hold")
    body.append("\\midrule")
    ratios = []
    for tier, name in ((HOLT_TIER, "families"), (ENTRY_TIER, "PDB entries")):
        lines = {"holt": [], "measured_identity_lt30": [], "identity30_plus_bstar": []}
        for short, _, task in TASK_KEYS:
            c = cells[cohort(task), task, tier, "0.1"]
            if not c["n"]:
                for r in lines:
                    lines[r].append("-- &")
                continue
            matched = cut(task, c["n"])
            lines["holt"].append(f"{count(c['n'])} & {pct_of(c['k'], c['n'])}")
            for rule_, got in matched.items():
                lines[rule_].append(f"{count(got['n'])} & {pct_of(got['leaks'], got['n'])}")
                if tier == HOLT_TIER:
                    put(f"Matched{'Id' if rule_ == 'measured_identity_lt30' else 'Fixed'}{short}K", count(got["leaks"]))
            ratios.append(matched["measured_identity_lt30"]["leaks"] / c["k"])
        body.append(f"\\holt{{}}, {name} held out & " + " & ".join(lines["holt"]) + " \\\\")
        for rule_, label in (("measured_identity_lt30", "identity $<$30\\%"), ("identity30_plus_bstar", "and Foldseek $<b^\\ast$")):
            body.append(f"\\quad {label}, same size & " + " & ".join(lines[rule_]) + " \\\\")
    put("MatchedRatioLo", f"{min(ratios):.1f}")
    put("MatchedRatioHi", f"{max(ratios):.0f}" if max(ratios) == round(max(ratios)) else f"{max(ratios):.1f}")
    (OUT / "tab_fixed.tex").write_text("\n".join(body) + "\n")

    # Appendix Table A-levels: per tier, the guarded representatives, the candidate that passes at delta / 4 (or the
    # smallest level that certifies), the test chains, and the unguarded protocol's test chains; MF also at 0.15.
    body = []
    for short, _, task in TASK_KEYS:
        for at in ("0.1", "0.15") if task == "mf" else ("0.1",):
            row = []
            for tier in tags:
                f, c = rule(tier, task, at), cells[cohort(task), task, tier, at]
                row.append(count(f["rule"]["calibration_n"]))
                if f["index"] is None:
                    row += [f"$\\geq${level(floors[tier, short, at])}", "0"]
                else:
                    chosen = f["rule"]["trace"][f["index"]]
                    row += [f"{chosen['leaks']}/{count(chosen['n'])} ({pct(chosen['ucb'])})", size(c["n"], c["k"])]
            row += [size(old[t, short]["n"], old[t, short]["leaks"]) if at == "0.1" else "--" for t in tags]
            label = short if short == "EC" else "GO-" + short
            body.append(f"{label}{', $\\alpha=0.15$' if at == '0.15' else ''} & " + " & ".join(row) + " \\\\")
    (OUT / "tab_levels.tex").write_text("\n".join(body) + "\n")

    # The realized rate on a finite set (Appendix A): n certified chains at risk exactly alpha.
    n = cells["go", "bp", HOLT_TIER, "0.1"]["n"]
    k = math.ceil(1.5 * alpha * n)
    put("BinomExampleN", str(n))
    put("BinomExampleK", str(k))
    put("BinomExampleP", f"{binom.sf(k - 1, n, alpha):.2f}")
    # Lemma 3's premise on every candidate of the frozen paths, at every level of the protocol and both per-test levels.
    margins = [binom.cdf(math.floor(r["n"] * a), r["n"], a) - level_
               for key, f in frozen.items() if key[3] == "0.1" for r in f["rule"]["trace"]
               for a in (0.05, 0.1, 0.15, 0.2, 0.3, 0.5) for level_ in (gamma, gamma / len(ec["path_quantiles"]))]
    if min(margins) <= 0:
        raise AssertionError("Appendix A says every candidate satisfies the premise of Lemma 3")
    put("TailChecksN", count(len(margins)))
    put("TailMarginMin", f"{math.floor(100 * min(margins)) / 100:.2f}")


def transfer_numbers():
    """Section 4.4, Table 2 and Appendix A: the guarded protocol chosen on exploration seeds 0-9, confirmed on 100 new
    pseudo-deployments per tier (seeds 100-199), each holding out training families (or one PDB entry of each) as
    pseudo-test chains that search all remaining training chains, and on the stress split that keeps every pseudo-test
    family in the library; the unguarded protocol's 100 runs and stress split beside it; DeepFRI's certified sets counted in
    families; the stress-split chain that shows the mechanism the guard removes."""
    alpha = 0.1
    gate = load(GATE)
    confirmed = {(c["scenario"], c["task"]): c for c in gate["cells"]}
    if not gate["pooled_gate_passed"] or {c["replicates"] for c in gate["cells"]} != {100}:
        raise AssertionError(f"Section 4.4 says 100 confirmation runs per tier pass the pooled gate; {GATE} disagrees")
    protocol = load(GATE_PROTOCOL)
    if protocol["confirm_replicates"] != [100, 200] or protocol["primary"]["guard"] != "identity30":
        raise AssertionError(f"Section 4.4 says the identity30 guard is confirmed on seeds 100 to 199; {GATE_PROTOCOL} disagrees")
    put("TransferRunsN", "100")
    if not all(c["nonempty"] and c["pooled_rate"] <= alpha for c in confirmed.values() if c["scenario"] in (HOLT_TIER, ENTRY_TIER)):
        raise AssertionError(f"Section 4.4 says the guarded protocol passed its preregistered gate; {GATE} disagrees")
    # The frozen rules on the same runs (Series15, per-deployment.json): per run, the certified chains homology answers
    # against the 90% quantile of Bin(N, alpha) for the N it certifies, chain by chain and one chain per family.
    runs = load(DEPLOYMENTS)
    mine = {(c["scenario"], c["task"]): c for c in runs["cells"] if c["method"] == ROUTE}
    if {c["runs"] for c in mine.values()} != {100}:
        raise AssertionError(f"Section 4.4 says 100 runs per tier; {DEPLOYMENTS} disagrees")
    mf15 = {c["tier"]: c for c in load(MF15_DEPLOYMENTS)["cells"] if c["method"] == ROUTE and c["alpha"] == 0.15}
    family = {short: mine[HOLT_TIER, EVENT[task]] for short, _, task in TASK_KEYS}
    entry = {short: mine[ENTRY_TIER, EVENT[task]] for short, _, task in TASK_KEYS}
    stress = {short: mine["stress", EVENT[task]] for short, _, task in TASK_KEYS}

    def share(c):
        return c["chain"]["k"] / c["chain"]["n"]
    every = list(family.values()) + list(entry.values())
    gated = {(s, t): c for (s, t), c in confirmed.items() if s in (HOLT_TIER, ENTRY_TIER, "stress")}
    if any((c["chain"]["n"], c["chain"]["k"]) != (gated[key]["n"], gated[key]["leaks"])
           for key, c in mine.items() if key in gated):
        raise AssertionError(f"{DEPLOYMENTS} and {GATE} read different confirmation runs")
    check({"every cell certifies a set in some run and keeps the pooled share at or below alpha, MF at 0.15 below 0.15":
               all(c["chain"]["nonempty"] and share(c) <= alpha for c in every + [stress["BP"]])
               and all(share(c) <= 0.15 for c in mf15.values())}, "Section 4.4 and Table 2")
    for short in family:
        put(f"TransferFam{short}Pct", pct(share(family[short])))
        put(f"TransferEntry{short}Pct", pct(share(entry[short])))
    put("TransferAllLo", pct(min(share(c) for c in every)))
    put("TransferAllHi", pct(max(share(c) for c in every)))
    put("TransferFifteenFamPct", pct(share(mf15[HOLT_TIER])))
    put("TransferFifteenEntryPct", pct(share(mf15[ENTRY_TIER])))

    def tally(group, unit="chain"):
        return f"{sum(c[unit]['exceed'] for c in group)}/{count(sum(c[unit]['nonempty'] for c in group))}"

    def tally_pct(group, unit="chain"):
        return pct_of(sum(c[unit]["exceed"] for c in group), sum(c[unit]["nonempty"] for c in group))
    put("TailFam", tally(family.values()))
    put("TailEntry", tally(entry.values()))
    put("TailFamPct", tally_pct(family.values()))
    put("TailEntryPct", tally_pct(entry.values()))
    put("TailStress", tally(stress.values()))
    put("TailFamOne", tally(family.values(), "family"))
    put("TailEntryOne", tally(entry.values(), "family"))
    put("TailFifteenFam", tally([mf15[HOLT_TIER]]))
    put("TailFifteenEntry", tally([mf15[ENTRY_TIER]]))
    beyond = [short for short, c in stress.items() if c["chain"]["exceed"]]
    if beyond != ["MF"] or stress["MF"]["chain"]["exceed"] != 1 or stress["MF"]["chain"]["n"] != 23 \
            or not 0 < float(tally_pct(family.values())) < float(tally_pct(entry.values())) < 10:
        raise AssertionError("Section 4.4 says one stress run passes the tail, on MF, and fewer than one run in ten per tier")
    worst = max(every, key=lambda c: c["chain"]["above_alpha"] / c["chain"]["nonempty"])
    if worst is not entry["MF"]:
        raise AssertionError("Section 4.4 names MF with entries held out as the cell whose runs most often pass alpha")
    put("TransferWorstAbove", word(worst["chain"]["above_alpha"]) if worst["chain"]["above_alpha"] < 10 else count(worst["chain"]["above_alpha"]))
    put("TransferWorstRuns", count(worst["chain"]["nonempty"]))
    st = stress["BP"]
    put("TransferStressBPK", count(st["chain"]["k"]))
    put("TransferStressBPN", count(st["chain"]["n"]))
    put("TransferStressBPPct", pct(share(st)))
    put("TransferStressBPRuns", count(st["chain"]["nonempty"]))
    for short in ("EC", "CC"):
        k = stress[short]["chain"]["k"]
        put(f"TransferStress{short}K", word(k) if k < 10 else count(k))
        put(f"TransferStress{short}N", count(stress[short]["chain"]["n"]))

    # The unguarded protocol: its 100 runs per tier and its stress split, which the guard was designed against.
    summary = {key: load(name) for key, name in TRANSFER.items()}
    old_stress = {key: load(name) for key, name in STRESS.items()}

    def old(scenario, task):
        return summary["EC" if task == "ec" else "GO"]["scenarios"][scenario][EVENT[task]]
    old_family = {short: old(HOLT_TIER, task) for short, _, task in TASK_KEYS}
    old_entry = {short: old(ENTRY_TIER, task) for short, _, task in TASK_KEYS}
    check({"without the guard the entry tier passes alpha on MF": old_entry["MF"]["pooled_rate"] > alpha}, "Section 4.4")
    put("UnguardedEntryMFPct", pct(old_entry["MF"]["pooled_rate"]))
    put("UnguardedEntryMFAbove", word(old_entry["MF"]["above_alpha_replicates"]))
    put("UnguardedEntryMFRuns", word(old_entry["MF"]["nonempty"]))
    s_out, s_in = (old_stress["GO"]["scenarios"][name]["bp_raw"] for name in ("family_excluded", "full_library"))
    if not s_in["pooled_rate"] > alpha > share(st):
        raise AssertionError("Section 4.4 says the BP stress split passes alpha without the guard and not with it")
    put("UnguardedStressBPK", count(s_in["leaks"]))
    put("UnguardedStressBPN", count(s_in["selected_chains_with_repetition"]))
    put("UnguardedStressBPPct", pct(s_in["pooled_rate"]))
    put("UnguardedStressBPMedian", word(round(s_in["nonempty_size_percentiles_2p5_50_97p5"][1])))
    put("UnguardedStressOutBPPct", pct(s_out["pooled_rate"]))
    case = load(STRESS_CASE)
    far, near = (case["scenarios"][name] for name in ("family_excluded", "full_library"))
    far_top, near_top = far["top_annotated"]["mmseqs"][0], near["top_annotated"]["mmseqs"][0]
    if far["leak"] or not near["leak"] or far_top["same_family"] or not near_top["same_family"]:
        raise AssertionError(f"Appendix A says {case['query']} is answered by its family member only when the family is kept")
    truth = set(case["truth"])
    if not set(far_top["labels"]) < truth or set(near_top["labels"]) != truth:
        raise AssertionError(f"Appendix A says the other family's hit carries a subset of {case['query']}'s terms, the member all")
    put("CaseChain", case["query"])
    put("CaseFar", far_top["id"])
    put("CaseNear", near_top["id"])
    put("CaseFarTerms", word(len(far_top["labels"])))
    put("CaseTerms", count(len(truth)))
    put("CaseFarF", f"{far['copy_f1']['mmseqs']['top1_f1']:.2f}")
    put("CaseNearF", f"{near['copy_f1']['mmseqs']['top1_f1']:.1f}")
    if not near["features"]["identity"] >= 0.3:
        raise AssertionError(f"Fig. 4c says the guard removes {case['query']} from the stress split")
    put("CaseNearIdentity", pct(near["features"]["identity"]))

    # Table 2: per label set, the frozen rules' runs in each scenario (runs that certify, and in parentheses those whose
    # count passes the binomial tail) and the pooled answered share; the unguarded protocol's pooled share; DeepFRI's
    # family-tier set, counted in families, with its interval over them. MF also at alpha = 0.15.
    _, cells = applied()
    intervals = load(INTERVALS)
    seed = load(f"{S15}/go-paired.json")["bootstrap"]["seed"]

    def runs_cell(c):
        if c is None or not c["chain"]["nonempty"]:
            return "-- & --"
        return f"{c['chain']['nonempty']} ({c['chain']['exceed']}) & {pct(share(c))}"
    body = []
    for short, _, task in TASK_KEYS:
        for at in ("0.1", "0.15") if task == "mf" else ("0.1",):
            if at == "0.1":
                row = [runs_cell(family[short]), runs_cell(entry[short]), runs_cell(stress[short])]
                row.append(" / ".join(pct(c["pooled_rate"]) if c["nonempty"] else "--" for c in (old_family[short], old_entry[short])))
            else:
                row = [runs_cell(mf15[HOLT_TIER]), runs_cell(mf15[ENTRY_TIER]), runs_cell(None), "--"]
            t = cells["ec" if task == "ec" else "go", task, HOLT_TIER, at]
            if t["n"]:
                subset = "holt_hard" if at == "0.1" else f"{HOLT_TIER}@{at}"
                kish = intervals["counts"][task][subset]["effective_families"]
                row += [f"{t['k']}/{t['n']}", f"{t['families']} ({kish:.0f})",
                        interval(*(100 * v for v in family_ci(t["rows"], intervals["replicates"], seed)))]
            else:
                row += ["\\multicolumn{3}{c}{$\\ncert=0$}"]
            label = (short if short == "EC" else "GO-" + short) + (", $\\alpha=0.15$" if at == "0.15" else "")
            body.append(f"{label} & " + " & ".join(row) + " \\\\")
    (OUT / "tab_transfer.tex").write_text("\n".join(body) + "\n")


def event_numbers():
    """Appendix C and Section 5: the GO event read as relative F1 and as one correct term, each recalibrated on the guarded
    protocol's representatives at delta / 4; the certified sets by training density tertile; the date of the SIFTS
    mapping behind the post-2020 labels (Section 3.4)."""
    ev = load(GUARDED_EVENTS)
    deployed = load(DEPLOYED["GO"])["cells"]
    _, cells = applied()
    if ev["alpha"] != 0.1 or "identity<0.3" not in ev["guard"]:
        raise AssertionError(f"{GUARDED_EVENTS} is not the guarded protocol at alpha = 0.1")
    for short, task in (("BP", "bp"), ("MF", "mf"), ("CC", "cc")):
        found = ev["tasks"][task]
        raw, relative, anyc = (found["events"][e]["risk"] for e in ("raw", "relative", "any_correct"))
        test = deployed[f"{task}_{found['tier']}_a0.1"]["risk"]
        mine = cells["go", task, found["tier"], "0.1"]
        if not (raw["n"], raw["answered"]) == (test["n"], test["leaks"]) == (mine["n"], mine["k"]):
            raise AssertionError(f"{short}: {GUARDED_EVENTS} reads another certified set than the frozen application")
        if found["tier"] != (ENTRY_TIER if short == "MF" else HOLT_TIER):
            raise AssertionError(f"Appendix C reads MF in the entry tier and BP and CC in the family tier; {short}: {found['tier']}")
        if not relative["n"] or relative["rate"] >= 0.1:
            raise AssertionError(f"Appendix C says relative F1 certifies {short} with an answered share below alpha")
        put(f"Relative{short}N", count(relative["n"]))
        put(f"Relative{short}K", word(relative["answered"]) if relative["answered"] < 10 else count(relative["answered"]))
        if short == "MF":
            if raw["n"] or anyc["n"]:
                raise AssertionError("Appendix C says MF abstains at F1 >= 0.5 and with one correct term")
            continue
        put(f"AnyCorrect{short}N", count(anyc["n"]))
        put(f"AnyCorrect{short}K", word(anyc["answered"]) if anyc["answered"] < 10 else count(anyc["answered"]))
        put(f"AnyCorrect{short}Pct", pct_of(anyc["answered"], anyc["n"]))
        put(f"AnyCorrect{short}CI", interval(*(100 * v for v in anyc["exact95"])))
        edges = ev["density_edges"][task]
        put(f"DensEdges{short}", f"{edges[0]} and {edges[1]}")
        strata = raw["strata"]
        if sum(x["n"] for x in strata) != raw["n"] or max(strata, key=lambda x: x["n"]) is not strata[-1]:
            raise AssertionError(f"Appendix C says the dense tertile holds the largest share of the {short} set")
        put(f"DensCert{short}", "/".join(str(x["n"]) for x in strata))
        put(f"DensAnswered{short}", "/".join(str(x["answered"]) for x in strata))
    check({"one correct term: BP answered above alpha, CC below": float(macros["AnyCorrectBPPct"]) > 10 > float(macros["AnyCorrectCCPct"])},
          "Appendix C and Section 5")
    stamp = re.match(r"# (\d{4})/(\d{2})/(\d{2})", load(EVENTS)["sifts"]["header"])
    months = "January February March April May June July August September October November December".split()
    put("SiftsDate", f"{int(stamp.group(3))} {months[int(stamp.group(2)) - 1]} {stamp.group(1)}")


def ablation_numbers():
    """Section 4.3, Appendix Table A-ablation and Appendix A's receipt: each component of the protocol removed alone (no
    guard; representatives in hash order of the chain name; identity or Foldseek bits alone as the score, each with its own
    path at the design percentiles), fitted on the full training set and applied once to the test set, with the runs of
    the 100 pseudo-deployments per tier that certify a set; and EC at alpha = 0.15 tested at delta / 16."""
    ab = load(ABLATION)
    alphas = {"ec": 0.1, "bp": 0.1, "mf": 0.15, "cc": 0.1}
    if f"{ab['protocol']['level']:.3f}" != macros["CrossLevel"] or ab["protocol"]["alpha"] != alphas:
        raise AssertionError(f"{ABLATION} is not the protocol at delta / 4, MF at alpha = 0.15 and the rest at 0.1")
    for source in ("features", "original", "robustness"):
        path = ROOT / ab["sources"][source]["path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != ab["sources"][source]["sha256"]:
            raise AssertionError(f"{ABLATION} summarizes another {path.name} than the one on disk")
    frozen, cells = applied()
    accept = {key: load(name) for key, name in ACCEPTED.items()}
    runs = {(c["scenario"], c["task"]): c for c in load(DEPLOYMENTS)["cells"] if c["method"] == ROUTE}
    mf15 = {c["tier"]: c for c in load(MF15_DEPLOYMENTS)["cells"] if c["method"] == ROUTE and c["alpha"] == 0.15}
    features = load(Path(ab["sources"]["features"]["path"]).name)["cells"]
    variants = (("full", "degree/identity30"), ("no_guard", "degree/none"), ("hash_order", "random/identity30"),
                ("identity", "identity"), ("bits", "bits"))
    names = {"ec": "EC", "bp": "GO-BP", "mf": "GO-MF", "cc": "GO-CC"}
    found, body = {}, []
    for tier, label in ((HOLT_TIER, "Families held out"), (ENTRY_TIER, "PDB entries held out")):
        body.append(f"\\rowgroupn{{6}}{{{label}}}")
        for _, dataset, task in TASK_KEYS:
            at = alphas[task]
            test, deployed = ab["component_ablation"][f"{task}/{tier}"], ab["confirmed100"]
            mine = cells[dataset, task, tier, str(at)]
            old = accept[dataset.upper()]["tiers"][tier][EVENT[task]]["learned"]["pooled"][str(at)]["test"]
            s15 = (mf15[tier] if task == "mf" else runs[tier, EVENT[task]])["chain"]["nonempty"]
            if test["alpha"] != at or (test["full"]["n"], test["full"]["answered"]) != (mine["n"], mine["k"]) \
                    or (test["no_guard"]["n"], test["no_guard"]["answered"]) != (old["n"], old["leaks"]) \
                    or deployed[f"{task}/{tier}/degree/identity30"]["chain"]["nonempty"] != s15:
                raise AssertionError(f"{ABLATION} {task} {tier}: another protocol, unguarded run or deployment than the paper's")
            if features[f"{task}/{tier}"]["calibration_n"] != frozen[dataset, task, tier, str(at)]["rule"]["calibration_n"]:
                raise AssertionError(f"{ABLATION} {task} {tier}: the single features use other representatives than the frozen rule")
            row = []
            for variant, key in variants:
                c, r = test[variant], deployed[f"{task}/{tier}/{key}"]
                if r["runs"] != 100 or r["alpha"] != at:
                    raise AssertionError(f"{ABLATION} {task} {tier} {variant}: not 100 runs at alpha = {at}")
                found[variant, task, tier] = c["n"]
                answer = f"{c['answered']}/{count(c['n'])}" if c["n"] else r"\xmark{}"
                row.append(f"\\ablcell{{{answer}}}{{{r['chain']['nonempty']}}}")
            body.append(f"{names[task]}" + (", $\\alpha=0.15$" if at == 0.15 else "") + " & " + " & ".join(row) + " \\\\")
    (OUT / "tab_ablation.tex").write_text("\n".join(body) + "\n")
    held = {(t, tier) for (v, t, tier), n in found.items() if v == "full" and n}

    def empties(variant):
        return {(t, tier) for t, tier in held if not found[variant, t, tier]}
    put("AblCellsN", word(len(held)))
    put("AblNoGuardEmptyN", word(len(empties("no_guard"))))
    put("AblHashEmptyN", word(len(empties("hash_order"))))
    put("AblBitsEntryBPN", count(found["bits", "bp", ENTRY_TIER]))
    check({"identity alone abstains in every cell": not any(n for (v, *_), n in found.items() if v == "identity"),
           "Foldseek bits alone empty exactly the two EC cells": empties("bits") == {("ec", HOLT_TIER), ("ec", ENTRY_TIER)},
           "Foldseek bits alone certify more BP chains with entries held out than the protocol":
               found["bits", "bp", ENTRY_TIER] > found["full", "bp", ENTRY_TIER],
           "without the guard the entry tier certifies more EC chains than the protocol":
               found["no_guard", "ec", ENTRY_TIER] > found["full", "ec", ENTRY_TIER],
           "no variant certifies CC with entries held out, nor does the protocol":
               not any(n for (v, t, tier), n in found.items() if (t, tier) == ("cc", ENTRY_TIER))},
          "Section 4.3 and Appendix Table A-ablation")

    # Appendix A's receipt: the same pass at delta / 16, checked against Series15's two routes where both state a cell.
    robust = ab["robustness"]
    _, paid = applied(RECEIPT)
    for key, cell in robust.items():
        task, tier, at = key.split("/")
        here = ("ec" if task == "ec" else "go", task, tier, at)
        if not cell["subset"] or here in cells and ((cell["main"]["n"], cell["main"]["answered"]) != (cells[here]["n"], cells[here]["k"])
                                                    or (cell["robust"]["n"], cell["robust"]["answered"]) != (paid[here]["n"], paid[here]["k"])):
            raise AssertionError(f"{ABLATION} {key}: another set at delta / 4 or delta / 16 than Series15's, or not a subset")
    for tier, tag in ((HOLT_TIER, ""), (ENTRY_TIER, "Entry")):
        main, kept = robust[f"ec/{tier}/0.15"]["main"], robust[f"ec/{tier}/0.15"]["robust"]
        if not kept["n"]:
            raise AssertionError(f"Appendix A says EC at alpha = 0.15 keeps a set at delta / 16; {tier} abstains")
        put(f"ReceiptECFifteen{tag}", count(kept["n"]))
        put(f"ReceiptECFifteen{tag}K", word(kept["answered"]) if kept["answered"] < 10 else count(kept["answered"]))
        put(f"ReceiptECFifteen{tag}From", count(main["n"]))
    entry = robust[f"ec/{ENTRY_TIER}/0.15"]["robust"]
    put("ReceiptECFifteenEntryPct", pct_of(entry["answered"], entry["n"]))
    check({"with entries held out EC at alpha = 0.15 answers more than alpha of its delta / 16 set, within the 90% quantile of Bin(N, 0.15)":
               0.15 * entry["n"] < entry["answered"] <= binom.ppf(0.9, entry["n"], 0.15)}, "Appendix A")


def synthetic_numbers():
    """Section 3.2 and Appendix A: the fixed-sequence test of the protocol (its path, start, level gamma and four label
    sets) on synthetic calibrations with independent units of known copying probability along a label-free score; a draw
    certifies falsely when some label set returns a candidate whose true risk is alpha or more."""
    sim = load(SYNTHETIC)
    path = load(CERTIFIED["EC"])["path_quantiles"]
    if (sim["alpha"], f"{sim['delta']:g}", f"{sim['gamma']:.3f}", len(sim["label_sets"])) != (0.1, macros["HoltDelta"], macros["CrossLevel"], 4):
        raise AssertionError(f"{SYNTHETIC} does not test four label sets at alpha = delta = 0.1 and delta / 4")
    runs = sim["scenarios"]
    for name, run in runs.items():
        if any(t["path_quantiles"] != path or str(t["start_n"]) != macros["CrossStartN"] for t in run["tasks"].values()):
            raise AssertionError(f"{SYNTHETIC} {name}: another path or start than the protocol's")
    put("SynthTrialsN", count(sim["trials_per_scenario"]))
    put("SynthDesignN", count(sim["design_n"]))
    put("SynthCalibN", count(sim["calibration_n"]))
    edge = runs["homogeneous_boundary"]
    put("SynthEdgeK", count(edge["familywise_false_trials"]))
    put("SynthEdgePct", pct(edge["familywise_false_rate"], 2))
    put("SynthEdgeCI", interval(*(100 * v for v in edge["familywise_ci95_exact"]), 2))
    put("SynthEdgeTaskMax", pct(max(t["false_rate"] for t in edge["tasks"].values()), 2))
    put("SynthNearPct", pct(runs["all_near_above"]["familywise_false_rate"], 2))
    put("SynthRisingPct", pct(runs["heterogeneous_all_above"]["familywise_false_rate"], 2))

    def span(run):
        return "--".join(f"{100 * run[k]:.1f}".removesuffix(".0") for k in ("pi_min", "pi_max"))
    put("SynthNearPi", f"{100 * runs['all_near_above']['pi_mean']:.1f}")
    put("SynthRisingPi", span(runs["heterogeneous_all_above"]))
    put("SynthBelowPi", span(runs["heterogeneous_boundary"]))
    put("SynthMixedPi", span(runs["mixed_below_and_above"]))
    gamma, delta = sim["gamma"], sim["delta"]
    check({"at pi = alpha everywhere the false rate and its upper bound lie below delta, each label set's below gamma":
               edge["familywise_ci95_exact"][1] < delta and all(t["false_rate"] < gamma for t in edge["tasks"].values())
               and all(t["pi_min"] == t["pi_max"] == 0.1 for t in (edge,)),
           "with pi above alpha everywhere the false rate lies below delta":
               all(runs[k]["pi_min"] > 0.1 and runs[k]["familywise_ci95_exact"][1] < delta for k in ("all_near_above", "heterogeneous_all_above")),
           "where pi starts below alpha no draw certifies falsely":
               all(runs[k]["pi_min"] < 0.1 and runs[k]["familywise_false_trials"] == 0 for k in ("heterogeneous_boundary", "mixed_below_and_above"))},
          "Section 3.2 and Appendix A")


def profile_factor_numbers():
    """Section 4.3 and Appendix Table A-profile: phmmer alone, the extra votes alone and both, each a guarded protocol at
    delta / 4 with its own score and path, applied once to the test set; an audit run after the test results were read."""
    factors = (("phmmer_only", "phmmer added, votes unchanged", "Phmmer"), ("vote_only", "votes added, no phmmer", "Vote"),
               ("both", "both: the profile adversary", "Both"))
    found = {}
    for (key, tier), name in PROFILE_FACTORS.items():
        run = load(name)
        if run["dataset"] != ("ec" if key == "EC" else "orig-go") or run.get("tier", HOLT_TIER) != tier \
                or key == "GO" and tier == HOLT_TIER and run.get("feature_source") != "fresh":
            raise AssertionError(f"{name} is not {key} with {tier} held out on task-specific design features")
        for factor, *_ in factors:
            for task, cell in run["variants"][factor].items():
                found[factor, task, tier] = cell
    tiers = [(tier, label, [t for _, _, t in TASK_KEYS if ("phmmer_only", t, tier) in found])
             for tier, label in ((HOLT_TIER, "Families held out"), (ENTRY_TIER, "PDB entries held out"))]
    if [len(tasks) for *_, tasks in tiers] != [4, 4]:
        raise AssertionError("Appendix Table A-profile holds every label set in both tiers")
    columns = [(tier, task) for tier, _, tasks in tiers for task in tasks]
    names = {"ec": "EC", "bp": "GO-BP", "mf": "GO-MF", "cc": "GO-CC"}
    body = ["\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}l " + " ".join("c" * len(tasks) for _, _, tasks in tiers) + "@{}}", "\\toprule",
            "\\hdstrut & " + " & ".join(f"\\multicolumn{{{len(tasks)}}}{{c}}{{{label}}}" for _, label, tasks in tiers) + " \\\\"]
    rules, at = [], 2
    for _, _, tasks in tiers:
        rules.append(f"\\cmidrule(lr){{{at}-{at + len(tasks) - 1}}}")
        at += len(tasks)
    body += ["".join(rules), "\\hdstrut Adversary & " + " & ".join(names[t] for _, t in columns) + " \\\\", "\\midrule"]
    for factor, label, stem in factors:
        cells_ = []
        for tier, task in columns:
            c = found[factor, task, tier]
            cells_.append(f"{c['answered']}/{count(c['n'])}" if c["n"] else "--")
            if c["n"] and tier == HOLT_TIER:
                put(f"Factor{stem}{task.upper()}N", count(c["n"]))
                put(f"Factor{stem}{task.upper()}K", word(c["answered"]) if c["answered"] < 10 else count(c["answered"]))
            elif c["n"]:
                put(f"Factor{stem}Entry{task.upper()}N", count(c["n"]))
                put(f"Factor{stem}Entry{task.upper()}K", word(c["answered"]) if c["answered"] < 10 else count(c["answered"]))
        body.append(f"{label} & " + " & ".join(cells_) + " \\\\")
    body += ["\\bottomrule", "\\end{tabular*}"]
    (OUT / "tab_profile.tex").write_text("\n".join(body) + "\n")

    def certifies(factor, tier=HOLT_TIER):
        return {t for _, _, t in TASK_KEYS if found.get((factor, t, tier), {}).get("n")}
    above = [found["phmmer_only", t, tier] for t, tier in (("bp", HOLT_TIER), ("cc", HOLT_TIER), ("ec", ENTRY_TIER), ("bp", ENTRY_TIER))]
    check({"phmmer alone certifies EC, BP and CC with families held out and MF abstains": certifies("phmmer_only") == {"ec", "bp", "cc"},
           "the extra votes alone certify BP and CC, and EC and MF abstain": certifies("vote_only") == {"bp", "cc"},
           "with entries held out phmmer alone certifies EC and BP, the extra votes alone BP":
               certifies("phmmer_only", ENTRY_TIER) == {"ec", "bp"} and certifies("vote_only", ENTRY_TIER) == {"bp"},
           "both together abstain on every label set in every tier present": not any(c["n"] for (f, *_), c in found.items() if f == "both"),
           "phmmer alone: BP and CC with families held out, EC and BP with entries held out answered above alpha and within "
           "the 90% quantile of Bin(N, alpha)": all(0.1 * c["n"] < c["answered"] <= binom.ppf(0.9, c["n"], 0.1) for c in above)},
          "Section 4.3 and Appendix Table A-profile")


def repo_numbers():
    """Code and data availability: the repository's certified sets, chain by chain the frozen application's on DeepFRI's
    test set and after 2020, and the paired differences `holt reproduce` checks on DeepFRI's sets, value by value
    Series15's paired files."""
    index = tsv(REPO / "index.tsv")
    _, cells = applied()
    cohorts = {"deepfri": lambda task: "ec" if task == "ec" else "go", "temporal": lambda task: "temporal-go"}
    for row in index:
        if row["cohort"] not in cohorts:
            continue
        key = (cohorts[row["cohort"]](row["task"]), row["task"], row["tier"], row["alpha"])
        ids = {r["id"] for r in tsv(REPO / row["chains_tsv"])}
        if ids != {r["id"] for r in cells[key]["rows"]} or int(row["chains"]) != len(ids):
            raise AssertionError(f"{REPO} {row['chains_tsv']}: other chains than the frozen application certifies")
    paired = {}
    for name in ("ec-paired.json", "go-paired.json", "temporal-go-paired-final.json"):
        for key, cell in load(f"{S15}/{name}")["cells"].items():
            if key.startswith(f"{ROUTE}/"):
                paired.update({(*key.split("/")[1:], pair): v for pair, v in cell["paired"].items()})
    diffs = tsv(REPO / "paired-differences.tsv")
    for row in diffs:
        cohort = cohorts[row["cohort"]](row["task"])
        mine = paired[cohort, row["task"], row["tier"], row["alpha"], row["comparison"]]
        if max(abs(float(row[a]) - b) for a, b in zip(("delta_fmax", "ci95_low", "ci95_high"), (mine["delta_fmax"], *mine["ci95"]))) > 1e-9:
            raise AssertionError(f"{REPO} paired-differences.tsv {row['cohort']} {row['task']} {row['comparison']}: not Series15's")
    certified = sum(int(row["chains"]) > 0 for row in index)
    put("RepoSetsN", str(certified))
    put("RepoAbstainN", word(len(index) - certified))
    # `holt reproduce` re-scores every row of both tables, DeepFRI and post-2020, intervals included.
    put("RepoFmaxN", str(len(tsv(REPO / "score-summary.tsv"))))
    put("RepoDeltaN", str(len(diffs)))


def timing_numbers():
    """Table 1 and Appendix E: query time on one host for the two homology searches and the ESM-2 route, 20 EC test chains
    of the release, five cold and five warm runs each, medians per chain; the one-minute load at every run's start and end."""
    t = load(TIMING)
    if (t["dataset"], t["batch_size"], t["repeats"], t["amortize_queries"]) != ("DeepFRI original EC", 20, 5, 10000) \
            or t["search_settings"]["mmseqs"]["version"][:7] != "564f40d" or t["search_settings"]["foldseek"]["version"][:7] != "463739e":
        raise AssertionError(f"{TIMING} times other chains or other commits than Appendix E names")
    put("TimeChainsN", str(t["batch_size"]))
    put("TimeRepeats", word(t["repeats"]))
    put("TimeThreads", word(t["threads"]))
    put("TimeAmortN", count(t["amortize_queries"]))
    put("TimeCores", str(TIMING_CORES))
    warm, loads = {}, {}
    for stem, key in (("Mmseqs", "mmseqs"), ("Foldseek", "foldseek"), ("Esm", "esm2-650m")):
        m = t["methods"][key]
        warm[stem] = m["warm_ms_per_chain"]["median_ms"]
        put(f"TimeWarm{stem}", f"{warm[stem]:.1f}")
        put(f"TimeAmort{stem}", f"{m['cold_amortized_ms_per_chain']['median_ms']:.1f}")
        put(f"TimeCold{stem}", f"{m['cold_first_batch_ms_per_chain']['median_ms']:.0f}" if stem != "Esm"
            else f"{m['cold_first_batch_ms_per_chain']['median_ms'] / 1000:.1f}")
        loads[stem] = [r[side][0] for phase in ("cold", "warm") for r in m[phase] for side in ("load_before_1_5_15", "load_after_1_5_15")]
    search = loads["Mmseqs"] + loads["Foldseek"]
    put("TimeLoadLo", f"{min(search + loads['Esm']):.1f}")
    put("TimeLoadSearchHi", f"{max(search):.1f}")
    put("TimeLoadEsmHi", f"{max(loads['Esm']):.1f}")
    put("TimeEncodeS", count(round(t["methods"]["esm2-650m"]["train_encoding_s"])))
    check({"warm, MMseqs2 is the fastest route and the ESM-2 route the slowest": warm["Mmseqs"] < warm["Foldseek"] < warm["Esm"]},
          "Table 1 and Appendix E")


def waterfall_numbers():
    """Fig. 2 and Section 4.1: the stated EC bin stage by stage, MMseqs2, Foldseek, ESM-2 or ESM C, sibling chains, on
    the release; what a certified set of the bin's size may hold at alpha = 0.1."""
    ec = load(WATERFALL)["tasks"]["ec"]
    if ec["data"] != "data/prepared/deepfri/ec":
        raise AssertionError(f"{WATERFALL} is not drawn from the release")
    stated = ec["official_lt30"]
    answered = stated["n"] - stated["lookup_fails"]
    if stated["n"] != int(macros["ECBinN"].replace("{,}", "")):
        raise AssertionError(f"{WATERFALL} and the audit hold different stated EC bins")
    put("WaterBinN", count(stated["n"]))
    put("WaterBinAnsweredN", count(answered))
    put("WaterBinAlphaN", f"{0.1 * stated['n']:.0f}")
    steps = {step["stage"]: step["answered"] for step in stated["steps"]}
    if list(steps) != ["mmseqs", "foldseek", "esm", "sibling"] or sum(steps.values()) != answered:
        raise AssertionError(f"{WATERFALL} no longer runs MMseqs2, Foldseek, the embeddings and sibling chains in that order")
    # The homology adversary alone (MMseqs2, Foldseek or a sibling chain copies the EC number), as Fig. 2 counts it.
    bin_rows = [r for r in tsv("fig-waterfall-ec-chains.tsv") if r["official_lt30"] == "True"]
    homology = sum(any(r[k] == "True" for k in ("mmseqs", "foldseek", "sibling")) for r in bin_rows)
    if len(bin_rows) != stated["n"] or homology != int(macros["BinHomECK"].replace("{,}", "")):
        raise AssertionError("Fig. 2 and Section 4.1 count different chains of the bin answered by homology")
    put("WaterBinHomN", count(homology))
    put("WaterBinHomRatio", f"{homology / (0.1 * stated['n']):.1f}")


def temporal_numbers():
    """Chains released after 2020 (Sections 3.4 and 4.5): the cohort searched against the release's GO training chains,
    the chains the frozen rules certify in each tier (BP and CC at alpha = 0.1, MF at 0.15), how many homology answers,
    per chain and per distinct sequence, the share of the cohort's chains annotated in each label set that it certifies,
    the same sets split by initial release year; phmmer added to the unguarded adversary."""
    cohort = load(TEMPORAL_COHORT)
    _, cells = applied()
    if cohort["train"] != str(PREPARED["GO"].relative_to(ROOT)):
        raise AssertionError(f"{TEMPORAL_COHORT} searches another library than Section 3.4 describes")
    cutoff = load(TEMPORAL_STAGE)["cutoff"]
    year, month, _ = cutoff.split("-")
    put("TemporalCutoff", f"{('January February March April May June July August September October November December').split()[int(month) - 1]} {year}")
    put("TemporalCohortN", count(cohort["cohort_chains"]))
    put("TemporalScoredN", count(cohort["scored_chains"]))
    put("TemporalDupN", count(cohort["duplicates_of_training_sequences"]))
    if any(cohort["adversaries"]["profile"]["tiers"][HOLT_TIER][t]["pooled"]["0.1"]["all"]["n"] for t in ("bp", "mf", "cc")):
        raise AssertionError("Section 4.5 says phmmer leaves every post-2020 GO set of the unguarded rules empty")
    rows = [row for row in tsv(APPLIED_CHAINS) if row["method"] == ROUTE and row["cohort"] == "temporal-go"]

    def wilson_high(k, n, z=1.959964):
        centre, half = (k + z * z / 2) / (n + z * z), z * math.sqrt(k * (n - k) / n + z * z / 4) / (n + z * z)
        return centre + half

    for tier, tag in ((HOLT_TIER, "TemporalCert"), (ENTRY_TIER, "TemporalEntry")):
        for short, task in (("BP", "bp"), ("MF", "mf"), ("CC", "cc")):
            for at in ("0.1", "0.15") if task == "mf" else ("0.1",):
                c = cells["temporal-go", task, tier, at]
                mine = [row for row in rows if (row["task"], row["tier"], row["alpha"]) == (task, tier, at)]
                if (len(mine), sum(row["leak"] == "True" for row in mine)) != (c["n"], c["k"]):
                    raise AssertionError(f"{short} {tier} {at}: {APPLIED_CHAINS} and {APPLIED} count different certified chains")
                if tier == HOLT_TIER and at == "0.1":
                    put(f"TemporalElig{short}N", count(c["eligible"]))
                if not mine:
                    if not (short == "MF" and at == "0.1" or tier == ENTRY_TIER and short == "CC"):
                        raise AssertionError(f"Section 4.5 says post-2020 {short} {tier} certifies at {at}")
                    continue
                stem = (f"Temporal{'MFFifteen' if tier == HOLT_TIER else 'MFFifteenEntry'}" if at == "0.15" else f"{tag}{short}")
                put(f"{stem}N", count(c["n"]))
                put(f"{stem}K", word(c["k"]) if c["k"] < 10 else count(c["k"]))
                put(f"{stem}Pct", pct_of(c["k"], c["n"], 2))
                if tier == HOLT_TIER:
                    put(f"{stem}Share", pct_of(c["n"], c["eligible"]))
                    if at == "0.1":
                        shares[short + "Temporal"] = c["n"] / c["eligible"]
                    # One distinct sequence per unit, answered when homology answers any of its chains.
                    units = {row["rep"] for row in mine}
                    answered = {row["rep"] for row in mine if row["leak"] == "True"}
                    unit = "TemporalUnit" + ("MFFifteen" if at == "0.15" else short)
                    put(f"{unit}N", count(len(units)))
                    put(f"{unit}K", word(len(answered)) if len(answered) < 10 else count(len(answered)))
                    put(f"{unit}Hi", pct(wilson_high(len(answered), len(units))))
    family = [row for row in rows if row["tier"] == HOLT_TIER and row["alpha"] == "0.1"]
    entry_bp = {row["id"] for row in rows if row["tier"] == ENTRY_TIER and row["task"] == "bp"}
    inside = len(entry_bp & {row["id"] for row in family if row["task"] == "bp"})
    if inside != len(entry_bp):
        raise AssertionError("Section 4.5 says every post-2020 BP chain of the entry tier lies in the family-tier BP set")
    put("TemporalCertAllN", count(len({row["id"] for row in family})))
    put("TemporalSeqN", count(len({row["rep"] for row in family})))
    put("TemporalEntriesN", count(len({row["pdb"] for row in family})))
    # The certified shares on DeepFRI's test set (EC, BP, CC) and after 2020 (BP, CC): the headline of the abstract.
    put("CertShareLo", pct(min(shares.values())))
    put("CertShareHi", pct(max(shares.values())))
    put("UncertShareLo", pct(1 - max(shares.values())))
    put("UncertShareHi", pct(1 - min(shares.values())))
    if not 0.01 < min(shares.values()) <= max(shares.values()) < 0.04 \
            or set(shares) != {"ECHolt", "BPHolt", "CCHolt", "BPTemporal", "CCTemporal"}:
        raise AssertionError("the abstract says HoLT certifies between 1% and 4% of the EC, BP and CC chains, on both cohorts")

    # The post-2020 sets split at years fixed before the dates were fetched, each period tested as one deployment.
    years = load(YEARS)
    mine = [c for c in years["cells"] if c["cell"].startswith(f"{ROUTE}/") and c["chain"]["n"]]
    bins = sorted({c["years"] for c in mine})
    if len(bins) != 4 or len(mine) != 20:
        raise AssertionError(f"Section 4.5 says four periods and five certified cells each; {YEARS} disagrees")
    put("YearCellsN", count(len(mine)))
    put("YearExceedN", word(sum(bool(c["chain"]["exceed"]) for c in mine)))
    put("YearExceedOneN", word(sum(bool(c["family"]["exceed"]) for c in mine)))
    if sum(bool(c["chain"]["exceed"]) or bool(c["family"]["exceed"]) for c in mine):
        raise AssertionError("Section 4.5 says no period passes the binomial tail")
    put("YearBins", ", ".join(b.replace("-", "--") if b.split("-")[0] != b.split("-")[1] else b.split("-")[0] for b in bins[:-1])
        + " and " + (bins[-1].replace("-", "--")))
    cc = {c["years"]: c["chain"] for c in mine if c["cell"] == f"{ROUTE}/temporal-go/cc/{HOLT_TIER}/0.1"}
    put("YearCC", ", ".join(f"{cc[b]['k']}/{count(cc[b]['n'])}" for b in bins[:-1]) + f" and {cc[bins[-1]]['k']}/{count(cc[bins[-1]]['n'])}")
    span = years["cross_period_families"]
    put("YearSpanLo", count(min(v for k, v in span.items() if k.startswith(ROUTE) and v)))
    put("YearSpanHi", count(max(v for k, v in span.items() if k.startswith(ROUTE))))
    if load(YEARS_ONCE)["summary"] != {"cells": len(mine), "exceed": 0}:
        raise AssertionError(f"Section 4.5 says each family kept once, at its earliest release, leaves no period above the tail")


def certified_numbers():
    """Table 3 and Sections 1 and 4.6: ESM-S and DeepFRI's GCN and CNN on every certified set of the frozen rules, against
    Naive, the strongest homology route and Folddisco, with paired intervals over the set's families; DeepFRI's sets from
    the release scoring (paper_intervals.py, equal to Series15's paired files where both state a pair), the post-2020 sets
    from Series15's paired file."""
    r = load(INTERVALS)
    _, cells = applied()
    temporal = load(PAIRED_TEMPORAL)
    if temporal["bootstrap"]["replicates"] != r["replicates"]:
        raise AssertionError(f"{PAIRED_TEMPORAL} and {INTERVALS} resample a different number of times")
    fields = {(HOLT_TIER, "0.1"): "holt_hard", **{tuple(f.split("@")): f for _, f in CELL_SUBSETS}}

    def diff(delta, low, high):  # the point, and under it the interval as [low, high]
        return (f"\\begin{{tabular}}[t]{{@{{}}r@{{}}}}{signed(100 * delta)}\\\\[-1.5pt]"
                f"{{\\scriptsize {interval(100 * low, 100 * high)}}}\\end{{tabular}}")

    def release(task, tier, alpha):
        field = fields[tier, alpha]
        c = cells["ec" if task == "ec" else "go", task, tier, alpha]
        if c["n"] != r["counts"][task][field]["chains"]:
            raise AssertionError(f"{task} {tier}@{alpha}: the application and {INTERVALS} hold different sets")
        point, paired = r["point"][field], r["paired"][field]

        def pair(a, b):
            entry = paired.get(f"{a}-{b}", {}).get(task)
            return None if entry is None else (entry["fmax"]["delta"], *entry["fmax"]["ci95"])
        route = max((s for s, _ in HOMOLOGY_ROUTES), key=lambda s: point[s][task]["fmax"])
        return {"n": c["n"], "k": c["k"], "esm": point["EsmS"][task]["fmax"], "route": route,
                "EsmS-Naive": pair("EsmS", "Naive"), "EsmS-route": pair("EsmS", route), "Gcn-Naive": pair("Gcn", "Naive"),
                "Cnn-Naive": pair("Cnn", "Naive"), "Four-EsmS": pair("Four", "EsmS"),
                "points": {s: point[s][task]["fmax"] for s in point if task in point[s]},
                "pairs": {p: pair(*p.split("-")) for p in paired if task in paired[p]}}

    def post2020(task, tier, alpha):
        c = cells["temporal-go", task, tier, alpha]
        scored = temporal["cells"][f"{ROUTE}/temporal-go/{task}/{tier}/{alpha}"]
        if scored["ids"] != sorted(row["id"] for row in c["rows"]):
            raise AssertionError(f"{PAIRED_TEMPORAL} scores other post-2020 {task} chains than the certified list")
        pending = [name for name in S15_NAMES.values() if scored["coverage"][name]["available"] != scored["coverage"][name]["expected"]]
        if pending:
            raise AssertionError(f"post-2020 {task} {tier} {alpha}: {', '.join(pending)} not scored on every certified chain")
        points = {stem: scored["metrics"][name]["fmax"] for stem, name in S15_NAMES.items()}

        def pair(a, b):
            entry = scored["paired"].get(f"{S15_NAMES[a]} - {S15_NAMES[b]}")
            if entry is not None:
                return (entry["delta_fmax"], *entry["ci95"])
            entry = scored["paired"][f"{S15_NAMES[b]} - {S15_NAMES[a]}"]
            return (-entry["delta_fmax"], -entry["ci95"][1], -entry["ci95"][0])
        route = max((s for s, _ in HOMOLOGY_ROUTES), key=lambda s: points[s])
        return {"n": c["n"], "k": c["k"], "esm": points["EsmS"], "route": route, "points": points,
                "EsmS-Naive": pair("EsmS", "Naive"), "EsmS-route": pair("EsmS", route),
                "Gcn-Naive": pair("Gcn", "Naive"), "Cnn-Naive": pair("Cnn", "Naive"), "Four-EsmS": pair("Four", "EsmS"),
                "pairs": {f"EsmS-{s}": pair("EsmS", s) for s, _ in HOMOLOGY_ROUTES + (("Folddisco", ""), ("Four", ""))}}

    rows = {(where, task, tier, alpha): (release if where == "deepfri" else post2020)(task, tier, alpha)
            for where, task, tier, alpha in CERT_ROWS}
    # After 2020, one chain per unique sequence: the units of Section 4.5 (answered when homology answers any of its
    # chains) and ESM-S against each method over the families those chains leave.
    dedup = load(DEDUP)
    chains = [row for row in tsv(APPLIED_CHAINS) if row["method"] == ROUTE and row["cohort"] == "temporal-go"]
    names = {v: k for k, v in S15_NAMES.items()}
    for (where, task, tier, alpha), row in rows.items():
        if where != "temporal":
            continue
        cell = dedup["cells"][f"{task}_{tier}_a{alpha}"]
        mine = [c for c in chains if (c["task"], c["tier"], c["alpha"]) == (task, tier, alpha)]
        units = {c["rep"] for c in mine}
        if (cell["n_chains"], cell["n_unique_sequences"]) != (row["n"], len(units)) \
                or {c["id"] for c in mine if c["id"] in set(cell["selected_ids"])} != set(cell["selected_ids"]):
            raise AssertionError(f"{DEDUP} {task} {tier} {alpha}: other chains or sequences than the certified list")
        row["units"] = (len({c["rep"] for c in mine if c["leak"] == "True"}), len(units))
        row["seq"] = {names[m]: (p["delta_fmax"], *p["ci95"]) for m, p in cell["paired_esm_s_minus"].items() if p is not None}
    # DeepFRI's certified sets: every chain has its own sequence in the release, so they need no such column.
    _, mine = applied()
    test = {key: fasta(PREPARED[key] / "test.fasta") for key in PREPARED}
    for where, task, tier, alpha in CERT_ROWS:
        ids = [r["id"] for r in mine["ec" if task == "ec" else "go", task, tier, alpha]["rows"]] if where == "deepfri" else []
        if len({test["EC" if task == "ec" else "GO"][i] for i in ids}) != len(ids):
            raise AssertionError(f"Table 3 says every chain of DeepFRI's {task} {tier} set has its own sequence")
    # Fig. 5e: an EC chain certified in both tiers that neither search reaches, read as figures/fig_certified.py reads it.
    spec = "4M3L-A"
    if not all(spec in {r["id"] for r in mine["ec", "ec", tier, "0.1"]["rows"]} for tier in (HOLT_TIER, ENTRY_TIER)):
        raise AssertionError(f"Fig. 5e says {spec} is certified in both EC tiers")
    evaluated = next(json.loads(line) for line in (RAW / EVALUATED["EC"]).open() if f'"{spec}"' in line)
    if evaluated["identity"] or evaluated["bits"]:
        raise AssertionError(f"Fig. 5e says {spec} has no MMseqs2 or Foldseek hit in training")
    truth = next(json.loads(line) for line in (PREPARED["EC"] / "test.jsonl").open() if f'"{spec}"' in line)["labels"]["ec"]
    number = max(truth, key=lambda t: ("-" not in t, t))

    def score(name):
        return next(json.loads(line) for line in (RAW / name).open() if f'"{spec}"' in line)["scores"].get(number, 0.0)
    put("SpecChain", spec)
    put("SpecEC", number)
    put("SpecEsmS", f"{score('published-esm-s-original-ec-ec-predictions.jsonl'):.2f}")
    put("SpecFolddisco", f"{score('folddisco-default-deepfri-ann-ec-ec-predictions.jsonl'):.2f}")
    put("SpecEsmTwo", f"{evaluated['esm2']:.3f}")
    put("SpecEsmC", f"{evaluated['esmc']:.3f}")
    groups = {"deepfri": "DeepFRI test set", "temporal": "PDB chains released after " + macros["TemporalCutoff"]}
    letter = dict(HOMOLOGY_ROUTES)

    def line(task, held, row):
        answered = f"{count(row['k'])}/{count(row['n'])} ({pct_of(row['k'], row['n'])})"
        if "units" in row:  # its sequences under it, as the paired cells set their intervals
            answered = (f"\\begin{{tabular}}[t]{{@{{}}r@{{}}}}{answered}\\\\[-1.5pt]"
                        f"{{\\scriptsize {row['units'][0]}/{count(row['units'][1])} seq.}}\\end{{tabular}}")
        cells_ = [task.upper(), held, answered, f"{100 * row['esm']:.1f}", diff(*row["EsmS-Naive"]),
                  diff(*row["seq"]["Naive"]) if "seq" in row else "", diff(*row["EsmS-route"]), letter[row["route"]],
                  diff(*row["pairs"]["EsmS-Folddisco"])]
        cells_ += [diff(*row[key]) for key in ("Gcn-Naive", "Cnn-Naive", "Four-EsmS")]
        return " & ".join(cells_) + " \\\\"
    body = []
    for where in groups:
        body.append(f"\\rowgroupn{{12}}{{{groups[where]}}}")
        for (w, task, tier, alpha), row in rows.items():
            if w == where:
                body.append(line(task, ("family" if tier == HOLT_TIER else "entry") + f", {alpha}", row))
        if where == "deepfri":
            body.append("\\midrule")
    (OUT / "tab_certified.tex").write_text("\n".join(body) + "\n")
    seq = {key: row["seq"] for key, row in rows.items() if "seq" in row}
    for task in ("bp", "cc"):
        s = seq["temporal", task, HOLT_TIER, "0.1"]
        put(f"SeqEsmSNaive{task.upper()}", signed(100 * s["Naive"][0]))
        put(f"SeqEsmSNaive{task.upper()}CI", interval(100 * s["Naive"][1], 100 * s["Naive"][2]))
    s = seq["temporal", "mf", HOLT_TIER, "0.15"]
    put("SeqEsmSFourMF", signed(100 * s["Four"][0]))
    put("SeqEsmSFourMFCI", interval(100 * s["Four"][1], 100 * s["Four"][2]))
    # Post-2020 BP near the floor: its chains carry about one BP term each, against tens on DeepFRI's BP set.
    after, before = dedup["temporal_bp_full"], dedup["original_bp"]
    if (after["n_chains"], before["n_chains"]) != (rows["temporal", "bp", HOLT_TIER, "0.1"]["n"], rows["deepfri", "bp", HOLT_TIER, "0.1"]["n"]) \
            or f"{100 * after['methods']['ESM-S']['fmax']:.1f}" != f"{100 * rows['temporal', 'bp', HOLT_TIER, '0.1']['esm']:.1f}" \
            or f"{100 * before['methods']['ESM-S']['fmax']:.1f}" != f"{100 * rows['deepfri', 'bp', HOLT_TIER, '0.1']['esm']:.1f}":
        raise AssertionError(f"{DEDUP} reads other BP sets or other ESM-S scores than Table 3")
    put("TemporalBPTerms", word(int(after["truth_terms"]["median"])))
    put("TemporalBPTermsHi", word(int(after["truth_terms"]["p90"])))
    put("DeepfriBPTerms", f"{before['truth_terms']['median']:.0f}")
    hit = after["methods"]["ESM-S"]
    put("TemporalEsmSBPHitN", count(hit["any_correct_n"]))
    put("TemporalEsmSBPHitPct", pct_of(hit["any_correct_n"], after["n_chains"]))
    check({"one chain per sequence: ESM-S leads Naive on every post-2020 set, every interval above zero":
               all(s["Naive"][1] > 0 for s in seq.values()),
           "one chain per sequence: the four-route vote and ESM-S level on BP and CC, ESM-S ahead on MF":
               all(seq["temporal", t, HOLT_TIER, "0.1"]["Four"][1] < 0 < seq["temporal", t, HOLT_TIER, "0.1"]["Four"][2] for t in ("bp", "cc"))
               and s["Four"][1] > 0,
           "one chain per sequence: on BP ESM-S leads BLAST, and its intervals against MMseqs2 and Foldseek span zero":
               seq["temporal", "bp", HOLT_TIER, "0.1"]["Blast"][1] > 0
               and all(seq["temporal", "bp", HOLT_TIER, "0.1"][m][1] < 0 for m in ("Mmseqs", "Foldseek"))},
          "Section 4.6 and Table 3")

    # Prose, by cell stem.
    stems = {("deepfri", "bp", ENTRY_TIER, "0.1"): "EntryBP", ("deepfri", "mf", HOLT_TIER, "0.15"): "MFFifteen",
             ("temporal", "bp", HOLT_TIER, "0.1"): "TemporalBP", ("temporal", "cc", HOLT_TIER, "0.1"): "TemporalCC",
             ("temporal", "bp", ENTRY_TIER, "0.1"): "TemporalEntryBP", ("temporal", "mf", HOLT_TIER, "0.15"): "TemporalMFFifteen"}
    for key, stem in stems.items():
        row = rows[key]
        for name in ("EsmS-Naive", "EsmS-route", "Gcn-Naive", "Cnn-Naive", "Four-EsmS"):
            if stem == "EntryBP" and name != "EsmS-route":
                continue  # release_numbers names the entry tier's other pairs by subset
            delta, low, high = row[name]
            tag = name.replace("-", "").replace("route", "Route")
            put(f"{tag}{stem}", signed(100 * delta))
            put(f"{tag}{stem}Abs", f"{abs(100 * delta):.1f}")
            put(f"{tag}{stem}CI", interval(100 * low, 100 * high))
        if key[0] == "temporal":
            delta, low, high = row["pairs"]["EsmS-Four"]
            put(f"EsmSFour{stem}", signed(100 * delta))
            put(f"EsmSFour{stem}Abs", f"{abs(100 * delta):.1f}")
            put(f"EsmSFour{stem}CI", interval(100 * low, 100 * high))
    post = {"bp": rows["temporal", "bp", HOLT_TIER, "0.1"], "cc": rows["temporal", "cc", HOLT_TIER, "0.1"]}
    for task, row in post.items():
        short = task.upper()
        for stem, value in row["points"].items():
            put(f"Temporal{stem}{short}", f"{100 * value:.1f}")
        for s, _ in HOMOLOGY_ROUTES + (("Folddisco", ""),):
            delta, low, high = row["pairs"][f"EsmS-{s}"]
            put(f"TemporalEsmS{s}{short}", signed(100 * delta))
            put(f"TemporalEsmS{s}{short}Abs", f"{abs(100 * delta):.1f}")
            put(f"TemporalEsmS{s}{short}CI", interval(100 * low, 100 * high))
            put(f"Temporal{s}EsmS{short}", signed(-100 * delta))  # Fig. 3 reads route minus ESM-S
        put(f"TemporalTop{short}", f"{100 * max(row['points'].values()):.1f}")
        all_scores = [float(r["fmax"]) for r in tsv("../../holt/certified_sets/score-summary.tsv")
                      if r["cohort"] == "temporal" and r["task"] == task]
        put(f"TemporalAllTop{short}", f"{100 * max(all_scores):.1f}")
        # Folddisco minus ESM-S after 2020, as Fig. 3's tray draws it from fig-scores.json.
        drawn = load("fig-scores.json")["cells"]["time_split/" + task]["paired"]["esm_s_minus"]["folddisco"]
        if max(abs(a - b) for a, b in zip(row["pairs"]["EsmS-Folddisco"], (drawn["delta_fmax"], *drawn["ci95"]))) > 1e-9:
            raise AssertionError(f"post-2020 {short}: Fig. 3 and {PAIRED_TEMPORAL} give different ESM-S - Folddisco")
    mf = rows["temporal", "mf", HOLT_TIER, "0.15"]
    delta, low, high = mf["pairs"]["EsmS-Folddisco"]
    put("TemporalEsmSFolddiscoMF", signed(100 * delta))
    put("TemporalEsmSFolddiscoMFCI", interval(100 * low, 100 * high))
    deepfri = {key: row for key, row in rows.items() if key[0] == "deepfri"}
    at10 = {key: row for key, row in deepfri.items() if key[3] == "0.1" and key[1] != "ec"}  # DeepFRI's BP and CC sets
    ec = {key: row for key, row in deepfri.items() if key[1] == "ec"}
    go = {key: row for key, row in rows.items() if key[3] == "0.1" and key[1] != "ec"}  # every BP and CC set
    leads = [100 * row["EsmS-Naive"][0] for row in at10.values()]
    put("EsmSNaiveCertLo", f"{min(leads):.1f}")
    put("EsmSNaiveCertHi", f"{max(leads):.1f}")
    gcn = [100 * row["Gcn-Naive"][0] for key, row in deepfri.items() if key[3] == "0.1"]
    put("GcnNaiveCertLo", f"{min(gcn):.1f}")
    put("GcnNaiveCertHi", f"{max(gcn):.1f}")
    folddisco = [row["pairs"]["EsmS-Folddisco"] for row in deepfri.values()]
    put("EsmSFolddiscoCertLo", f"{100 * min(d for d, _, _ in folddisco):.1f}")
    put("EsmSFolddiscoCertHi", f"{100 * max(d for d, _, _ in folddisco):.1f}")
    # Fig. 3 draws, per label set, the family tier: BP and CC at alpha = 0.1, MF at 0.15.
    lane = [row["pairs"]["EsmS-Folddisco"][0] for key, row in deepfri.items() if key[2] == HOLT_TIER]
    put("EsmSFolddiscoFigLo", f"{100 * min(lane):.1f}")
    put("EsmSFolddiscoFigHi", f"{100 * max(lane):.1f}")
    route = [100 * row["EsmS-route"][0] for key, row in deepfri.items() if key[1] != "ec"]  # BP, CC and MF
    put("EsmSRouteGOCertLo", f"{min(route):.1f}")
    put("EsmSRouteGOCertHi", f"{max(route):.1f}")
    gap = [100 * row["pairs"]["EsmS-Gcn"][0] for row in deepfri.values()]
    put("EsmSGcnCertLo", signed(min(gap)))
    put("EsmSGcnCertHi", signed(max(gap)))
    four = [100 * row["Four-EsmS"][0] for row in go.values()]
    put("FourEsmSCertLo", signed(min(four)))
    put("FourEsmSCertHi", signed(max(four)))
    cc = [post["cc"]["pairs"][f"EsmS-{s}"] for s, _ in HOMOLOGY_ROUTES]
    put("TemporalEsmSRouteCCLo", f"{100 * min(d for d, _, _ in cc):.1f}")
    put("TemporalEsmSRouteCCFloor", f"{math.floor(100 * min(low for _, low, _ in cc)):.0f}")
    bp = post["bp"]

    def above(entry):
        return entry is not None and entry[1] > 0

    def spans(entry):
        return entry[1] < 0 < entry[2]
    check({
        "DeepFRI's GCN and CNN lead Naive on every certified set, every interval above zero":
            all(above(row["Gcn-Naive"]) and above(row["Cnn-Naive"]) for key, row in rows.items() if key != ("temporal", "cc", HOLT_TIER, "0.1"))
            and above(post["cc"]["Gcn-Naive"]),
        "ESM-S leads Naive on every certified BP, CC and MF set, every interval above zero; on both EC sets the interval "
        "spans zero": all(above(row["EsmS-Naive"]) for key, row in rows.items() if key[1] != "ec")
            and all(spans(row["EsmS-Naive"]) for row in ec.values()),
        "on both EC sets the four-route vote leads ESM-S, and ESM-S trails Foldseek's interval across zero":
            all(above(row["Four-EsmS"]) and row["route"] == "Foldseek" and spans(row["EsmS-route"]) for row in ec.values()),
        "ESM-S leads Folddisco on each of DeepFRI's certified sets, every interval above zero": all(low > 0 for _, low, _ in folddisco),
        "BLAST is the strongest homology route on every certified BP and CC set of DeepFRI's test set":
            all(row["route"] == "Blast" for row in at10.values()),
        "ESM-S leads the strongest homology route on every certified BP, CC and MF set of DeepFRI's test set, every "
        "interval above zero": all(above(row["EsmS-route"]) for key, row in deepfri.items() if key[1] != "ec"),
        "post-2020 CC: ESM-S leads every single homology route, every interval above zero": all(low > 0 for _, low, _ in cc),
        "post-2020 BP: ESM-S leads BLAST, MMseqs2 and Foldseek and spans zero against DiamondScore":
            all(above(bp["pairs"][f"EsmS-{s}"]) for s in ("Blast", "Mmseqs", "Foldseek")) and spans(bp["pairs"]["EsmS-Diamond"]),
        "the four-route vote and ESM-S: intervals across zero on every BP and CC set and on DeepFRI's MF set; ESM-S ahead "
        "on post-2020 MF": all(spans(row["Four-EsmS"]) for key, row in rows.items()
                               if key != ("temporal", "mf", HOLT_TIER, "0.15") and key[1] != "ec")
            and mf["Four-EsmS"][2] < 0,
        "ESM-S and DeepFRI's GCN are level on each of DeepFRI's certified sets": all(spans(row["pairs"]["EsmS-Gcn"]) for row in deepfri.values()),
        "post-2020 MF at alpha = 0.15: ESM-S leads every single homology route, every interval above zero":
            all(above(mf["pairs"][f"EsmS-{s}"]) for s, _ in HOMOLOGY_ROUTES),
    }, "Sections 1 and 4.6, Table 3")


def shake_numbers():
    """ProteinShake (Sections 3.4, 4.5 and 5): the guarded protocol refit on each split, family tier, homology adversary,
    alpha = 0.1; and the two-route lower bounds on every test protein and on those that pass fixed thresholds."""
    cells, proteins = [], 0
    for dataset in ("ec", "go"):
        for split in ("random", "sequence", "structure"):
            run = load(GUARDED_SHAKE.format(dataset=dataset, split=split))
            if run["alpha"] != 0.1 or "identity<0.3" not in run["guard"] or run["dataset"] != f"ps-{dataset}-{split}":
                raise AssertionError(f"{GUARDED_SHAKE.format(dataset=dataset, split=split)} is not the guarded protocol at alpha = 0.1")
            ids = set()
            for task in (("ec",) if dataset == "ec" else ("bp", "mf", "cc")):
                c = run["cells"][f"{task}_{HOLT_TIER}"]
                if len(c["ids"]) != c["n"]:
                    raise AssertionError(f"ProteinShake {dataset} {split} {task}: the certified list and its count differ")
                cells.append({"split": split, "task": task, "n": c["n"], "leaks": c["leaks"]})
                ids |= set(c["ids"])
            proteins += len(ids)
    certified = [c for c in cells if c["n"]]
    if len(certified) != len(cells):
        raise AssertionError("Section 4.5 says the guarded family tier certifies a set in every ProteinShake cell")
    put("ShakeCellsN", word(len(cells)) if len(cells) < 10 else str(len(cells)))
    n, k = sum(c["n"] for c in cells), sum(c["leaks"] for c in cells)
    put("ShakeCertN", count(n))
    put("ShakeCertK", count(k))
    put("ShakeCertPct", pct_of(k, n))
    put("ShakeProteinsN", count(proteins))
    # Cells answered above alpha, with the binomial probability of so many answered chains at risk exactly alpha.
    above = [{**c, "tail": binom.sf(c["leaks"] - 1, c["n"], 0.1)} for c in certified if c["leaks"] > 0.1 * c["n"]]
    worst = min(above, key=lambda c: c["tail"])
    rest = [c["tail"] for c in above if c is not worst]
    if (worst["split"], worst["task"]) != ("structure", "bp") or worst["tail"] >= 0.05 or min(rest) < 0.05:
        raise AssertionError("Section 4.5 says only BP on the structure split is answered more often than chance at risk alpha allows")
    put("ShakeAboveN", word(len(above)))
    put("ShakeWorstK", count(worst["leaks"]))
    put("ShakeWorstN", count(worst["n"]))
    put("ShakeWorstPct", pct_of(worst["leaks"], worst["n"]))
    put("ShakeWorstTail", f"{worst['tail']:.3f}")
    put("ShakeAboveTailLo", f"{min(rest):.2f}")
    put("ShakeAboveTailHi", f"{max(rest):.2f}")
    splits = load(SHAKE_BOUNDS)["splits"]
    every = [cell["all"]["lower_bound_rate"] for split in splits.values() for cell in split.values()]
    screened = [cell["fixed_30_bstar"]["lower_bound_rate"] for split in splits.values() for cell in split.values()]
    if len(screened) != len(cells):
        raise AssertionError(f"{SHAKE_BOUNDS} and {GUARDED_SHAKE} hold different cells")
    put("ShakeTransferLo", pct(min(every), 0))
    put("ShakeTransferHi", pct(max(every), 0))
    put("ShakeScreenedLo", pct(min(screened), 0))
    put("ShakeScreenedHi", pct(max(screened), 0))
    put("ShakeScreenedAboveN", str(sum(rate > 0.1 for rate in screened)))


def calibration_numbers():
    """Fig. A2: the resampling intervals of the transfer level at 30% identity and of b* over PDB entries, from the
    figure's own resamples (figures/fig_calibration.py), so that caption and panel quote one interval."""
    import sys
    sys.path.insert(0, str(HERE / "figures"))
    import fig_calibration as calibration  # its style module reads the source names of this file
    for key in ("EC", "GO"):
        entry, x_seq, y_seq, x_str, y_str = calibration.pairs(key.lower())
        at_thirty, strong, _, _ = calibration.threshold(x_seq, y_seq, x_str, y_str)
        stated = load(f"audit-{AUDIT[key]}.json")["calibration"]
        if abs(at_thirty - stated["f1_at_30pct_identity"]) > 1e-6 or strong != stated["strong_bits"]:
            raise AssertionError(f"{key}: the calibration pairs no longer give the audit's level and b*")
        boot = calibration.resample(entry, x_seq, y_seq, x_str, y_str)
        for stem, values, digits in (("Level", boot["level"], 2), ("Bstar", boot["strong"], 0)):
            low, high = calibration.ci(values)
            put(f"Cal{stem}{key}Lo", f"{low:.{digits}f}")
            put(f"Cal{stem}{key}Hi", f"{high:.{digits}f}")
    put("CalReps", count(calibration.REPS))


def settings_table():
    """Appendix Table A-settings: the vote settings, blend weights and stacked-combiner configuration validation chose,
    read from the files the scored predictions came from."""
    routes = (("MMseqs2", "mmseqs-deepfri-ann"), ("Foldseek", "foldseek-deepfri-ann"), ("ESM-2", "esm2-650m-deepfri-ann"),
              ("ESM~C", "esmc-600m-deepfri-ann"), ("Folddisco", "folddisco-default-deepfri-ann"))
    score = {2: "cos", 3: "0.05", 4: "0.02", 5: "0.01"}  # hit columns of scripts/plm_knn.py: cosine, then exp((cos-1)/T)
    tasks = [(task, "ec" if task == "ec" else "go") for _, _, task in TASK_KEYS]
    ens = {ds: load(f"ens-deepfri-ann-{ds}-metrics.json") for ds in ("ec", "go")}
    stack = {ds: load(f"stack4-deepfri-ann-{ds}-metrics.json") for ds in ("ec", "go")}
    body = []
    for label, base in routes:
        cells = []
        for task, ds in tasks:
            prefix = ens[ds][task]["picked_routes"].get(base, base)
            chosen = load(f"{prefix}-{ds}-metrics.json")[task]
            cell = f"{chosen['k']}, {chosen['weight_power']}"
            if prefix != base or base.startswith("esm"):
                variant = prefix[len(base) - len("ann"):-len("-ann")]
                cell += f"; {'mid' if 'mid' in variant else 'last'}{' z' if 'z' in variant else ''}; {score[chosen['score_column']]}"
            cells.append(cell)
        body.append(" & ".join([label] + cells) + " \\\\")
    body.append("\\midrule")
    for label, members in (("Two routes", routes[:2]), ("Four routes", routes[:4])):
        cells = []
        for task, ds in tasks:
            key = "ens-deepfri-ann-" + "+".join(base for _, base in members)
            weights = ens[ds][task]["blends"][key]["weights"]
            cells.append(", ".join(f"{weights[base]:.1f}" for _, base in members))
        body.append(" & ".join([label] + cells) + " \\\\")
    cells = []
    for task, ds in tasks:
        c = stack[ds][task]["config"]
        cells.append(f"{c['learning_rate']}, {c['max_iter']}, {c['max_leaf_nodes']}, {c.get('l2_regularization', 0):g}")
    body.append(" & ".join(["Stacked combiner"] + cells) + " \\\\")
    (OUT / "tab_settings.tex").write_text("\n".join(body) + "\n")


def plain_abstract():
    """abstract.txt for the submission form: abstract.tex with every macro replaced by its value."""
    title = re.search(r"\\newcommand\{\\papertitle\}\{(.+?)\}\n", (HERE / "main.tex").read_text()).group(1).replace("\\titlebreak ", " ")
    text = (HERE / "abstract.tex").read_text()
    text = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("%"))
    name = re.search(r"\\newcommand\{\\holt\}\{(.+?)\}", (HERE / "main.tex").read_text()).group(1)
    text = text.replace("\\holt{}", name)

    def value(match):
        if match.group(1) not in macros:
            raise KeyError(f"abstract.tex uses \\{match.group(1)}, which make_numbers.py does not define")
        return macros[match.group(1)].replace("{,}", ",")
    text = re.sub(r"\\([A-Za-z]+)\{\}", value, text)
    text = re.sub(r"\\(?:texttt|url|emph)\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\$\\alpha=([0-9.]+)\$", "\u03b1 = \\1", text)
    text = re.sub(r"\$\\alpha=\\delta=([0-9.]+)\$", "\u03b1 = \u03b4 = \\1", text)
    for old, new in (("$1-\\delta$", "1\u2212\u03b4"), ("$\\alpha$", "\u03b1"), ("$\\delta$", "\u03b4"), ("\\%", "%"), ("$<$", "<"), ("$\\geq$", "\u2265"), ("$-$", "\u2212"), ("``", '"'),
                     ("''", '"'), ("~", " "), ("--", "\u2013"), ("\\,", " "), ("\\par", "\n")):
        text = text.replace(old, new)
    if "\\" in text or "$" in text:
        raise ValueError("abstract.tex still holds LaTeX after conversion: " + re.findall(r"[\\$]\S*", text)[0])
    paragraphs = [" ".join(block.split()) for block in text.split("\n\n") if block.strip()]
    (HERE / "abstract.txt").write_text(title + "\n\n" + "\n\n".join(paragraphs) + "\n")
    return len(" ".join(paragraphs).split())


def main():
    OUT.mkdir(exist_ok=True)
    audit_numbers()
    coverage_numbers()
    example_numbers()
    audit_table()
    integrity_numbers()
    published_numbers()
    release_numbers()
    main_table()
    holt_numbers()
    transfer_numbers()
    event_numbers()
    ablation_numbers()
    synthetic_numbers()
    profile_factor_numbers()
    timing_numbers()
    repo_numbers()
    waterfall_numbers()
    temporal_numbers()
    certified_numbers()
    shake_numbers()
    calibration_numbers()
    settings_table()
    # Missing table cells use the same cross as the manuscript's legend.
    for table in OUT.glob("tab_*.tex"):
        table.write_text(table.read_text().replace("--", r"\xmark{}"))
    lines =["% Written by make_numbers.py from the result files. Do not edit by hand."]
    lines += [f"\\newcommand{{\\{name}}}{{{value}}}" for name, value in sorted(macros.items())]
    (OUT / "numbers.tex").write_text("\n".join(lines) + "\n")
    words = plain_abstract()
    print(f"{len(macros)} numbers -> {OUT / 'numbers.tex'}; abstract {words} words")


if __name__ == "__main__":
    main()
