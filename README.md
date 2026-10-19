# HoLT

Given an existing training and test split for protein function prediction, HoLT (Homology Learn-then-Test) finds the test chains that homology transfer cannot answer and pairs that subset with a risk certificate. Re-scoring published predictions on this risk-controlled hard set measures what a predictor learns beyond homology, without building a new split or retraining.

Across the released DeepFRI benchmark split and PDB chains released after May 2020, HoLT certifies 1.9% to 3.7% of annotated test chains against sequence search (MMseqs2), structure search (Foldseek) and sibling chains at confidence 90% (δ=0.1). A full-benchmark score therefore averages over at least 97% of chains where homology transfer remains active.

| Split and label set | Level α (δ=0.1) | Certified chains (family / entry tier) | Family-tier share (certified / eligible) | Paired Fmax difference on the family-tier hard set [95% CI] | Chain list |
|---|---:|---:|---:|---|---|
| DeepFRI EC | 0.1 | 37 / 41 | 2.0% (37 / 1,876) | ESM-S over Folddisco +20.1 [8.2, 31.9]; 4-route transfer over ESM-S +7.9 [0.4, 20.2] | [TSV](certified_sets/deepfri/ec-mmseqs30-a010/chains.tsv) |
| DeepFRI GO-BP | 0.1 | 109 / 63 | 3.2% (109 / 3,416) | ESM-S over Folddisco +19.9 [14.8, 25.6], over Naive +11.2 [5.9, 17.8] | [TSV](certified_sets/deepfri/bp-mmseqs30-a010/chains.tsv) |
| DeepFRI GO-CC | 0.1 | 64 / abstains | 1.9% (64 / 3,416) | ESM-S over Folddisco +19.7 [8.1, 29.1], over Naive +13.9 [4.4, 24.2] | [TSV](certified_sets/deepfri/cc-mmseqs30-a010/chains.tsv) |
| DeepFRI GO-MF | 0.15 | 128 / 117 | 3.7% (128 / 3,416) | ESM-S over Folddisco +20.7 [14.7, 27.2], over Naive +20.2 [13.6, 27.9] | [TSV](certified_sets/deepfri/mf-mmseqs30-a015/chains.tsv) |
| Post-2020 PDB GO-BP | 0.1 | 2,248 / 1,461 | 2.8% (2,248 / 79,923) | ESM-S over Naive +4.0 [1.8, 5.6] | [TSV](certified_sets/temporal/bp-mmseqs30-a010/chains.tsv) |
| Post-2020 PDB GO-CC | 0.1 | 3,109 / abstains | 2.5% (3,109 / 122,067) | ESM-S over every homology route by +12.6 or more; over Naive +10.4 [6.5, 16.6] | [TSV](certified_sets/temporal/cc-mmseqs30-a010/chains.tsv) |
| Post-2020 PDB GO-MF | 0.15 | 2,308 / 1,970 | 2.6% (2,308 / 88,690) | ESM-S over Naive +19.2 [6.0, 29.5], over 4-route transfer +7.1 [2.2, 9.9] | [TSV](certified_sets/temporal/mf-mmseqs30-a015/chains.tsv) |

All 42 certified sets (DeepFRI, post-2020 PDB and ProteinShake, both tiers) are in [`certified_sets/`](certified_sets/), listed in [`certified_sets/index.tsv`](certified_sets/index.tsv) with a `certificate.json` beside each chain list. Filter your own predictions to a TSV and re-score.

## Reproduce the paper numbers on CPU

Reproducing the certified hard sets and paired Fmax scores from the precomputed evidence requires Python 3.10+ on a laptop CPU, with no GPU, no MMseqs2 or Foldseek binaries, and no model weights.

Install the package from this directory:

```sh
pip install .
```

Run the reproduction command. From a fresh clone it takes 3.2 seconds of wall time on a laptop CPU (Python 3.10; tested on 3.10 and 3.13):

```sh
holt reproduce
```

Expected terminal output:

```text
DeepFRI EC pdb alpha=0.1: 41 chains; exact IDs; calibration 3/97
DeepFRI EC mmseqs30 alpha=0.1: 37 chains; exact IDs; calibration 6/141
DeepFRI BP pdb alpha=0.1: 63 chains; exact IDs; calibration 1/111
DeepFRI MF pdb alpha=0.1: 0 chains; exact IDs; abstain
DeepFRI MF pdb alpha=0.15: 117 chains; exact IDs; calibration 9/146
DeepFRI CC pdb alpha=0.1: 0 chains; exact IDs; abstain
DeepFRI BP mmseqs30 alpha=0.1: 109 chains; exact IDs; calibration 18/301
DeepFRI MF mmseqs30 alpha=0.1: 0 chains; exact IDs; abstain
DeepFRI MF mmseqs30 alpha=0.15: 128 chains; exact IDs; calibration 26/264
DeepFRI CC mmseqs30 alpha=0.1: 64 chains; exact IDs; calibration 5/135
Certified lists: 10/10 exact
Paired Fmax EC mmseqs30 alpha=0.1: 14 methods, 28 paired deltas exact
Paired Fmax EC pdb alpha=0.1: 14 methods, 28 paired deltas exact
Paired Fmax BP mmseqs30 alpha=0.1: 13 methods, 21 paired deltas exact
Paired Fmax BP pdb alpha=0.1: 13 methods, 21 paired deltas exact
Paired Fmax MF mmseqs30 alpha=0.15: 13 methods, 21 paired deltas exact
Paired Fmax MF pdb alpha=0.15: 13 methods, 21 paired deltas exact
Paired Fmax CC mmseqs30 alpha=0.1: 13 methods, 21 paired deltas exact
Paired scores: 93 Fmax and 161 deltas exact
Wall time: 1.447s
```

## Audit a new split

Verify the installation on the bundled minimal sample, which needs no external data:

```sh
holt audit examples/minimal/manifest.json --no-embeddings --output audit-smoke
```

Audit a split from a manifest of precomputed search hits and training embeddings:

```sh
holt audit manifest.json --output audit-out
```

For a new split that needs ESM-2 and ESM C vectors computed locally, install the embedding dependencies:

```sh
pip install '.[embeddings]'
```

MMseqs2 and Foldseek are separate command-line tools. Model weights must already be cached or supplied in local directories; HoLT does not download weights during an audit.

For a split with `train.jsonl`, `train.fasta`, `test.jsonl`, `test.fasta`, and per-chain structures under `train/pdb_ca/` and `test/pdb_ca/`, generate all four routes locally:

```sh
holt audit --split-dir SPLIT --esm2-model ESM2_WEIGHTS --esmc-model ESMC_BF16_WEIGHTS --output audit-out
```

## Output

HoLT writes two files to `--output` and prints a summary to the terminal:

- `hard-set.tsv`: one certified chain per row with `tier`, `task`, `id`, `score` and `test_copy_event` (test copying outcome).
- `certificate.json`: input hashes, role checks, fitted score, calibration trace, per-candidate level, selected cut and counts for each task and tier.
- Terminal summary: certified/eligible counts, calibration upper risk bound and selected IDs.

## Protocol

HoLT audits a protein function benchmark split and writes a certified hard set for each label set and holdout tier. It fits a learned score of transfer risk and the calibration path on training chains alone. HoLT freezes the rule before reading test labels, which it uses only at the end to report eligible counts and realized copying events.

A chain enters design, calibration or deployment only when it passes the guard, which requires maximum MMseqs2 sequence identity below 30% over hits at any coverage. Along the candidate path ordered by the learned score, the fixed-sequence test starts at the first candidate large enough to pass with zero events at α=0.05 and stops at its first failure. The test checks each candidate at significance level `δ / (K × 4)`. An empty certified set is abstention.

The binomial statement controls risk over independent calibration families. Its extension to deployed test chains requires the transfer condition described in the HoLT manuscript.

## Inputs and parameters

The manifest (`manifest.json`) names the training and test JSONL/FASTA, a training-only 30% cluster table, training self-search hits, test-to-training hits, and the two training embedding matrices. JSONL rows have an `id` and a `labels` map, and FASTA IDs must match. Search hit tables contain query and target IDs followed by route scores. See `examples/minimal/manifest.json` for the schema.

When running with `--split-dir`, `--mmseqs` and `--foldseek` accept executable paths when those tools are outside `PATH`. `--esm2-python` and `--esmc-python` accept separate Python environments with the respective model packages and HoLT installed. Each model runs in its own process with batch size one, and HoLT windows sequences over 1,000 residues. The evidence directory retains the generated search tables, vectors and logs. With a bfloat16 ESM C weight directory, each model process stays under 4 GB of memory on a 32 GB Apple M5 laptop.

Defaults are α=0.1 and δ=0.1; `--alpha 0.15` runs the secondary level the paper reports for MF. The default `--protocols-k 1` gives `δ/4`; use `--protocols-k 4` when accounting for a choice among four protocols on the same training labels. To omit both embedding routes, use `--no-embeddings`; this fits a separate two-route certificate and does not reproduce a four-route set.

## Citation

Yan Su. *HoLT: Homology Learn-then-Test* (manuscript, 2026).
