# Certified sets

[index.tsv](index.tsv) lists every released cell. Each `chains.tsv` gives the
chain IDs, PDB entry, chain, grouping key, task, holdout tier, alpha, and
gamma. The adjacent `certificate.json` records the training calibration,
selected path point, binomial upper bound, source checksum, and provenance.
The index marks cells without model scores as `NaN（未重算）`.
Open the DeepFRI family-tier lists directly: [EC](deepfri/ec-mmseqs30-a010/chains.tsv),
[BP](deepfri/bp-mmseqs30-a010/chains.tsv),
[MF at alpha 0.15](deepfri/mf-mmseqs30-a015/chains.tsv), and
[CC](deepfri/cc-mmseqs30-a010/chains.tsv).
Zero-chain cells are recorded as abstentions. All cells use delta = 0.1 and
gamma = delta / 4 = 0.025. The ProteinShake cells use the two-search-route
score; the DeepFRI and post-2020 cells use the four-route score. Each
certificate names its score routes.
`NaN` in a supplementary feature field means the frozen source did not record
that per-chain value; it does not mean zero.

[score-summary.tsv](score-summary.tsv) and
[paired-differences.tsv](paired-differences.tsv) contain matched-chain scores
from the frozen paired reports. From this directory's `replay.json.gz` and
`predictions.json.gz`, `holt reproduce` recomputes the DeepFRI training fit,
calibration, exact certified IDs, 93 Fmax values, and 161 paired Fmax
differences. Recomputed coefficients are checked against the frozen training
fit. The post-2020 score rows retain their
source-report values and are not recomputed by this command. ProteinShake
model scores are `NaN（未重算）` and have no rows in the score tables.

From the repository root, after `pip install .`, run:

```sh
holt reproduce
```
