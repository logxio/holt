"""Acceptance checks against independent, previously frozen chain lists."""

import csv
import os
from pathlib import Path
import subprocess
import sys

import pytest

from holt._audit import audit, load_manifest
from holt._certify import assert_tail_position
from holt._prepare import MMSEQS_INSTALL, binary
from holt.cli import main


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = PACKAGE_ROOT / "examples"
SETS = PACKAGE_ROOT / "certified_sets"


@pytest.mark.parametrize("alpha,delta", ((.51, .1), (.1, .251)))
def test_theorem_domain_is_enforced(alpha, delta):
    with pytest.raises(ValueError, match="Lemma 3"):
        audit({}, alpha, delta)
    with pytest.raises(ValueError, match="Lemma 3"):
        assert_tail_position(100, alpha, delta)


def test_installed_entry_runs_bundled_sample(tmp_path):
    manifest = EXAMPLES / "minimal" / "manifest.json"
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    run = subprocess.run((sys.executable, "-m", "holt", "audit", str(manifest),
                          "--no-embeddings", "--output", str(tmp_path)),
                         capture_output=True, text=True, env=env)
    assert run.returncode == 0, run.stderr
    assert (tmp_path / "certificate.json").is_file()
    assert (tmp_path / "hard-set.tsv").is_file()


def test_missing_search_binary_names_install_command(tmp_path):
    with pytest.raises(FileNotFoundError, match="brew install mmseqs2"):
        binary("mmseqs", tmp_path / "missing", MMSEQS_INSTALL)


def test_direct_split_requires_structure_evidence(tmp_path):
    assert main(["audit", "--split-dir", str(EXAMPLES / "minimal"),
                 "--no-embeddings", "--output", str(tmp_path)]) == 2
    assert not (tmp_path / "certificate.json").exists()


@pytest.mark.parametrize("name,dataset,cells", (
    ("deepfri-original-ec", "ec", (("mmseqs30", "ec", 37), ("pdb", "ec", 41))),
    ("deepfri-original-go", "go", (("mmseqs30", "bp", 109), ("pdb", "bp", 63),
                                    ("mmseqs30", "mf", 0), ("pdb", "mf", 0),
                                    ("mmseqs30", "cc", 64), ("pdb", "cc", 0))),
))
def test_frozen_chain_membership(name, dataset, cells):
    manifest = EXAMPLES / f"{name}.json"
    if not manifest.is_file():
        pytest.skip("full-input benchmark manifest is absent")
    config = load_manifest(manifest)
    result, selected = audit(config)
    for tier, task, expected in cells:
        path = SETS / "deepfri" / f"{task}-{tier}-a010" / "chains.tsv"
        with path.open(newline="") as handle:
            prior = {row["id"] for row in csv.DictReader(handle, delimiter="\t")}
        current = {row["id"] for row in selected if row["tier"] == tier and row["task"] == task}
        assert len(prior) == len(current) == expected
        assert current == prior
        assert result["summary"][tier][task]["certified"] == expected
        assert result["training"][tier]["role_checks"][task]["one_per_family"]
        assert result["training"][tier]["role_checks"][task]["no_representative_copy_source"]
        assert result["training"][tier]["role_checks"][task]["design_excludes_representatives"]


def test_prespecified_mf_secondary_level():
    manifest = EXAMPLES / "deepfri-original-go.json"
    if not manifest.is_file():
        pytest.skip("full-input benchmark manifest is absent")
    result, selected = audit(load_manifest(manifest), alpha=.15)
    for tier, expected in (("mmseqs30", 128), ("pdb", 117)):
        path = SETS / "deepfri" / f"mf-{tier}-a015" / "chains.tsv"
        with path.open(newline="") as handle:
            prior = {row["id"] for row in csv.DictReader(handle, delimiter="\t")}
        current = {row["id"] for row in selected if row["tier"] == tier and row["task"] == "mf"}
        assert len(prior) == len(current) == expected
        assert prior == current
        assert result["per_candidate_level"] == .025


def test_protocol_count_pays_bonferroni_factor():
    config = load_manifest(EXAMPLES / "minimal" / "manifest.json", no_embeddings=True)
    result, _ = audit(config, protocols_k=4)
    assert result["per_candidate_level"] == .00625


def test_bundled_reproduction():
    bundle = PACKAGE_ROOT / "certified_sets" / "replay.json.gz"
    if not bundle.is_file():
        pytest.skip("bundled reproduction data is absent")
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    run = subprocess.run((sys.executable, "-m", "holt", "reproduce"),
                         cwd=PACKAGE_ROOT, capture_output=True, text=True, env=env)
    assert run.returncode == 0, run.stderr
    assert "Certified lists: 10/10 exact" in run.stdout
    assert "Paired scores: 93 Fmax and 161 deltas exact" in run.stdout
