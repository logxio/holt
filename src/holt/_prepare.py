"""Generate the two search routes needed for a new split's no-embedding audit."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


MMSEQS_INSTALL = "brew install mmseqs2 (https://github.com/soedinglab/MMseqs2#installation)"
FOLDSEEK_INSTALL = ("macOS universal binary: https://mmseqs.com/foldseek/foldseek-osx-universal.tar.gz "
                    "(https://github.com/steineggerlab/foldseek#installation)")


def binary(name, supplied, instruction):
    found = str(supplied) if supplied else shutil.which(name)
    if not found or not Path(found).is_file() or not os.access(found, os.X_OK):
        raise FileNotFoundError(f"{name} executable missing; install {instruction} or pass --{name} PATH")
    return str(Path(found).resolve())


def structure_dir(root, part):
    for suffix in ("pdb_ca", "pdb"):
        path = root / part / suffix
        if path.is_dir():
            return path
    raise FileNotFoundError(f"Foldseek structures missing: expected {root / part / 'pdb_ca'} "
                            f"or {root / part / 'pdb'} (one chain per file)")


def run(command, log):
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as handle:
        handle.write("command: " + " ".join(command) + "\n")
        handle.flush()
        result = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        tail = "\n".join(log.read_text(errors="replace").splitlines()[-15:])
        raise RuntimeError(f"search failed with exit {result.returncode}: {log}\n{tail}")


def prepare(root: Path, output: Path, mmseqs_path=None, foldseek_path=None,
            no_embeddings=False, esm2_model="facebook/esm2_t33_650M_UR50D",
            esmc_model="biohub/ESMC-600M", esm2_python=None, esmc_python=None):
    root = root.resolve()
    for part in ("train", "test"):
        for suffix in ("jsonl", "fasta"):
            path = root / f"{part}.{suffix}"
            if not path.is_file():
                raise FileNotFoundError(f"split input missing: {path}")
    structures = {part: structure_dir(root, part) for part in ("train", "test")}
    for part in ("train", "test"):
        with (root / f"{part}.jsonl").open() as handle:
            ids = [json.loads(line)["id"] for line in handle if line.strip()]
        missing = [name for name in ids if not any(
            (structures[part] / f"{name}{suffix}").is_file()
            for suffix in (".pdb", ".cif", ".pdb.gz", ".cif.gz"))]
        if missing:
            raise FileNotFoundError(f"Foldseek structure files missing for {part}: "
                                    f"{len(missing)} IDs, first {missing[:5]}")
    mmseqs = binary("mmseqs", mmseqs_path, MMSEQS_INSTALL)
    foldseek = binary("foldseek", foldseek_path, FOLDSEEK_INSTALL)
    evidence = (output / "evidence").resolve()
    evidence.mkdir(parents=True, exist_ok=True)
    scratch = evidence / "scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    train_hits = {}
    test_hits = {}
    for part, destination in (("train", train_hits), ("test", test_hits)):
        for route, executable in (("mmseqs", mmseqs), ("foldseek", foldseek)):
            target = evidence / f"{part}_{route}.tsv"
            partial = target.with_suffix(".tsv.part")
            partial.unlink(missing_ok=True)
            work = scratch / f"{part}_{route}"
            query = root / f"{part}.fasta" if route == "mmseqs" else structures[part]
            library = root / "train.fasta" if route == "mmseqs" else structures["train"]
            command = [executable, "easy-search", str(query), str(library), str(partial), str(work),
                       "--threads", "1", "--max-seqs", "300", "--format-output",
                       ("query,target,fident,alnlen,qlen,tlen,bits" if route == "mmseqs"
                        else "query,target,bits,fident,alntmscore"), "-v", "1"]
            if route == "mmseqs":
                command += ["--split-memory-limit", "2G"]
            run(command, evidence / f"{part}_{route}.log")
            partial.replace(target)
            destination[route] = str(target)
            shutil.rmtree(work, ignore_errors=True)
    cluster_prefix = evidence / "train_mmseqs30"
    cluster = Path(str(cluster_prefix) + "_cluster.tsv")
    run([mmseqs, "easy-cluster", str(root / "train.fasta"), str(cluster_prefix),
         str(scratch / "cluster"), "--min-seq-id", "0.3", "-c", "0.8", "--cov-mode", "0",
         "--threads", "1", "-v", "1"], evidence / "train_mmseqs30.log")
    if not cluster.is_file():
        raise FileNotFoundError(f"MMseqs2 did not produce cluster table: {cluster}")
    shutil.rmtree(scratch, ignore_errors=True)
    for suffix in ("_all_seqs.fasta", "_rep_seq.fasta"):
        Path(str(cluster_prefix) + suffix).unlink(missing_ok=True)
    with (root / "train.jsonl").open() as handle:
        first = json.loads(handle.readline())
    manifest = {"task_family": "ec" if "ec" in first["labels"] else "go",
                "train_jsonl": str(root / "train.jsonl"), "train_fasta": str(root / "train.fasta"),
                "test_jsonl": str(root / "test.jsonl"), "test_fasta": str(root / "test.fasta"),
                "train_cluster": str(cluster), "train_hits": train_hits, "test_hits": test_hits}
    if not no_embeddings:
        manifest["train_embeddings"] = {}
        for route, source, python in (("esm2-650m", esm2_model, esm2_python),
                                      ("esmc-600m", esmc_model, esmc_python)):
            executable = str(python or sys.executable)
            run([executable, "-m", "holt._embed", route, str(root), str(evidence),
                 "--model", str(source)], evidence / f"{route}.log")
            prefix = evidence / route
            manifest["train_embeddings"][route] = {
                "ids": str(prefix.with_name(prefix.name + "-train-ids.txt")),
                "vectors": str(prefix.with_name(prefix.name + "-train-last.npy"))}
            manifest["test_hits"][route] = str(prefix.with_name(prefix.name + "-test-hits.tsv"))
    path = evidence / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return path
