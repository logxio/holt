"""Training-to-training retrieval with same-PDB exclusion, using existing indices/vectors.

Run a 500-query pilot before a full run. The full score vector and annotated
top-five panel are produced from the same search settings as certify_four.py.
EC sequence routes read the restored residue letters (data/seqfix/ec and the
ec-seqfix vectors, scripts/restore_ec_sequences.py), as the held-out tables do.
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np

from certify_split import records, sequences


RAW = Path("results/raw")
DATA = {"ec": Path("data/seqfix/ec"), "go": Path("data/prepared/go")}
VECTORS = {"ec": "ec-seqfix", "go": "go"}
# ProteinShake's six splits (scripts/fetch_external.py) as further datasets: "ps-<task>-<split>". Their
# hit tables come from scripts/proteinshake_search.sh, their vectors from scripts/proteinshake_embed.py.
PROTEINSHAKE = [f"ps-{task}-{split}" for task in ("ec", "go") for split in ("random", "sequence", "structure")]
for _name in PROTEINSHAKE:
    DATA[_name] = Path("data/external/proteinshake") / _name[3:]
    VECTORS[_name] = _name
FORMATS = {"mmseqs": "query,target,fident,alnlen,qlen,tlen,bits",
           "foldseek": "query,target,bits,fident,alntmscore"}


def selected(dataset, limit):
    data = DATA[dataset]
    rows = records(data / "train.jsonl")
    ids = [row["id"] for row in rows]
    if limit:
        ids = sorted(ids, key=lambda name: __import__("hashlib").sha256(name.encode()).digest())[:limit]
    return ids


def query_sources(dataset, ids, tag):
    data = DATA[dataset]
    if tag == "full":
        return data / "train.fasta", data / "train" / "pdb_ca", 0
    seq = sequences(data / "train.fasta")
    fasta = RAW / f"crossfit-{dataset}-{tag}-query.fasta"
    with fasta.open("w") as handle:
        for name in ids:
            handle.write(f">{name}\n{seq[name]}\n")
    structures = RAW / f"crossfit-{dataset}-{tag}-query-pdb_ca"
    structures.mkdir(exist_ok=True)
    source = data / "train" / "pdb_ca"
    missing = 0
    for name in ids:
        target = source / f"{name}.pdb"
        if target.exists():
            link = structures / target.name
            if not link.exists():
                link.symlink_to(target.resolve())
        else:
            missing += 1
    return fasta, structures, missing


def search(dataset, tag, route, source, threads=4):
    output = RAW / f"crossfit-{dataset}-{tag}-{route}-hits.tsv"
    scratch = Path("data/tmp/crossfit") / f"{dataset}-{tag}-{route}"
    scratch.parent.mkdir(parents=True, exist_ok=True)
    # MMseqs2 searches the training FASTA itself (the held-out audit tables do the same).
    db = DATA[dataset] / "train.fasta" if route == "mmseqs" else Path("data/index/foldseek") / dataset / "train_ca"
    binary = Path("data/tools") / route / "bin" / route
    command = [str(binary), "easy-search", str(source), str(db), str(output), str(scratch),
               "--threads", str(threads), "--max-seqs", "300", "--format-output", FORMATS[route]]
    if route == "mmseqs":
        command += ["--split-memory-limit", "2G"]
    start = time.perf_counter()
    result = subprocess.run(command, capture_output=True, text=True)
    elapsed = time.perf_counter() - start
    if result.returncode:
        raise RuntimeError(f"{command}: {result.returncode}\n{result.stderr[-6000:]}")
    shutil.rmtree(scratch, ignore_errors=True)
    return {"seconds": elapsed, "output": str(output), "bytes": output.stat().st_size,
            "child_maxrss": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss}


def cosine(dataset, tag, ids, model):
    base = Path("data/plm") / model
    names = (base / f"{VECTORS[dataset]}-train-ids.txt").read_text().splitlines()
    if len(names) != len(set(names)):
        raise ValueError(f"duplicate vector IDs: {base} {dataset}")
    loc = {name: i for i, name in enumerate(names)}
    missing = set(ids) - set(loc)
    if missing:
        raise ValueError(f"missing {len(missing)} query vectors: {base} {dataset}")
    matrix = np.asarray(np.load(base / f"{VECTORS[dataset]}-train-last.npy"), dtype=np.float32)
    if len(matrix) != len(names):
        raise ValueError(f"vector/ID count mismatch: {base} {dataset}")
    matrix /= np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)
    pdb = np.array([name.split("-", 1)[0] for name in names])
    # Top-20 overall plus the five best annotated hits per ontology, so the
    # top-5 vote panel never runs short on sparsely annotated ontologies.
    train = {row["id"]: row["labels"] for row in records(DATA[dataset] / "train.jsonl")}
    tasks = ("ec",) if dataset == "ec" else ("bp", "mf", "cc")
    annotated = {task: np.flatnonzero([bool(train[name][task]) for name in names]) for task in tasks}
    output = RAW / f"crossfit-{dataset}-{tag}-{model}-hits.tsv"
    start = time.perf_counter()
    with output.open("w") as handle:
        for offset in range(0, len(ids), 128):
            query_ids = ids[offset:offset + 128]
            positions = np.array([loc[name] for name in query_ids])
            scores = matrix[positions] @ matrix.T
            for i, name in enumerate(query_ids):
                scores[i, pdb == name.split("-", 1)[0]] = -np.inf
                picks = set(map(int, np.argpartition(scores[i], -20)[-20:]))
                for task in tasks:
                    pool = annotated[task][np.isfinite(scores[i, annotated[task]])]
                    if len(pool):
                        k = min(5, len(pool))
                        picks.update(map(int, pool[np.argpartition(scores[i, pool], -k)[-k:]]))
                for j in sorted(picks, key=lambda j: scores[i, j], reverse=True):
                    if np.isfinite(scores[i, j]):
                        handle.write(f"{name}\t{names[j]}\t{scores[i,j]:.8f}\n")
    return {"seconds": time.perf_counter() - start, "output": str(output),
            "bytes": output.stat().st_size,
            "self_maxrss": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=("ec", "go"))
    parser.add_argument("--limit", type=int, default=500, help="0 selects all train queries")
    parser.add_argument("--routes", nargs="+", default=["mmseqs", "foldseek", "esm2-650m", "esmc-600m"])
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
    ids = selected(args.dataset, args.limit)
    tag = "pilot500" if args.limit == 500 else "full" if args.limit == 0 else f"pilot{args.limit}"
    fasta, structures, missing = query_sources(args.dataset, ids, tag)
    report = {"dataset": args.dataset, "tag": tag, "n_queries": len(ids),
              "n_training": len(records(DATA[args.dataset] / "train.jsonl")),
              "missing_structures": missing, "threads": args.threads, "routes": {}}
    for route, source in (("mmseqs", fasta), ("foldseek", structures)):
        if route not in args.routes:
            continue
        report["routes"][route] = search(args.dataset, tag, route, source, args.threads)
        print(route, report["routes"][route], flush=True)
    for model in ("esm2-650m", "esmc-600m"):
        if model not in args.routes:
            continue
        report["routes"][model] = cosine(args.dataset, tag, ids, model)
        print(model, report["routes"][model], flush=True)
    path = RAW / f"crossfit-{args.dataset}-{tag}-timing.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(path, flush=True)


if __name__ == "__main__":
    main()
