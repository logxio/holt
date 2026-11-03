"""Run HMMER phmmer against training sequences and retain ranked chain hits.

The train crossfit table includes every training query.  HMMER computes each
query/target bit score independently; after search we remove the query's entire
MMseqs2 30% cluster.  The score cutoff is in bits, so database E-value changes
cannot change this ranking.  The certification code can further mask its
design/calibration reference library when reading this table.
"""
from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import resource
import subprocess
import tempfile
import time
from pathlib import Path


DATA = {"ec": Path("data/seqfix/ec"), "go": Path("data/prepared/go")}
RAW = Path("results/raw")


def fasta(path):
    name, pieces = None, []
    with path.open() as stream:
        for line in stream:
            if line.startswith(">"):
                if name is not None:
                    yield name, "".join(pieces)
                name, pieces = line[1:].split()[0], []
            else:
                pieces.append(line.strip())
        if name is not None:
            yield name, "".join(pieces)


def cluster_map(dataset):
    mapping = {}
    with (RAW / f"crossfit-{dataset}-mmseqs30_cluster.tsv").open() as stream:
        for line in stream:
            rep, member = line.rstrip("\n").split("\t")[:2]
            if member in mapping:
                raise ValueError(f"duplicate cluster member: {member}")
            mapping[member] = rep
    return mapping


def select(path, limit):
    rows = list(fasta(path))
    if len(rows) != len({name for name, _ in rows}):
        raise ValueError(f"duplicate FASTA identifiers: {path}")
    if limit:
        wanted = set(sorted((name for name, _ in rows),
                            key=lambda name: hashlib.sha256(name.encode()).digest())[:limit])
        rows = [(name, seq) for name, seq in rows if name in wanted]
    return rows


def normalize(table, output, query_ids, train_ids, clusters, train_crossfit, top):
    heaps = {name: [] for name in query_ids}
    accepted = 0
    with table.open() as stream:
        for line in stream:
            if not line or line[0] == "#":
                continue
            fields = line.split()
            if len(fields) < 6:
                raise ValueError(f"short HMMER tblout line: {line[:200]}")
            target, query = fields[0], fields[2]
            if query not in heaps or target not in train_ids:
                raise ValueError(f"unexpected HMMER pair: {query} {target}")
            if train_crossfit and clusters[query] == clusters[target]:
                continue
            score = float(fields[5])
            heap = heaps[query]
            item = (score, target)
            if len(heap) < top:
                heapq.heappush(heap, item)
            elif item > heap[0]:
                heapq.heapreplace(heap, item)
            accepted += 1
    with output.open("w") as stream:
        for query in sorted(heaps):
            for score, target in sorted(heaps[query], reverse=True):
                stream.write(f"{query}\t{target}\t{score:.6f}\n")
    return {"reported_pairs_after_cluster_filter": accepted,
            "queries_with_hits": sum(bool(x) for x in heaps.values()),
            "retained_hits": sum(map(len, heaps.values()))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=DATA)
    parser.add_argument("part", choices=("train-crossfit", "test"))
    parser.add_argument("--limit", type=int, default=0, help="pilot: deterministic first N query IDs by SHA-256")
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--top", type=int, default=300)
    parser.add_argument("--phmmer", default="phmmer")
    args = parser.parse_args()
    if args.limit < 0 or args.threads < 1 or args.top < 5:
        parser.error("limit >= 0, threads >= 1 and top >= 5 are required")
    data = DATA[args.dataset]
    train_path = data / "train.fasta"
    query_path = data / ("train.fasta" if args.part == "train-crossfit" else "test.fasta")
    train_ids = {name for name, _ in fasta(train_path)}
    queries = select(query_path, args.limit)
    clusters = cluster_map(args.dataset) if args.part == "train-crossfit" else {}
    if clusters and train_ids != set(clusters):
        raise ValueError("training FASTA and frozen MMseqs2 30% clusters disagree")
    if not queries:
        raise ValueError("empty query FASTA")
    RAW.mkdir(parents=True, exist_ok=True)
    tag = f"pilot{args.limit}-" if args.limit else ""
    output = RAW / f"profile-{args.dataset}-{tag}{args.part}-hits.tsv"
    report_path = RAW / f"profile-{args.dataset}-{tag}{args.part}-timing.json"
    start = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="motif-phmmer-") as tmp:
        tmp = Path(tmp)
        source = tmp / "queries.fasta"
        with source.open("w") as stream:
            for name, seq in queries:
                stream.write(f">{name}\n{seq}\n")
        table = tmp / "phmmer.tbl"
        command = [args.phmmer, "--cpu", str(args.threads), "--noali", "-E", "10",
                   "--domE", "10", "--tblout", str(table), "-o", str(tmp / "stdout.txt"),
                   str(source), str(train_path)]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(f"phmmer exit {result.returncode}: {result.stderr[-6000:]}")
        counts = normalize(table, output, {name for name, _ in queries}, train_ids, clusters,
                           args.part == "train-crossfit", args.top)
    report = {"method": "HMMER phmmer", "dataset": args.dataset, "part": args.part,
              "queries": len(queries), "reference_chains": len(train_ids),
              "sequence_source": str(data), "cluster_source":
              str(RAW / f"crossfit-{args.dataset}-mmseqs30_cluster.tsv") if clusters else None,
              "score": "full-sequence bit score", "reported_evalue_limit": 10,
              "top_hits_per_query": args.top, "threads": args.threads,
              "seconds": time.perf_counter() - start,
              "max_rss_kib": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
              "output": str(output), "output_bytes": output.stat().st_size,
              "command": command, **counts}
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
