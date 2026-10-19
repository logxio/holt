"""Command-line entry point for the HoLT audit."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import resource
import sys
import time
from pathlib import Path

from . import __version__

def evidence_hashes(config):
    from ._audit import sha256

    paths = {}
    for key, value in config.items():
        if isinstance(value, Path):
            paths[key] = value
        elif isinstance(value, dict):
            for name, nested in value.items():
                if isinstance(nested, Path):
                    paths[f"{key}.{name}"] = nested
                elif isinstance(nested, dict):
                    for item, path in nested.items():
                        if isinstance(path, Path):
                            paths[f"{key}.{name}.{item}"] = path
    return {name: {"path": str(path), "sha256": sha256(path)} for name, path in paths.items()}


def main(argv=None):
    parser = argparse.ArgumentParser(prog="holt", description="Audit a held-out protein split with HoLT")
    sub = parser.add_subparsers(dest="command", required=True)
    replay = sub.add_parser("reproduce", help="recompute the bundled DeepFRI certificates and paired scores on CPU")
    replay.add_argument("--data", type=Path, default=Path("certified_sets/replay.json.gz"))
    replay.add_argument("--sets", type=Path, default=Path("certified_sets"))
    command = sub.add_parser("audit", help="fit on training chains, then apply the frozen rule to test chains")
    command.add_argument("manifest", type=Path, nargs="?", help="JSON paths to splits and search evidence")
    command.add_argument("--split-dir", type=Path,
                         help="generate MMseqs2/Foldseek evidence from train/test files and structures")
    command.add_argument("--mmseqs", type=Path, help="MMseqs2 executable for --split-dir")
    command.add_argument("--foldseek", type=Path, help="Foldseek executable for --split-dir")
    command.add_argument("--esm2-model", default="facebook/esm2_t33_650M_UR50D",
                         help="locally cached ESM-2 weights or model directory")
    command.add_argument("--esmc-model", default="biohub/ESMC-600M",
                         help="locally cached ESM C weights or bfloat16 model directory")
    command.add_argument("--esm2-python", type=Path, help="Python with ESM-2 dependencies")
    command.add_argument("--esmc-python", type=Path, help="Python with ESM C dependencies")
    command.add_argument("--output", type=Path, default=Path("holt-output"))
    command.add_argument("--alpha", type=float, default=.1)
    command.add_argument("--delta", type=float, default=.1)
    command.add_argument("--protocols-k", type=int, default=1,
                         help="number of protocols compared on the same labels (default: 1)")
    command.add_argument("--no-embeddings", action="store_true",
                         help="fit a separate two-search-route certificate with zero embedding features")
    args = parser.parse_args(argv)
    try:
        if args.command == "reproduce":
            from ._reproduce import reproduce

            return reproduce(args.data, args.sets)
        if not 0 < args.alpha <= .5:
            raise ValueError("alpha must satisfy 0 < alpha <= 1/2 (Lemma 3)")
        if not 0 < args.delta <= .25:
            raise ValueError("delta must satisfy 0 < delta <= 1/4 (Lemma 3)")
        if args.protocols_k < 1:
            raise ValueError("--protocols-k must be a positive integer")
        if (args.manifest is None) == (args.split_dir is None):
            raise ValueError("provide exactly one manifest or --split-dir")
        from ._audit import audit, load_manifest, sha256

        started = time.monotonic()
        args.output.mkdir(parents=True, exist_ok=True)
        if args.split_dir is not None:
            from ._prepare import prepare

            manifest = prepare(args.split_dir, args.output, args.mmseqs, args.foldseek,
                               args.no_embeddings, args.esm2_model, args.esmc_model,
                               args.esm2_python, args.esmc_python)
        else:
            manifest = args.manifest.resolve()
        config = load_manifest(manifest, args.no_embeddings)
        result, hard = audit(config, args.alpha, args.delta, args.protocols_k)
        result["package_version"] = __version__
        maximum_rss = max(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                          resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)
        result["runtime"] = {"seconds": time.monotonic() - started,
                             "peak_rss_mb": maximum_rss / (1024**2 if platform.system() == "Darwin" else 1024)}
        result["inputs"] = evidence_hashes(config)
        result["manifest_sha256"] = sha256(manifest)
        certificate = args.output / "certificate.json"
        hard_set = args.output / "hard-set.tsv"
        temporary = certificate.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        temporary.replace(certificate)
        temporary = hard_set.with_suffix(".tsv.tmp")
        with temporary.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=("tier", "task", "id", "score", "test_copy_event"),
                                    delimiter="\t")
            writer.writeheader()
            writer.writerows(hard)
        temporary.replace(hard_set)
        print(f"certificate: {certificate}")
        print(f"hard set: {hard_set}")
        print(f"runtime: {result['runtime']['seconds']:.1f}s; peak RSS: {result['runtime']['peak_rss_mb']:.1f} MB")
        for tier, tasks in result["summary"].items():
            for task, row in tasks.items():
                upper = row["calibration_upper_bound"]
                bound = f"{upper:.4f}" if upper is not None else "none"
                print(f"{task.upper()} [{tier}]: certified {row['certified']}/{row['eligible_test']}; "
                      f"calibration risk upper bound {bound}; test copy events "
                      f"{row['test_copy_events']}/{row['certified']}")
                if row["hard_ids"]:
                    print("  " + ", ".join(row["hard_ids"]))
        return 0
    except (KeyError, ValueError, FileNotFoundError, OSError, RuntimeError) as exc:
        print(f"holt: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
