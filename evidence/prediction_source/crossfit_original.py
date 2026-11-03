"""Isolated original-DeepFRI input adapter for frozen HoLT-2.

The existing crossfit modules remain unchanged while ProteinShake is running.
Generated files use orig-ec/orig-go, never the processed DeepFRI prefixes.
"""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path("data/prepared/deepfri")
RAW = Path("results/raw")


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def names(path):
    with path.open() as handle:
        result = [json.loads(line)["id"] for line in handle]
    if len(result) != len(set(result)):
        raise ValueError(f"duplicate ID in {path}")
    return set(result)


def decode(name):
    if not name.startswith("X"):
        raise ValueError(f"not an encoded Foldseek ID: {name}")
    original = bytes.fromhex(name[1:]).decode("ascii")
    if "-" not in original or "X" + original.encode("ascii").hex().upper() != name:
        raise ValueError(f"noncanonical Foldseek ID: {name}")
    return original


def decode_table(source, output, queries, training):
    temporary = output.with_name(output.name + ".part")
    seen, count = set(), 0
    try:
        with source.open() as inp, temporary.open("w") as out:
            for line in inp:
                fields = line.rstrip("\n").split("\t")
                if len(fields) < 3:
                    raise ValueError(f"short Foldseek row: {source}")
                query, hit = decode(fields[0]), decode(fields[1])
                if query not in queries or hit not in training:
                    raise ValueError(f"outside original {source}: {query} {hit}")
                fields[:2] = [query, hit]
                out.write("\t".join(fields) + "\n")
                seen.add(query)
                count += 1
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return {"rows": count, "queries_with_hits": len(seen), "sha256": sha(output)}


def paths(dataset, part):
    ds = dataset.removeprefix("orig-")
    return {"mmseqs": RAW / f"audit-deepfri-original-{ds}-mmseqs-{part}.tsv",
            "foldseek": RAW / f"crossfit-orig-{ds}-{part}-foldseek-hits.tsv",
            "esm2": RAW / f"esm2-650m-deepfri-{ds}-{part}-hits.tsv",
            "esmc": RAW / f"esmc-600m-deepfri-{ds}-{part}-hits.tsv"}


def check(ds):
    data, dataset = ROOT / ds, f"orig-{ds}"
    train = names(data / "train.jsonl")
    report = {"data_root": str(data), "train": len(train), "parts": {}, "vectors": {}}
    for model in ("esm2-650m", "esmc-600m"):
        base = Path("data/plm") / model / "deepfri"
        path = base / f"{ds}-train-ids.txt"
        ids = path.read_text().splitlines()
        if len(ids) != len(train) or set(ids) != train:
            raise ValueError(f"original vector IDs disagree: {model} {ds}")
        report["vectors"][model] = {"ids_sha256": sha(path),
                                     "matrix": str(base / f"{ds}-train-last.npy")}
    for part in ("valid", "test"):
        held = names(data / f"{part}.jsonl")
        original = RAW / f"foldseek-deepfri-{ds}-{part}-hits.tsv"
        output = paths(dataset, part)["foldseek"]
        entry = {"n": len(held), "foldseek": decode_table(original, output, held, train),
                 "foldseek_source_sha256": sha(original)}
        for route, path in paths(dataset, part).items():
            if not path.is_file() or not path.stat().st_size:
                raise FileNotFoundError(path)
            if route == "foldseek":
                continue
            seen, count = set(), 0
            with path.open() as handle:
                for line in handle:
                    fields = line.rstrip("\n").split("\t")
                    if len(fields) < (7 if route == "mmseqs" else 3):
                        raise ValueError(f"short {route} row: {path}")
                    if fields[0] not in held or fields[1] not in train:
                        raise ValueError(f"outside original {path}: {fields[:2]}")
                    seen.add(fields[0]); count += 1
            entry[route] = {"rows": count, "queries_with_hits": len(seen), "sha256": sha(path)}
        report["parts"][part] = entry
    output = RAW / f"crossfit-{dataset}-input-audit.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(output, report["parts"], flush=True)


def decode_self(ds):
    train = names(ROOT / ds / "train.jsonl")
    stem = RAW / f"crossfit-orig-{ds}-full-foldseek-hits.tsv"
    source = stem.with_name(stem.name + ".raw.part")
    print(decode_table(source, stem, train, train), flush=True)
    source.unlink()


def verify_search(ds):
    training = names(ROOT / ds / "train.jsonl")
    report = {"data_root": str(ROOT / ds), "training": len(training), "routes": {}, "clusters": {}}
    for route in ("mmseqs", "foldseek"):
        path = RAW / f"crossfit-orig-{ds}-full-{route}-hits.tsv"
        count, queries = 0, set()
        with path.open() as handle:
            for line in handle:
                fields = line.rstrip("\n").split("\t")
                minimum = 7 if route == "mmseqs" else 5
                if len(fields) < minimum:
                    raise ValueError(f"short training hit: {path}")
                if fields[0] not in training or fields[1] not in training:
                    raise ValueError(f"training hit outside original: {fields[:2]}")
                queries.add(fields[0]); count += 1
        report["routes"][route] = {"rows": count, "queries_with_hits": len(queries),
                                   "sha256": sha(path)}
    for part in ("train", "valid", "test"):
        path = (RAW / f"crossfit-orig-{ds}-mmseqs30_cluster.tsv" if part == "train" else
                RAW / f"certify-mmseqs30-old-orig-{ds}-{part}_cluster.tsv")
        expected = names(ROOT / ds / f"{part}.jsonl")
        members, reps = [], set()
        with path.open() as handle:
            for line in handle:
                rep, member = line.rstrip("\n").split("\t")[:2]
                members.append(member); reps.add(rep)
        if len(members) != len(expected) or set(members) != expected or not reps <= expected:
            raise ValueError(f"original cluster incomplete: {part} {path}")
        report["clusters"][part] = {"members": len(members), "clusters": len(reps),
                                    "sha256": sha(path)}
    path = RAW / f"crossfit-orig-{ds}-search-audit.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(path, report, flush=True)


def adapt(ds):
    import certify_four
    import crossfit_accept
    import crossfit_engine
    import crossfit_retrieve
    import profile_search

    dataset = f"orig-{ds}"
    crossfit_retrieve.DATA[dataset] = ROOT / ds
    crossfit_retrieve.VECTORS[dataset] = f"deepfri/{ds}"
    profile_search.DATA[dataset] = ROOT / ds
    old_kind = crossfit_engine.kind
    def kind(value):
        return value.removeprefix("orig-") if value.startswith("orig-") else old_kind(value)
    for module in (certify_four, crossfit_accept, crossfit_engine):
        module.kind = kind
    old_paths = certify_four.hit_files
    def hit_files(root, base, name, part):
        return paths(name, part) if name == dataset else old_paths(root, base, name, part)
    certify_four.hit_files = hit_files
    crossfit_accept.hit_files = hit_files
    old_fixed = crossfit_accept.fixed_comparison
    def fixed(name, task, eligible, selected, event, seed):
        return old_fixed(f"original-{ds}" if name == dataset else name,
                         task, eligible, selected, event, seed)
    crossfit_accept.fixed_comparison = fixed
    class NoPublishedScores:
        def __init__(self, *args):
            pass
        def evaluate(self, *args):
            return None
    crossfit_accept.Models = NoPublishedScores
    return crossfit_accept


def execute(ds, stage, adversary):
    accept = adapt(ds)
    sys.argv = [sys.argv[0], f"orig-{ds}", "--adversary", adversary]
    if stage == "certify":
        if adversary == "profile":
            sys.argv += ["--tiers", "mmseqs30"]
        import crossfit_certify
        crossfit_certify.main()
        return
    accept.main()
    output = RAW / f"crossfit-orig-{ds}-{adversary}-acceptance.json"
    result = json.loads(output.read_text())
    result["data_root"] = str(ROOT / ds)
    result["published_model_scores"] = None
    for tier in result["tiers"].values():
        for event in tier.values():
            for screen in event.values():
                for variant in screen.values():
                    for entry in variant.values():
                        entry.pop("models", None)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")


def summary(ds, adversary):
    stem = f"crossfit-orig-{ds}-{adversary}"
    certification = RAW / f"{stem}-full-certification.json"
    accepted = RAW / f"{stem}-acceptance.json"
    chain_path = RAW / f"{stem}-acceptance-chains.tsv"
    payload = json.loads(accepted.read_text())
    with chain_path.open() as handle:
        selected = [row for row in csv.DictReader(handle, delimiter="\t")
                    if row["tier"] == "mmseqs30" and row["screen"] == "learned"
                    and row["variant"] == "pooled" and row["alpha"] == "0.1"
                    and row["part"] == "test" and row["event"] in ("exact", "raw")]
    output = RAW / f"{stem}-primary-chains.tsv"
    columns = ("tier", "screen", "variant", "alpha", "task", "event", "part", "id", "pdb", "leak")
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t")
        writer.writeheader(); writer.writerows(selected)
    report = {"data_root": str(ROOT / ds), "adversary": adversary, "alpha": 0.1,
              "protocol_sha256": sha(RAW / "crossfit-protocol-v2.json"),
              "certification_sha256": sha(certification), "acceptance_sha256": sha(accepted),
              "primary_chains": str(output), "primary_chains_sha256": sha(output), "tasks": {}}
    for event, screens in payload["tiers"]["mmseqs30"].items():
        if event.endswith(("_exact", "_raw")):
            test = screens["learned"]["pooled"]["0.1"]["test"]
            report["tasks"][event] = {name: test.get(name) for name in
                                      ("n", "leaks", "rate", "wilson95", "exact95", "families", "family_ci95")}
    path = RAW / f"{stem}-primary-summary.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(path, report["tasks"], flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("check", "decode-self", "verify-search", "profile-train", "profile-test", "certify", "accept", "summary"))
    parser.add_argument("dataset", choices=("ec", "go"))
    parser.add_argument("--adversary", choices=("homology", "profile"), default="homology")
    args = parser.parse_args()
    if args.stage == "check":
        check(args.dataset)
    elif args.stage == "decode-self":
        decode_self(args.dataset)
    elif args.stage == "verify-search":
        verify_search(args.dataset)
    elif args.stage.startswith("profile-"):
        adapt(args.dataset)
        import profile_search
        part = "train-crossfit" if args.stage == "profile-train" else "test"
        sys.argv = [sys.argv[0], f"orig-{args.dataset}", part, "--threads", "4"]
        profile_search.main()
    elif args.stage in ("certify", "accept"):
        execute(args.dataset, args.stage, args.adversary)
    else:
        summary(args.dataset, args.adversary)


if __name__ == "__main__":
    main()
