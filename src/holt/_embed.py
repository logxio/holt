"""Compute one language-model route for a new split in a separate process."""

from __future__ import annotations

import argparse
import json
import resource
from pathlib import Path

import numpy as np

from ._common import records, sequences


def model_and_tokenizer(route, source, device):
    try:
        import torch
        if route == "esm2-650m":
            from transformers import AutoTokenizer, EsmModel
            tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
            model = EsmModel.from_pretrained(source, torch_dtype=torch.float16,
                                             add_pooling_layer=False, local_files_only=True)
            model = model.to(device).eval()
        else:
            from esm.models.esmc import EsmcForMaskedLM, EsmcTokenizer
            tokenizer = EsmcTokenizer.from_pretrained(source, local_files_only=True)
            model = EsmcForMaskedLM.from_pretrained(
                source, device="cpu", dtype=torch.bfloat16,
                attn_implementation="sdpa", local_files_only=True).esmc.to(device).eval()
        return model, tokenizer
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            f"{route} needs local model weights and the embedding extra; "
            "install 'holt-audit[embeddings]' and pass --esm2-model/--esmc-model "
            f"to a downloaded model directory: {exc}") from exc


def windows(sequence, limit=1000):
    count = (len(sequence) + limit - 1) // limit
    if not count:
        raise ValueError("empty protein sequence")
    width, extra = divmod(len(sequence), count)
    offset = 0
    for i in range(count):
        size = width + (i < extra)
        yield sequence[offset:offset + size]
        offset += size


def vectors(root, route, source):
    import torch

    torch.set_num_threads(1)
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = model_and_tokenizer(route, source, device)
    names = {}
    output = {}
    with torch.inference_mode():
        for part in ("train", "test"):
            seqs = sequences(root / f"{part}.fasta")
            names[part] = list(seqs)
            rows = []
            for name, sequence in seqs.items():
                weighted = None
                for piece in windows(sequence):
                    encoded = tokenizer([piece], return_tensors="pt", padding=True,
                                        pad_to_multiple_of=64).to(device)
                    if route == "esmc-600m":
                        mask = encoded["attention_mask"].bool()
                        states = model(input_ids=encoded["input_ids"],
                                       attention_mask=mask).last_hidden_state
                    else:
                        states = model(**encoded).last_hidden_state
                    # Both tokenizers use BOS, residues, EOS.
                    if int(encoded["attention_mask"].sum()) != len(piece) + 2:
                        raise ValueError(f"token/residue count mismatch for {name}")
                    mean = states[0, 1:len(piece) + 1].float().mean(0).cpu().numpy()
                    contribution = mean * (len(piece) / len(sequence))
                    weighted = contribution if weighted is None else weighted + contribution
                    del states, encoded
                rows.append(weighted)
            output[part] = np.stack(rows).astype(np.float16)
    return names, output, device


def hits(path, train_names, query_names, train_vectors, query_vectors, labels):
    train = train_vectors.astype(np.float32)
    query = query_vectors.astype(np.float32)
    train /= np.maximum(np.linalg.norm(train, axis=1, keepdims=True), 1e-12)
    query /= np.maximum(np.linalg.norm(query, axis=1, keepdims=True), 1e-12)
    tasks = tuple(next(iter(labels.values())))
    annotated = {task: np.array([i for i, name in enumerate(train_names)
                                  if labels[name][task]], dtype=np.int32) for task in tasks}
    with path.open("w") as handle:
        for offset in range(0, len(query_names), 128):
            block = query[offset:offset + 128] @ train.T
            for i, scores in enumerate(block):
                count = min(500, len(train_names))
                picks = set(map(int, np.argpartition(scores, -count)[-count:]))
                for task in tasks:
                    pool = annotated[task]
                    if len(pool):
                        top = min(5, len(pool))
                        picks.update(map(int, pool[np.argpartition(scores[pool], -top)[-top:]]))
                for j in sorted(picks, key=lambda j: (scores[j], train_names[j]), reverse=True):
                    handle.write(f"{query_names[offset + i]}\t{train_names[j]}\t{scores[j]:.8f}\n")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("route", choices=("esm2-650m", "esmc-600m"))
    parser.add_argument("root", type=Path)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("--model", required=True)
    args = parser.parse_args(argv)
    names, matrices, device = vectors(args.root, args.route, args.model)
    prefix = args.evidence / args.route
    ids = prefix.with_name(prefix.name + "-train-ids.txt")
    matrix = prefix.with_name(prefix.name + "-train-last.npy")
    table = prefix.with_name(prefix.name + "-test-hits.tsv")
    ids.write_text("\n".join(names["train"]) + "\n")
    np.save(matrix, matrices["train"])
    labels = {row["id"]: row["labels"] for row in records(args.root / "train.jsonl")}
    if set(names["train"]) != set(labels):
        raise ValueError("embedding train FASTA and JSONL IDs disagree")
    hits(table, names["train"], names["test"], matrices["train"], matrices["test"], labels)
    print(json.dumps({"route": args.route, "device": device, "train": len(names["train"]),
                      "test": len(names["test"]), "peak_rss_bytes": resource.getrusage(
                          resource.RUSAGE_SELF).ru_maxrss}), flush=True)


if __name__ == "__main__":
    main()
