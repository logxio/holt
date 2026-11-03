"""Protein language model mean embeddings and cosine nearest neighbours over train.

Per dataset: embed train (= building the index), valid and test, then write the
top-100 train neighbours of each valid/test chain in the same hit format as
MMseqs2/Foldseek: query, target, then score columns. Extra score columns are
monotone sharpenings of cosine, so evaluate.py picks the vote weighting on valid
exactly as it picks k and power. Each layer is searched twice: raw vectors and
vectors z-scored per dimension with train mean and spread (suffix -z); valid picks. Runs on CUDA or Apple MPS in half precision;
batches are length-sorted and capped by a token budget so memory stays bounded.
Finished splits are kept under data/plm/<model>/ and skipped on restart;
--hits-only rewrites the neighbour files from them without loading the model.
"""
import argparse
import gc
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

MODELS = {
    "esm2-650m": ("facebook/esm2_t33_650M_UR50D", 22),
    "esmc-600m": ("data/plm/esmc-600m-bf16", 24),  # bf16 copy of biohub/ESMC-600M, scripts/fetch_esmc.sh
    "prott5-xl": ("Rostlab/prot_t5_xl_half_uniref50-enc", 16),
}
TEMPERATURES = (0.05, 0.02, 0.01)
TOP = 100
# Batches are padded to a multiple of this many tokens and sized by that padded length, so the
# MPS graph cache sees a few dozen shapes instead of one per batch (ESM C crossed 4 GB without it).
PAD = 64
DEVICE = "cuda" if torch.cuda.is_available() else "mps"


def sync():
    (torch.cuda if DEVICE == "cuda" else torch.mps).synchronize()


def fasta(path):
    name, parts = None, []
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            if name is not None:
                yield name, "".join(parts)
            name, parts = line[1:], []
        else:
            parts.append(line)
    if name is not None:
        yield name, "".join(parts)


class Embedder:
    """Mean over residue positions of the last layer and of one middle layer."""

    def __init__(self, model):
        self.kind = model
        source, mid = MODELS[model]
        self.captured = None
        if model == "esm2-650m":
            from transformers import AutoTokenizer, EsmModel
            self.tokenizer = AutoTokenizer.from_pretrained(source)
            self.model = EsmModel.from_pretrained(source, torch_dtype=torch.float16, add_pooling_layer=False)
            self.model.encoder.layer[mid - 1].register_forward_hook(self.capture)
        elif model == "esmc-600m":
            from esm.models.esmc import EsmcForMaskedLM, EsmcTokenizer
            self.tokenizer = EsmcTokenizer.from_pretrained(source)
            # ESM C is trained in bfloat16; built on the CPU and then moved, the GPU holds one copy (1.1 GB).
            self.model = EsmcForMaskedLM.from_pretrained(source, device="cpu", dtype=torch.bfloat16,
                                                         attn_implementation="sdpa").esmc
            self.model.transformer.blocks[mid - 1].register_forward_hook(self.capture)
        else:
            from transformers import T5EncoderModel, T5Tokenizer
            self.tokenizer = T5Tokenizer.from_pretrained(source, do_lower_case=False, legacy=True)
            self.model = T5EncoderModel.from_pretrained(source, torch_dtype=torch.float16)
            self.model.encoder.block[mid - 1].register_forward_hook(self.capture)
        self.model = self.model.to(DEVICE).eval()

    def capture(self, module, inputs, output):
        self.captured = output[0] if isinstance(output, tuple) else output

    @torch.inference_mode()
    def batch(self, sequences):
        if self.kind == "esmc-600m":
            enc = self.tokenizer(sequences, return_tensors="pt", padding=True, pad_to_multiple_of=PAD).to(DEVICE)
            mask = enc["attention_mask"].bool()
            last = self.model(input_ids=enc["input_ids"], attention_mask=mask).last_hidden_state
            first = 1  # BOS residues EOS
        elif self.kind == "prott5-xl":
            spaced = [" ".join(s.translate(str.maketrans("UZOB", "XXXX"))) for s in sequences]
            enc = self.tokenizer(spaced, add_special_tokens=True, padding="longest", pad_to_multiple_of=PAD,
                                 return_tensors="pt").to(DEVICE)
            mask = enc["attention_mask"].bool()
            last = self.model(**enc).last_hidden_state
            first = 0  # residues </s>
        else:
            enc = self.tokenizer(sequences, padding=True, pad_to_multiple_of=PAD, return_tensors="pt").to(DEVICE)
            mask = enc["attention_mask"].bool()
            last = self.model(**enc).last_hidden_state
            first = 1  # <cls> residues <eos>
        lengths = mask.sum(1, keepdim=True)
        position = torch.arange(mask.shape[1], device=DEVICE)[None]
        residues = (position >= first) & (position < lengths - 1)
        expected = torch.tensor([len(s) for s in sequences], device=DEVICE)
        if not torch.equal(residues.sum(1), expected):
            raise ValueError(f"token/residue count mismatch: {residues.sum(1).tolist()} vs {expected.tolist()}")
        weight = residues.unsqueeze(-1).to(torch.float32) / expected[:, None, None]
        return [(states.float() * weight).sum(1).cpu().numpy() for states in (last, self.captured)]


def embed(embedder, sequences, token_budget):
    order = sorted(range(len(sequences)), key=lambda i: -len(sequences[i]))
    outs = [None, None]
    sync()
    start = time.perf_counter()
    i = 0
    while i < len(order):
        size = max(1, token_budget // (-(-(len(sequences[order[i]]) + 2) // PAD) * PAD))
        chunk = order[i:i + size]
        for slot, values in enumerate(embedder.batch([sequences[j] for j in chunk])):
            if outs[slot] is None:
                outs[slot] = np.zeros((len(sequences), values.shape[1]), dtype=np.float32)
            outs[slot][chunk] = values
        i += size
    sync()
    return outs, time.perf_counter() - start


def knn(train, queries, chunk=512):
    """Exact cosine top-k on the CPU in query chunks; returns similarities, indices and seconds.
    Under a second for these sizes, and it keeps the GPU heap of the embedding pass from growing."""
    t = torch.nn.functional.normalize(torch.from_numpy(np.asarray(train, dtype=np.float32)), dim=1)
    q = torch.nn.functional.normalize(torch.from_numpy(np.asarray(queries, dtype=np.float32)), dim=1)
    start = time.perf_counter()
    parts = [(part @ t.T).topk(min(TOP, t.shape[0]), dim=1) for part in q.split(chunk)]
    sims, index = torch.cat([s for s, _ in parts]), torch.cat([i for _, i in parts])
    return sims.numpy(), index.numpy(), time.perf_counter() - start


def write_hits(path, query_ids, train_ids, sims, index):
    with path.open("w") as file:
        for qid, row_sims, row_index in zip(query_ids, sims, index):
            for sim, j in zip(row_sims, row_index):
                columns = [f"{sim:.6f}"] + [f"{np.exp((sim - 1) / t):.6e}" for t in TEMPERATURES]
                file.write(f"{qid}\t{train_ids[j]}\t" + "\t".join(columns) + "\n")


def cold(args):
    """One fresh process: load model and train index, embed one chain, search."""
    embedder = Embedder(args.model)
    index = np.load(args.index)
    knn(index, embedder.batch([args.sequence])[0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=MODELS)
    parser.add_argument("--token-budget", type=int, default=4096)
    parser.add_argument("--latency-samples", type=int, default=16)
    parser.add_argument("--datasets", nargs="+", default=["ec", "go"])
    parser.add_argument("--data-root", type=Path, default=Path("data/prepared"))
    parser.add_argument("--tag", default="", help="separate outputs for a different data release")
    parser.add_argument("--probe-only", action="store_true", help="embed a small and a long chain, then exit")
    parser.add_argument("--cold", action="store_true")
    parser.add_argument("--index")
    parser.add_argument("--sequence")
    parser.add_argument("--hits-only", action="store_true", help="neighbour files from stored vectors only")
    args = parser.parse_args()
    if args.cold:
        return cold(args)
    if args.probe_only:
        sample = list(fasta(args.data_root / args.datasets[0] / "train.fasta"))
        selected = [sample[0][1], max(sample, key=lambda row: len(row[1]))[1]]
        _, seconds = embed(Embedder(args.model), selected, args.token_budget)
        print(json.dumps({"model": args.model, "lengths": list(map(len, selected)), "seconds": seconds}))
        return
    tag = f"-{args.tag}" if args.tag else ""
    raw, logs, store = Path("results/raw"), Path("results/logs"), Path("data/plm") / args.model / args.tag if args.tag else Path("data/plm") / args.model
    for path in (raw, logs, store):
        path.mkdir(parents=True, exist_ok=True)
    load_start = time.perf_counter()
    embedder = None if args.hits_only else Embedder(args.model)
    load_seconds = time.perf_counter() - load_start
    environment = {"torch": torch.__version__, "device": DEVICE, "python": sys.version.split()[0],
                   "machine": subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                                             text=True).stdout.strip() if DEVICE == "mps" else torch.cuda.get_device_name(0)}
    for package in ("transformers", "esm"):
        try:
            environment[package] = __import__(package).__version__
        except (ImportError, AttributeError):
            pass
    # Embed every dataset first, then free the model so the search pass runs on a small heap.
    work = {}
    for dataset in args.datasets:
        data = args.data_root / dataset
        splits = {split: list(fasta(data / f"{split}.fasta")) for split in ("train", "valid", "test")}
        seconds = {}
        work[dataset] = splits, seconds
        for split, items in splits.items():
            done = store / f"{dataset}-{split}.json"
            if done.exists():
                seconds[split] = json.loads(done.read_text())["seconds"]
                for layer in ("last", "mid"):
                    if not (store / f"{dataset}-{split}-{layer}.npy").is_file():
                        raise FileNotFoundError(store / f"{dataset}-{split}-{layer}.npy")
                continue
            (last, mid), seconds[split] = embed(embedder, [s for _, s in items], args.token_budget)
            for layer, values in (("last", last), ("mid", mid)):
                np.save(store / f"{dataset}-{split}-{layer}.npy", values.astype(np.float16))
            (store / f"{dataset}-{split}-ids.txt").write_text("\n".join(n for n, _ in items) + "\n")
            done.write_text(json.dumps({"chains": len(items), "seconds": seconds[split]}))
            print(f"{args.model} {dataset} {split}: {len(items)} chains in {seconds[split]:.1f}s", flush=True)
            del last, mid
            gc.collect()
    embedder = None
    gc.collect()
    if DEVICE == "mps":
        torch.mps.empty_cache()
    for dataset, (splits, seconds) in work.items():
        vectors = {split: {layer: np.load(store / f"{dataset}-{split}-{layer}.npy").astype(np.float32)
                           for layer in ("last", "mid")} for split in ("train", "valid", "test")}
        train_ids = [n for n, _ in splits["train"]]
        summary = {"method": args.model, "dataset": dataset, "model_source": MODELS[args.model][0],
                   "mid_layer": MODELS[args.model][1], "model_load_seconds": load_seconds,
                   "environment": environment, "token_budget": args.token_budget,
                   "build_seconds": seconds["train"],
                   "index_bytes": vectors["train"]["last"].astype(np.float16).nbytes}
        names = []
        for layer in ("last", "mid"):
            mean, spread = vectors["train"][layer].mean(0), vectors["train"][layer].std(0) + 1e-6
            for scaled in (False, True):
                name = args.model + tag + ("-mid" if layer == "mid" else "") + ("-z" if scaled else "")
                names.append(name)
                train = (vectors["train"][layer] - mean) / spread if scaled else vectors["train"][layer]
                for split in ("valid", "test"):
                    queries = (vectors[split][layer] - mean) / spread if scaled else vectors[split][layer]
                    sims, index, search = knn(train, queries)
                    write_hits(raw / f"{name}-{dataset}-{split}-hits.tsv", [n for n, _ in splits[split]],
                               train_ids, sims, index)
                    if not scaled:
                        summary[f"{layer}_{split}_search_seconds"] = search
        if args.hits_only:
            continue
        test_n = len(splits["test"])
        summary.update({"valid_batch_seconds": seconds["valid"] + summary["last_valid_search_seconds"],
                        "test_batch_seconds": seconds["test"] + summary["last_test_search_seconds"],
                        "test_query_count": test_n})
        summary["test_throughput_ms"] = 1000 * summary["test_batch_seconds"] / test_n
        # Cold start: a new process loads cached weights and the train index, embeds one chain, searches.
        rng = np.random.default_rng(0)
        picks = rng.choice(test_n, size=min(args.latency_samples, test_n), replace=False)
        latencies = []
        for j in picks:
            begin = time.perf_counter()
            subprocess.run([sys.executable, __file__, args.model, "--cold", "--index",
                            str(store / f"{dataset}-train-last.npy"), "--sequence", splits["test"][j][1]],
                           check=True, env={**os.environ, "HF_HUB_OFFLINE": "1"})
            latencies.append(1000 * (time.perf_counter() - begin))
        if latencies:
            summary.update({"latency_sample_n": len(latencies), "query_mean_ms": statistics.mean(latencies),
                            "query_median_ms": statistics.median(latencies)})
        for name in names:
            (logs / f"{name}-{dataset}-summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary), flush=True)
        del vectors
        gc.collect()


if __name__ == "__main__":
    main()
