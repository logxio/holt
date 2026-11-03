"""Training features and copying events against an excluded reference library.

Families connect chains sharing a PDB entry or a 30% MMseqs2 cluster. The
library mask removes a query's own family or PDB entry during calibration.
"""
from __future__ import annotations

import hashlib
import math
from collections import Counter

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from ._common import ec4, f1, records, sequences


TASKS = {"ec": ("ec",), "go": ("bp", "mf", "cc")}
ROUTES = ("mmseqs", "foldseek", "esm2-650m", "esmc-600m")
COSINE_K = 500
TOP = 5


def unit_hash(name, salt=""):
    return int.from_bytes(hashlib.sha256((salt + name).encode()).digest()[:4], "big") / 2**32


def deep_cosine(ids_path, matrix_path, cache_path, k=COSINE_K):
    """k nearest training chains by cosine, self and same PDB entry removed."""
    names = ids_path.read_text().splitlines()
    shape = np.load(matrix_path, mmap_mode="r").shape
    if len(shape) != 2 or shape[0] != len(names):
        raise ValueError(f"embedding matrix rows disagree with IDs: {matrix_path}")
    if cache_path is not None and cache_path.exists():
        return np.load(cache_path)
    matrix = np.asarray(np.load(matrix_path, mmap_mode="r"), dtype=np.float32).copy()
    matrix /= np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)
    pdb = np.array([name.split("-", 1)[0] for name in names])
    _, code = np.unique(pdb, return_inverse=True)
    q_out, t_out, s_out = [], [], []
    k = min(k, max(len(names) - 1, 1))
    for offset in range(0, len(names), 128):
        block = matrix[offset:offset + 128] @ matrix.T
        rows = np.arange(offset, min(offset + 128, len(names)))
        block[code[rows][:, None] == code[None, :]] = -np.inf
        pick = np.argpartition(block, -k, axis=1)[:, -k:]
        score = np.take_along_axis(block, pick, axis=1)
        keep = np.isfinite(score)
        q_out.append(np.repeat(rows, k)[keep.ravel()])
        t_out.append(pick[keep])
        s_out.append(score[keep])
    payload = {"names": np.array(names), "q": np.concatenate(q_out).astype(np.int32),
               "t": np.concatenate(t_out).astype(np.int32), "s": np.concatenate(s_out).astype(np.float32)}
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache_path, **payload)
        return np.load(cache_path)
    return payload


class Route:
    """Hits of one route: query, target, rank score and feature value (MMseqs2: identity)."""

    def __init__(self, q, t, rank, value, name_rank):
        # Sorted by query, then rank and target name, both descending, as the list-based panels did.
        order = np.lexsort((-name_rank[t], -rank, q))
        self.q, self.t = q[order], t[order]
        self.rank, self.value = rank[order], value[order]


class Pool:
    """Training chains of one dataset with their labels, families and hit routes."""

    def __init__(self, dataset, config):
        self.dataset = dataset
        self.config = config
        self.tasks = TASKS[dataset]
        train = records(config["train_jsonl"])
        self.ids = [row["id"] for row in train]
        # Ties in rank break by target name, descending, as the list-based panels did.
        order = sorted(range(len(self.ids)), key=lambda i: self.ids[i])
        self.name_rank = np.empty(len(self.ids), dtype=np.int64)
        self.name_rank[order] = np.arange(len(self.ids))
        self.index = {name: i for i, name in enumerate(self.ids)}
        n = len(self.ids)
        lengths = sequences(config["train_fasta"])
        if set(lengths) != set(self.ids) or len(self.ids) != len(set(self.ids)):
            raise ValueError("training FASTA/JSONL IDs disagree or repeat")
        self.length = np.array([len(lengths[name]) for name in self.ids])
        self.pdb_name = [name.split("-", 1)[0] for name in self.ids]
        _, self.pdb = np.unique(self.pdb_name, return_inverse=True)
        cluster = {}
        with config["train_cluster"].open() as handle:
            for line in handle:
                rep, member = line.rstrip("\n").split("\t")[:2]
                if member in cluster:
                    raise ValueError(f"duplicate training cluster member: {member}")
                cluster[member] = rep
        if set(cluster) != set(self.ids) or not set(cluster.values()) <= set(self.ids):
            raise ValueError("training 30% cluster table does not cover training IDs")
        self.cluster_name = [cluster[name] for name in self.ids]
        _, self.cluster = np.unique(self.cluster_name, return_inverse=True)
        rows = np.concatenate([np.arange(n), np.arange(n)])
        first_of_pdb = np.full(self.pdb.max() + 1, -1)
        for i in range(n):
            if first_of_pdb[self.pdb[i]] < 0:
                first_of_pdb[self.pdb[i]] = i
        rep_of_cluster = np.array([self.index[cluster[name]] for name in self.ids])
        cols = np.concatenate([rep_of_cluster, first_of_pdb[self.pdb]])
        graph = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
        _, component = connected_components(graph, directed=False)
        # Family name: its alphabetically first chain, so the role hash does not depend on numbering.
        first = {}
        for i in order:
            first.setdefault(component[i], self.ids[i])
        self.family_name = [first[component[i]] for i in range(n)]
        _, self.family = np.unique(self.family_name, return_inverse=True)
        self.labels = {task: [frozenset(row["labels"][task]) for row in train] for task in self.tasks}
        self.terms = {task: ([frozenset(ec4(x)) for x in self.labels[task]] if task == "ec"
                             else self.labels[task]) for task in self.tasks}
        self.annotated = {task: np.array([bool(x) for x in self.labels[task]]) for task in self.tasks}
        # A chain is scored for a label set when it has labels; EC exact matching needs a level-4 number.
        self.eligible = {task: np.array([bool(x) for x in self.terms[task]]) for task in self.tasks}
        self.routes = self._routes()

    def _routes(self):
        out = {}
        for route in ("mmseqs", "foldseek"):
            q, t, rank, value = [], [], [], []
            with self.config["train_hits"][route].open() as handle:
                for line in handle:
                    fields = line.split("\t")
                    a, b = self.index.get(fields[0]), self.index.get(fields[1])
                    if a is None or b is None or self.pdb[a] == self.pdb[b]:
                        continue
                    q.append(a)
                    t.append(b)
                    if route == "mmseqs":
                        rank.append(float(fields[6]))
                        value.append(float(fields[2]))
                    else:
                        rank.append(float(fields[2]))
                        value.append(float(fields[2]))
            out[route] = Route(np.array(q, dtype=np.int32), np.array(t, dtype=np.int32),
                               np.array(rank, dtype=np.float64), np.array(value, dtype=np.float64),
                               self.name_rank)
        for model in ("esm2-650m", "esmc-600m"):
            if self.config.get("no_embeddings"):
                out[model] = Route(np.array([], dtype=np.int32), np.array([], dtype=np.int32),
                                   np.array([]), np.array([]), self.name_rank)
                continue
            source = self.config["train_embeddings"][model]
            z = deep_cosine(source["ids"], source["vectors"], source.get("cache"))
            names = list(z["names"])
            if source["ids"].read_text().splitlines() != names:
                raise ValueError(f"{model} neighbour cache and vector IDs disagree")
            if len(names) != len(self.ids) or set(names) != set(self.ids):
                raise ValueError(f"{model} training vector IDs disagree with train JSONL")
            if names != self.ids:
                remap = np.array([self.index[name] for name in names])
                q, t = remap[z["q"]], remap[z["t"]]
            else:
                q, t = z["q"], z["t"]
            score = z["s"].astype(np.float64)
            out[model] = Route(q, t, score, score, self.name_rank)
        return out

    def panels(self, queries, library, key, tasks=None):
        """For each query: the best score of each route, and its top five annotated hits per task.

        ``library`` masks the chains a query may retrieve; ``key`` (PDB entry or family code per chain)
        removes the query's own entry or family."""
        queries = np.asarray(queries)
        pos = np.full(len(self.ids), -1)
        pos[queries] = np.arange(len(queries))
        best, top = {}, {}
        for name, route in self.routes.items():
            take = (pos[route.q] >= 0) & library[route.t] & (key[route.t] != key[route.q])
            idx = np.flatnonzero(take)
            qs = route.q[idx]
            score = np.zeros(len(queries))
            if len(idx):
                starts = np.flatnonzero(np.r_[True, qs[1:] != qs[:-1]])
                score[pos[qs[starts]]] = np.maximum.reduceat(route.value[idx], starts)
            best[name] = score
            top[name] = {}
            for task in (self.tasks if tasks is None else tasks):
                sub = idx[self.annotated[task][route.t[idx]]]
                out = np.full((len(queries), TOP), -1, dtype=np.int64)
                if len(sub):
                    qq = route.q[sub]
                    first = np.r_[True, qq[1:] != qq[:-1]]
                    group_start = np.maximum.accumulate(np.where(first, np.arange(len(sub)), 0))
                    rank = np.arange(len(sub)) - group_start
                    keep = rank < TOP
                    out[pos[qq[keep]], rank[keep]] = route.t[sub[keep]]
                top[name][task] = out
        return best, top

    def rows(self, queries, library, key, naive=None, attackers=ROUTES, tasks=None):
        """Per query: label-free features and copying events against the library."""
        tasks = self.tasks if tasks is None else tasks
        best, top = self.panels(queries, library, key, tasks)
        out = []
        for i, q in enumerate(queries):
            row = {"index": int(q), "id": self.ids[q], "pdb": self.pdb_name[q], "family": self.family_name[q],
                   "cluster": self.cluster_name[q], "length": int(self.length[q]),
                   "identity": best["mmseqs"][i], "bits": best["foldseek"][i],
                   "esm2": best["esm2-650m"][i], "esmc": best["esmc-600m"][i]}
            for task in tasks:
                labels, terms = self.labels[task], self.terms[task]
                truth = labels[q]
                row[f"labels_{task}"] = len(terms[q])
                sizes, shares, tops = [], [], Counter()
                predictions = {}
                for route in ROUTES:
                    hits = [int(t) for t in top[route][task][i] if t >= 0]
                    if not hits:
                        sizes.append(0)
                        continue
                    count = Counter(term for t in hits for term in labels[t])
                    voted = {term for term, c in count.items() if c > len(hits) / 2}
                    sizes.append(len(voted))
                    held = Counter(term for t in hits for term in terms[t])
                    shares.append(max(held.values(), default=0) / len(hits))
                    tops.update(terms[hits[0]])
                    predictions[route] = (labels[hits[0]], voted)
                row[f"vote_size_{task}"] = float(np.mean(sizes))
                row[f"agree_{task}"] = max(shares, default=0.)
                row[f"consensus_{task}"] = max(tops.values(), default=0)
                attacks = [p for route in attackers if route in predictions for p in predictions[route]]
                if task == "ec":
                    target = terms[q]
                    row["leak_ec_exact"] = bool(target) and any(ec4(p) & target for p in attacks)
                else:
                    base = f1(naive[task], truth) if naive and truth else 0.
                    values = [f1(p, truth) for p in attacks]
                    row[f"naive_f1_{task}"] = base
                    row[f"leak_{task}_raw"] = bool(truth) and any(v >= .5 for v in values)
                    row[f"leak_{task}_relative"] = bool(truth) and any(v >= .5 and v > base for v in values)
            out.append(row)
        return out


def features(row, task):
    """The eight label-free quantities of the learned score."""
    return [row["identity"], math.log1p(row["bits"]), row["esm2"], row["esmc"],
            row[f"agree_{task}"], row[f"consensus_{task}"], row[f"vote_size_{task}"],
            math.log(max(row["length"], 1))]
