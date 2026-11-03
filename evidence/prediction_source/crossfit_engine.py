"""Features and copying events of training chains against any reference library.

One engine serves every HoLT calibration design. All hit tables are read once
into arrays sorted by query, then rank: MMseqs2 and Foldseek from the
training self-searches (``crossfit_retrieve.py``, up to 300 hits) and, for the
two embedding routes, the 500 nearest training chains by cosine of the last
layer mean vectors (computed here, cached). A library is a boolean mask over
training chains; a query never sees chains that share its exclusion key
(its PDB entry, or its family). Families are connected components of
same-PDB-entry and same-MMseqs2-30%-cluster links.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from certify_four import ec4
from certify_split import f1, kind, records, sequences
from crossfit_retrieve import DATA, VECTORS


RAW = Path("results/raw")
CACHE = Path("data/tmp/crossfit")
TASKS = {"ec": ("ec",), "go": ("bp", "mf", "cc")}
ROUTES = ("mmseqs", "foldseek", "esm2-650m", "esmc-600m")
# ``profile`` adds HMMER phmmer (results/对手-261013.md: training self-search with each query's own 30%
# cluster removed, so it serves the mmseqs30 tier only) to the homology routes.
ADVERSARIES = {"homology": ("mmseqs", "foldseek"), "four": ROUTES, "profile": ("mmseqs", "foldseek", "profile")}
# Label-transfer rules per adversary on each attacking route's top annotated hits; the profile adversary
# adds bits5, union3 and union5 (results/对手-261013.md) to the top hit and the top-5 majority.
BASE_RULES = ("top1", "majority5")
RULES = {"profile": ("top1", "majority5", "bits5", "union3", "union5")}
COSINE_K = 500
TOP = 5


def transfer(labels, scores, rule):
    """One rule's predicted label set from ranked annotated hits (label sets with their scores)."""
    if not labels:
        return frozenset()
    if rule == "top1":
        return labels[0]
    k = 3 if rule == "union3" else 5
    picked, weights = labels[:k], list(scores[:k])
    if rule in ("union3", "union5"):
        return frozenset().union(*picked)
    if rule == "bits5":
        weights = [max(w, 0.) for w in weights]
        if not sum(weights):
            weights = [1.] * len(picked)
    else:
        weights = [1.] * len(picked)
    support = Counter()
    for w, terms in zip(weights, picked):
        for term in terms:
            support[term] += w
    return frozenset(term for term, value in support.items() if value > sum(weights) / 2)


def unit_hash(name, salt=""):
    return int.from_bytes(hashlib.sha256((salt + name).encode()).digest()[:4], "big") / 2**32


def deep_cosine(dataset, model, k=COSINE_K):
    """k nearest training chains by cosine, self and same PDB entry removed."""
    path = CACHE / f"{dataset}-{model}-top{k}.npz"
    if path.exists():
        return np.load(path)
    base = Path("data/plm") / model
    names = (base / f"{VECTORS[dataset]}-train-ids.txt").read_text().splitlines()
    matrix = np.asarray(np.load(base / f"{VECTORS[dataset]}-train-last.npy"), dtype=np.float32)
    matrix /= np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)
    pdb = np.array([name.split("-", 1)[0] for name in names])
    _, code = np.unique(pdb, return_inverse=True)
    q_out, t_out, s_out = [], [], []
    for offset in range(0, len(names), 512):
        block = matrix[offset:offset + 512] @ matrix.T
        rows = np.arange(offset, min(offset + 512, len(names)))
        block[code[rows][:, None] == code[None, :]] = -np.inf
        pick = np.argpartition(block, -k, axis=1)[:, -k:]
        score = np.take_along_axis(block, pick, axis=1)
        keep = np.isfinite(score)
        q_out.append(np.repeat(rows, k)[keep.ravel()])
        t_out.append(pick[keep])
        s_out.append(score[keep])
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(path, names=np.array(names), q=np.concatenate(q_out).astype(np.int32),
             t=np.concatenate(t_out).astype(np.int32), s=np.concatenate(s_out).astype(np.float32))
    return np.load(path)


class Route:
    """Hits of one route: query, target, rank score and feature value (MMseqs2: identity)."""

    def __init__(self, q, t, rank, value, name_rank):
        # Sorted by query, then rank and target name, both descending, as the list-based panels did.
        order = np.lexsort((-name_rank[t], -rank, q))
        self.q, self.t = q[order], t[order]
        self.rank, self.value = rank[order], value[order]


class Pool:
    """Training chains of one dataset with their labels, families and hit routes."""

    def __init__(self, dataset, extra=()):
        self.dataset = dataset
        self.rules = BASE_RULES
        self.tasks = TASKS[kind(dataset)]
        data = DATA[dataset]
        train = records(data / "train.jsonl")
        self.ids = [row["id"] for row in train]
        # Ties in rank break by target name, descending, as the list-based panels did.
        order = sorted(range(len(self.ids)), key=lambda i: self.ids[i])
        self.name_rank = np.empty(len(self.ids), dtype=np.int64)
        self.name_rank[order] = np.arange(len(self.ids))
        self.index = {name: i for i, name in enumerate(self.ids)}
        n = len(self.ids)
        lengths = sequences(data / "train.fasta")
        self.length = np.array([len(lengths[name]) for name in self.ids])
        self.pdb_name = [name.split("-", 1)[0] for name in self.ids]
        _, self.pdb = np.unique(self.pdb_name, return_inverse=True)
        cluster = {}
        with (RAW / f"crossfit-{dataset}-mmseqs30_cluster.tsv").open() as handle:
            for line in handle:
                rep, member = line.rstrip("\n").split("\t")[:2]
                cluster[member] = rep
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
        for route in extra:
            self.routes[route] = self._extra(route)

    def _extra(self, route):
        """A further route's training self-search: query, target, bits (results/raw/<route>-<dataset>-train-crossfit-hits.tsv)."""
        q, t, s = [], [], []
        with (RAW / f"{route}-{self.dataset}-train-crossfit-hits.tsv").open() as handle:
            for line in handle:
                fields = line.split("\t")
                a, b = self.index.get(fields[0]), self.index.get(fields[1])
                if a is None or b is None or self.pdb[a] == self.pdb[b]:
                    continue
                q.append(a)
                t.append(b)
                s.append(float(fields[2]))
        score = np.array(s, dtype=np.float64)
        return Route(np.array(q, dtype=np.int32), np.array(t, dtype=np.int32), score, score, self.name_rank)

    def _routes(self):
        cache = CACHE / f"{self.dataset}-routes.npz"
        out = {}
        if cache.exists():
            z = np.load(cache)
            for route in ("mmseqs", "foldseek"):
                out[route] = Route(z[f"{route}_q"], z[f"{route}_t"], z[f"{route}_rank"], z[f"{route}_value"],
                                   self.name_rank)
        else:
            arrays = {}
            for route in ("mmseqs", "foldseek"):
                q, t, rank, value = [], [], [], []
                with (RAW / f"crossfit-{self.dataset}-full-{route}-hits.tsv").open() as handle:
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
                arrays[route] = [np.array(q, dtype=np.int32), np.array(t, dtype=np.int32),
                                 np.array(rank, dtype=np.float64), np.array(value, dtype=np.float64)]
                out[route] = Route(*arrays[route], self.name_rank)
            CACHE.mkdir(parents=True, exist_ok=True)
            np.savez(cache, **{f"{route}_{key}": arrays[route][i] for route in arrays
                               for i, key in enumerate(("q", "t", "rank", "value"))})
        for model in ("esm2-650m", "esmc-600m"):
            z = deep_cosine(self.dataset, model)
            names = list(z["names"])
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
        best, top, self.top_scores = {}, {}, {}
        for name, route in self.routes.items():
            take = (pos[route.q] >= 0) & library[route.t] & (key[route.t] != key[route.q])
            idx = np.flatnonzero(take)
            qs = route.q[idx]
            score = np.zeros(len(queries))
            if len(idx):
                starts = np.flatnonzero(np.r_[True, qs[1:] != qs[:-1]])
                score[pos[qs[starts]]] = np.maximum.reduceat(route.value[idx], starts)
            best[name] = score
            top[name], self.top_scores[name] = {}, {}
            for task in (self.tasks if tasks is None else tasks):
                sub = idx[self.annotated[task][route.t[idx]]]
                out = np.full((len(queries), TOP), -1, dtype=np.int64)
                scores = np.zeros((len(queries), TOP))
                if len(sub):
                    qq = route.q[sub]
                    first = np.r_[True, qq[1:] != qq[:-1]]
                    group_start = np.maximum.accumulate(np.where(first, np.arange(len(sub)), 0))
                    rank = np.arange(len(sub)) - group_start
                    keep = rank < TOP
                    out[pos[qq[keep]], rank[keep]] = route.t[sub[keep]]
                    scores[pos[qq[keep]], rank[keep]] = route.rank[sub[keep]]
                top[name][task] = out
                self.top_scores[name][task] = scores
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
                if self.rules == BASE_RULES:
                    attacks = [p for route in attackers if route in predictions for p in predictions[route]]
                else:
                    attacks = []
                    for route in attackers:
                        hits = [int(t) for t in top[route][task][i] if t >= 0]
                        ranked = [labels[t] for t in hits]
                        attacks += [transfer(ranked, self.top_scores[route][task][i][:len(hits)], rule)
                                    for rule in self.rules]
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


def save_json(path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
