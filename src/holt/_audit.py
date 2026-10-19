"""Train a HoLT certificate and apply its frozen rule to held-out chains."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from . import _certify as cert
from ._common import ec4, f1, records, sequences
from ._engine import Pool


ROUTES = ("mmseqs", "foldseek", "esm2-650m", "esmc-600m")
TASKS = {"ec": ("ec",), "go": ("bp", "mf", "cc")}


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest(path: Path, no_embeddings=False):
    raw = json.loads(path.read_text())
    if raw.get("task_family") not in TASKS:
        raise ValueError("manifest task_family must be ec or go")

    def resolved(name, tool=None):
        candidate = (path.parent / name).resolve()
        if not candidate.is_file():
            hint = ""
            if tool == "mmseqs":
                hint = ("; MMseqs2 generates this evidence (macOS: brew install mmseqs2; "
                        "https://github.com/soedinglab/MMseqs2#installation)")
            elif tool == "foldseek":
                hint = ("; Foldseek generates this evidence (macOS universal binary: "
                        "https://mmseqs.com/foldseek/foldseek-osx-universal.tar.gz; "
                        "https://github.com/steineggerlab/foldseek#installation)")
            raise FileNotFoundError(f"required audit evidence absent: {candidate}{hint}")
        return candidate

    config = {"task_family": raw["task_family"], "no_embeddings": no_embeddings}
    for key in ("train_jsonl", "train_fasta", "test_jsonl", "test_fasta", "train_cluster"):
        config[key] = resolved(raw[key], "mmseqs" if key == "train_cluster" else None)
    config["train_hits"] = {route: resolved(raw["train_hits"][route], route)
                            for route in ("mmseqs", "foldseek")}
    config["test_hits"] = {route: resolved(raw["test_hits"][route], route)
                           for route in ("mmseqs", "foldseek")}
    if not no_embeddings:
        config["test_hits"].update({route: resolved(raw["test_hits"][route]) for route in ROUTES[2:]})
        config["train_embeddings"] = {}
        for route in ROUTES[2:]:
            source = raw["train_embeddings"][route]
            config["train_embeddings"][route] = {
                "ids": resolved(source["ids"]), "vectors": resolved(source["vectors"]),
                "cache": resolved(source["cache"]) if source.get("cache") else None}
    return config


def validate_inputs(config):
    train = records(config["train_jsonl"])
    train_ids = [row["id"] for row in train]
    if len(train_ids) != len(set(train_ids)):
        raise ValueError("duplicate training chain ID")
    for row in train:
        if not set(TASKS[config["task_family"]]) <= set(row["labels"]):
            raise ValueError(f"training labels missing a task for {row['id']}")
    train_seqs = sequences(config["train_fasta"])
    if set(train_ids) != set(train_seqs):
        raise ValueError("training FASTA and JSONL IDs disagree")
    test_seqs = sequences(config["test_fasta"])
    if set(test_seqs) & set(train_ids):
        raise ValueError("train and test chain IDs overlap")
    for group, query, target in (("train_hits", set(train_ids), set(train_ids)),
                                 ("test_hits", set(test_seqs), set(train_ids))):
        for route, path in config[group].items():
            minimum = 7 if route == "mmseqs" else 3
            with path.open() as handle:
                for line_no, line in enumerate(handle, 1):
                    fields = line.rstrip("\n").split("\t")
                    if len(fields) < minimum or fields[0] not in query or fields[1] not in target:
                        raise ValueError(f"{path}:{line_no}: malformed or out-of-split hit")
    return train, train_seqs, test_seqs


def hit_panel(path, route, train_labels, test_ids, tasks):
    """Match the frozen route ordering: rank then target ID, both descending."""
    scores = defaultdict(float)
    top = {task: defaultdict(list) for task in tasks}
    with path.open() as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            query, target = fields[:2]
            if query not in test_ids or target not in train_labels or query == target:
                continue
            value = float(fields[6] if route == "mmseqs" else fields[2])
            scores[query] = max(scores[query], value)
            for task in tasks:
                if not train_labels[target].get(task):
                    continue
                hits = top[task][query]
                hits.append((value, target))
                hits.sort(reverse=True)
                del hits[5:]
    return scores, top


def vote(hits, labels):
    counts = Counter(term for _, name in hits for term in set(labels[name]))
    return {term for term, count in counts.items() if count > len(hits) / 2}


def panel_features(panels, labels, task):
    terms_of = (lambda name: ec4(labels[name])) if task == "ec" else (lambda name: set(labels[name]))
    sizes, shares, tops = [], [], Counter()
    for hits in panels:
        if not hits:
            sizes.append(0)
            continue
        sizes.append(len(vote(hits, labels)))
        terms = Counter(term for _, name in hits for term in terms_of(name))
        shares.append(max(terms.values(), default=0) / len(hits))
        tops.update(terms_of(hits[0][1]))
    return float(np.mean(sizes)), max(shares, default=0.), max(tops.values(), default=0)


def heldout_features(config, train, train_seqs, test_seqs):
    """Read only held-out IDs and sequences; test labels are opened after calibration."""
    tasks = TASKS[config["task_family"]]
    train_labels = {row["id"]: row["labels"] for row in train}
    test_ids = set(test_seqs)
    panels, scores = {}, {}
    for route, path in config["test_hits"].items():
        scores[route], panels[route] = hit_panel(path, route, train_labels, test_ids, tasks)
    if config["no_embeddings"]:
        for route in ROUTES[2:]:
            scores[route] = defaultdict(float)
            panels[route] = {task: defaultdict(list) for task in tasks}
    identity = defaultdict(float)
    with config["test_hits"]["mmseqs"].open() as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            query, target = fields[:2]
            if query in test_ids and target in train_labels and query != target:
                identity[query] = max(identity[query], float(fields[2]))
    siblings = defaultdict(list)
    for name, labels in train_labels.items():
        siblings[name.split("-", 1)[0]].append(labels)
    train_sequence_set = set(train_seqs.values())
    by_task = {task: {name: label[task] for name, label in train_labels.items()} for task in tasks}
    result = {}
    for query, sequence in test_seqs.items():
        pdb = query.split("-", 1)[0]
        row = {"id": query, "pdb": pdb, "length": len(sequence),
               "identity": identity[query], "bits": scores["foldseek"][query],
               "esm2": scores["esm2-650m"][query], "esmc": scores["esmc-600m"][query],
               "sibling": pdb in siblings, "duplicate": sequence in train_sequence_set}
        for task in tasks:
            values = panel_features([panels[route][task][query] for route in ROUTES], by_task[task], task)
            row[f"vote_size_{task}"], row[f"agree_{task}"], row[f"consensus_{task}"] = values
        result[query] = row
    return result, panels, siblings, train_labels


def test_event(query, task, truth, panels, siblings, train_labels, by_task):
    attacks = []
    for route in ("mmseqs", "foldseek"):
        hits = panels[route][task][query]
        if hits:
            attacks.extend((train_labels[hits[0][1]][task], vote(hits, by_task)))
    attacks.extend(labels[task] for labels in siblings.get(query.split("-", 1)[0], ()) if labels[task])
    if task == "ec":
        target = ec4(truth)
        return bool(target) and any(ec4(pred) & target for pred in attacks)
    return bool(truth) and any(f1(pred, truth) >= .5 for pred in attacks)


def audit(config, alpha=.1, delta=.1, protocols_k=1):
    if not 0 < alpha <= .5:
        raise ValueError("alpha must satisfy 0 < alpha <= 1/2 (Lemma 3)")
    if not 0 < delta <= .25:
        raise ValueError("delta must satisfy 0 < delta <= 1/4 (Lemma 3)")
    if not isinstance(protocols_k, int) or protocols_k < 1:
        raise ValueError("protocols_k must be a positive integer")
    train, train_seqs, test_seqs = validate_inputs(config)
    # The entire certificate, including score and path, is frozen before test labels are read.
    pool = Pool(config["task_family"], config)
    cert.ALPHA, cert.DELTA = alpha, delta
    per_outcome = delta / (protocols_k * len(cert.FAMILY))
    cert.PER_TEST = {"pooled": per_outcome, "bonferroni": per_outcome / len(cert.Q),
                     "mondrian": per_outcome / cert.STRATA, "cluster": per_outcome}
    naive = {}
    representatives = cert.Representatives(pool)
    training = {}
    for tier in cert.TIERS:
        design, calibration, _, task_results = cert.run(pool, representatives, tier, naive,
                                                         ("mmseqs", "foldseek"))
        roles = {}
        for task in pool.tasks:
            chosen = representatives.representatives[tier][task]
            selected = set(map(int, chosen))
            r1 = len(set(pool.family[chosen])) == len(chosen)
            r2 = all(not (representatives.sources[tier][task][int(i)] & selected) for i in chosen)
            r3 = not any(row["index"] in selected for row in design[task])
            if not (r1 and r2 and r3):
                raise AssertionError(f"representative independence failed: {tier} {task}")
            roles[task] = {"ids": [pool.ids[int(i)] for i in chosen], "one_per_family": r1,
                           "no_representative_copy_source": r2,
                           "design_excludes_representatives": r3}
        training[tier] = {"tasks": task_results,
                          "representatives": representatives.stats[tier],
                          "role_checks": roles,
                          "calibration_n": {task: task_results[f"{task}_{'exact' if task == 'ec' else 'raw'}"]["calibration_n"]
                                            for task in pool.tasks}}
    rows, panels, siblings, train_labels = heldout_features(config, train, train_seqs, test_seqs)
    test_rows = records(config["test_jsonl"])
    test_ids = [row["id"] for row in test_rows]
    if len(test_ids) != len(set(test_ids)) or set(test_ids) != set(test_seqs):
        raise ValueError("test FASTA and JSONL IDs disagree or repeat")
    for row in test_rows:
        if not set(TASKS[config["task_family"]]) <= set(row.get("labels", {})):
            raise ValueError(f"test labels missing a task for {row['id']}; required for eligible-set evaluation")
    test_labels = {row["id"]: row["labels"] for row in test_rows}
    by_task = {task: {name: labels[task] for name, labels in train_labels.items()} for task in pool.tasks}
    hard = []
    summary = {}
    for tier, detail in training.items():
        summary[tier] = {}
        for task in pool.tasks:
            event = "exact" if task == "ec" else "raw"
            fitted = detail["tasks"][f"{task}_{event}"]
            selection = fitted.get("learned")
            path = selection["pooled"] if selection else None
            index = cert.fixed_sequence(path["trace"], alpha, path["level"]) if path else None
            point = path["trace"][index] if index is not None else None
            model = selection["model"] if selection else None
            eligible, chosen, leaks = 0, [], 0
            for query in test_seqs:
                truth = test_labels[query].get(task)
                if not truth or (task == "ec" and not ec4(truth)):
                    continue
                eligible += 1
                row = rows[query]
                if point is None or row["identity"] >= .3 or row["sibling"] or row["duplicate"]:
                    continue
                score = cert.copy_score(row, task, model)
                if score >= point["cut"][0]:
                    continue
                leak = test_event(query, task, truth, panels, siblings, train_labels, by_task[task])
                leaks += int(leak)
                chosen.append(query)
                hard.append({"tier": tier, "task": task, "id": query, "score": score,
                             "test_copy_event": int(leak)})
            summary[tier][task] = {"eligible_test": eligible, "certified": len(chosen),
                                   "unavailable": fitted.get("unavailable"),
                                   "test_copy_events": leaks, "realized_rate": leaks / len(chosen) if chosen else None,
                                   "calibration_upper_bound": point["ucb"] if point else None,
                                   "calibration_n": point["n"] if point else 0,
                                   "calibration_copy_events": point["leaks"] if point else 0,
                                   "selected_quantile": point["q"] if point else None,
                                   "hard_ids": chosen}
    return {"protocol": cert.PROTOCOL, "alpha": alpha, "delta": delta,
            "protocols_k": protocols_k, "outcomes": list(cert.FAMILY),
            "per_candidate_level": per_outcome,
            "guard": "maximum MMseqs2 identity below 0.3 at any coverage; design, calibration, deployment",
            "protocol_variant": "guarded_two_route" if config["no_embeddings"] else "guarded_four_route",
            "paper_protocol_reproduced": not config["no_embeddings"] and protocols_k == 1,
            "embedding_features": "disabled" if config["no_embeddings"] else "ESM-2 650M and ESM C 600M",
            "adversary": ["MMseqs2", "Foldseek", "same-PDB siblings"],
            "claim": "calibration representative risk bound; transfer to test requires the paper's transfer assumption",
            "training": training, "summary": summary}, hard
