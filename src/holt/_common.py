"""Small data and risk helpers used by the frozen HoLT protocol."""

import json
from pathlib import Path

from scipy.stats import beta


def records(path: Path):
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sequences(path: Path):
    result = {}
    name, parts = None, []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line.startswith(">"):
                if name is not None:
                    if name in result:
                        raise ValueError(f"duplicate FASTA ID {name}: {path}")
                    result[name] = "".join(parts)
                name, parts = line[1:].split()[0], []
            elif line:
                parts.append(line)
    if name is not None:
        if name in result:
            raise ValueError(f"duplicate FASTA ID {name}: {path}")
        result[name] = "".join(parts)
    return result


def f1(left, right):
    a, b = set(left), set(right)
    return 2 * len(a & b) / (len(a) + len(b)) if a and b else 0.0


def ec4(labels):
    return {x for x in labels if len(x.split(".")) == 4 and all(p and p != "-" for p in x.split("."))}


def cp_upper(k, n, delta):
    return float(beta.ppf(1 - delta, k + 1, n - k)) if n and k < n else 1.0
