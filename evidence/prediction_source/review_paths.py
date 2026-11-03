"""Resolve either the current repaired EC split or the original DeepFRI split."""
from pathlib import Path


def dataset_dir(root: Path, dataset: str) -> Path:
    if dataset == "go":
        path = root / "go"
    elif dataset == "ec":
        path = root / ("ec-seqfix" if (root / "ec-seqfix" / "test.jsonl").exists() else "ec")
    else:
        raise ValueError(dataset)
    if not (path / "test.jsonl").exists():
        raise FileNotFoundError(f"prepared split missing: {path / 'test.jsonl'}")
    return path


def audit_stem(data_root: Path, dataset: str) -> str:
    release = "deepfri-original" if data_root.name == "deepfri" else "deepfri"
    return f"audit-{release}-{dataset}"


def review_stem(data_root: Path) -> str:
    """Keep original SEQRES predictions separate from the legacy prepared split."""
    return "review-original" if data_root.name == "deepfri" else "review"
