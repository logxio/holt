"""MotiL's Fmax metric plus micro average precision for the new baselines."""
import numpy as np


# Copied from MotiL_macromolecule/utils.py without changing its convention:
# precision excludes queries with no predictions; recall includes every query.
def fmax(probs, labels):
    thresholds = np.arange(0, 1, 0.01)
    f_max = 0.0

    for threshold in thresholds:
        precision = 0.0
        recall = 0.0
        precision_cnt = 0
        recall_cnt = 0
        for idx in range(probs.shape[0]):
            prob = probs[idx]
            label = labels[idx]
            pred = (prob > threshold).astype(np.int32)
            correct_sum = np.sum(label*pred)
            pred_sum = np.sum(pred)
            label_sum = np.sum(label)
            if pred_sum > 0:
                precision += correct_sum/pred_sum
                precision_cnt += 1
            if label_sum > 0:
                recall += correct_sum/label_sum
            recall_cnt += 1
        if recall_cnt > 0:
            recall = recall / recall_cnt
        else:
            recall = 0
        if precision_cnt > 0:
            precision = precision / precision_cnt
        else:
            precision = 0
        f = (2.*precision*recall)/max(precision+recall, 1e-8)
        f_max = max(f, f_max)

    return f_max


def fmax_fast(probs, labels):
    """Vectorized equivalent for validation-grid selection."""
    label_counts = labels.sum(axis=1)
    best = 0.0
    for threshold in np.arange(0, 1, 0.01):
        pred = probs > threshold
        pred_counts = pred.sum(axis=1)
        correct = np.logical_and(pred, labels).sum(axis=1)
        active = pred_counts > 0
        precision = (correct[active] / pred_counts[active]).mean() if active.any() else 0.0
        recall = np.divide(correct, label_counts, out=np.zeros(len(correct)), where=label_counts > 0).mean()
        best = max(best, 2 * precision * recall / max(precision + recall, 1e-8))
    return float(best)


def micro_aupr(probs, labels):
    """Average precision over all query-label pairs, grouping equal scores."""
    score = np.asarray(probs, dtype=np.float32).ravel()
    truth = np.asarray(labels, dtype=np.uint8).ravel()
    positives = int(truth.sum())
    if positives == 0:
        return 0.0
    order = np.argsort(-score, kind="mergesort")
    sorted_score = score[order]
    sorted_truth = truth[order]
    ends = np.r_[np.flatnonzero(sorted_score[:-1] != sorted_score[1:]), len(score) - 1]
    tp = np.cumsum(sorted_truth, dtype=np.int64)[ends]
    previous = np.r_[0, tp[:-1]]
    precision = tp / (ends + 1)
    return float(np.sum(precision * (tp - previous)) / positives)
