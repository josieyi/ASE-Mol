#!/usr/bin/env python

import numpy as np

class UndefinedROCError(ValueError):
    def __init__(self, counts):
        self.counts = counts
        detail = "; ".join("task=%s negative=%d positive=%d missing=%d" %
                           (t, c["negative"], c["positive"], c["missing"])
                           for t, c in counts["per_task"].items())
        super().__init__("ROC-AUC undefined: no task has both positive and negative labels; " + detail)

def label_counts(labels):
    labels = np.asarray(labels)
    if labels.ndim != 2 or labels.shape[1] == 0:
        raise ValueError("Labels must be a 2D [molecules, tasks] array")
    if not np.isin(labels, [-1, 0, 1]).all():
        raise ValueError("Labels must be -1/0/1; zero means missing")
    rows, eligible = {}, []
    for t in range(labels.shape[1]):
        values = labels[:, t]
        neg, pos, missing = (int((values == v).sum()) for v in (-1, 1, 0))
        reason = None if neg and pos else ("all_missing" if not neg and not pos else
                                          "only_positive" if pos else "only_negative")
        rows[str(t)] = {"negative": neg, "positive": pos, "missing": missing,
                        "roc_auc_defined": bool(neg and pos), "excluded_reason": reason}
        if neg and pos:
            eligible.append(t)
    return {"n": len(labels), "num_tasks": labels.shape[1], "per_task": rows,
            "eligible_tasks": eligible, "roc_auc_defined": bool(eligible)}

def roc_details(y_true, y_scores):
    y_true, y_scores = np.asarray(y_true), np.asarray(y_scores)
    if y_true.shape != y_scores.shape or y_true.ndim != 2 or not np.isfinite(y_scores).all():
        raise ValueError("ROC inputs have incompatible dimensions or non-finite scores")
    counts = label_counts(y_true)
    if not counts["roc_auc_defined"]:
        raise UndefinedROCError(counts)
    from sklearn.metrics import roc_auc_score
    values, excluded = {}, []
    for t in range(y_true.shape[1]):
        if t in counts["eligible_tasks"]:
            keep = y_true[:, t] != 0
            values[str(t)] = float(roc_auc_score((y_true[keep, t] + 1) / 2, y_scores[keep, t]) * 100)
        else:
            excluded.append(t)
    return {"roc_auc": float(np.mean(list(values.values()))), "per_task": values,
            "excluded_tasks": excluded, "evaluated_tasks": len(values),
            "status": "ok", "label_counts": counts}

def audit_dataset(args, dataset, subsets):
    """Report observed labels and reject splits with undefined validation/test AUC."""
    parts = {}
    for name, subset in zip(("train", "valid", "test"), subsets):
        labels = [d.y.detach().cpu().numpy().reshape(-1) for d in subset]
        if any(len(row) != args.num_tasks for row in labels):
            raise ValueError("Cached task count differs from configuration")
        parts[name] = np.asarray(labels).reshape(-1, args.num_tasks)
    counts = {name: label_counts(labels) for name, labels in parts.items()}
    errors = []
    for name, count in counts.items():
        if not count["n"]:
            errors.append(name + ": empty_split")
        elif name == "train" and not any(
                item["negative"] + item["positive"] for item in count["per_task"].values()):
            errors.append("train: no_observed_labels")
        elif name in ("valid", "test") and not count["roc_auc_defined"]:
            errors.append(name + ": no_task_with_both_classes")
    return {"status": "failed" if errors else "ok", "errors": errors, "splits": counts}


def require_usable_split(report):
    if report["status"] != "ok":
        details = []
        for name in ("valid", "test"):
            count = report["splits"][name]
            if not count["roc_auc_defined"]:
                for task, values in count["per_task"].items():
                    details.append("%s task=%s negative=%d positive=%d missing=%d" % (
                        name, task, values["negative"], values["positive"], values["missing"]))
        raise ValueError("Dataset split invalid before training: %s; %s" % (
            ", ".join(report["errors"]), "; ".join(details)))
