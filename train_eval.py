#!/usr/bin/env python

import time
import numpy as np
import torch
from model import masked_bce
from losses import similarity_scores
from util import batch_nodes, sync_device
from metrics import roc_details
from training_optim import gradient_step

def pre_train(args, model, loader, optimizer, cache, schedule=None):
    """Recognition-stage optimization of BCE plus motif ranking loss."""
    model.train()
    totals = {"loss": 0.0, "task_loss": 0.0, "margin_loss": 0.0}
    margin_counts = {"observed_pairs": 0, "active_pairs": 0, "zero_gradient_pairs": 0,
                     "sigmoid_saturated_pairs": 0, "positive_gap_pairs": 0,
                     "gap_sum": 0.0}
    count = 0
    for batch in loader:
        batch = batch.to(args.device)
        pos, neg = batch_nodes(batch, cache)
        whole, hp, hn, logits = model.pre_forward(batch, pos, neg)
        labels = batch.y.reshape(-1, args.num_tasks)
        task = masked_bce(logits, labels)
        margin = model.margin_loss(whole, hp, hn, labels)
        loss = task + args.pre_cl * margin
        gradient_step(args, loss, model, optimizer, schedule)

        with torch.no_grad():
            valid = labels != 0
            if valid.any():
                all_sp, all_sn = similarity_scores(whole, hp, hn, args.margin_temperature)
                sp, sn = all_sp[valid], all_sn[valid]
                gap = sp - sn
                nobs = int(valid.sum().item())
                active = int((gap < args.margin).sum().item())
                saturated = int(((sp < 1e-4) | (sp > 1-1e-4) |
                                 (sn < 1e-4) | (sn > 1-1e-4)).sum().item())
                margin_counts["observed_pairs"] += nobs
                margin_counts["active_pairs"] += active
                margin_counts["zero_gradient_pairs"] += nobs - active
                margin_counts["sigmoid_saturated_pairs"] += saturated
                margin_counts["positive_gap_pairs"] += int((gap > 0).sum().item())
                margin_counts["gap_sum"] += float(gap.sum().item())
        n = batch.num_graphs
        for key, value in (("loss", loss), ("task_loss", task), ("margin_loss", margin)):
            totals[key] += value.item() * n
        count += n
    if not count:
        raise ValueError("Training set is empty")
    result = {k: v / count for k, v in totals.items()}
    obs = margin_counts["observed_pairs"]
    result["margin_diagnostic"] = {
        "observed_pairs": obs,
        "active_fraction": margin_counts["active_pairs"] / obs if obs else None,
        "inactive_fraction": margin_counts["zero_gradient_pairs"] / obs if obs else None,
        "sigmoid_saturated_fraction": margin_counts["sigmoid_saturated_pairs"] / obs if obs else None,
        "positive_gap_fraction": margin_counts["positive_gap_pairs"] / obs if obs else None,
        "mean_similarity_gap": margin_counts["gap_sum"] / obs if obs else None,
    }
    return result

def train(args, model, loader, optimizer, cache, schedule=None):
    """Expert-stage optimization with importance regularization."""
    model.train()
    total, count = 0.0, 0
    entropy_sum = {"pos": 0.0, "neg": 0.0}
    entropy_count = 0
    sync_device(args.device)
    start = time.perf_counter()
    for batch in loader:
        batch = batch.to(args.device)
        pos, neg = batch_nodes(batch, cache)
        labels = batch.y.reshape(-1, args.num_tasks)
        logits = model(batch, pos, neg)
        loss = masked_bce(logits, labels)
        loss = loss + args.cl * model.compute_importance_and_loss()

        with torch.no_grad():
            import math
            for name, routes in (("pos", model.pos_route_score), ("neg", model.neg_route_score)):
                norm = math.log(max(2, args.num_experts))
                entropy_sum[name] += float((-(routes * routes.clamp_min(1e-12).log()).sum(-1) / norm).sum().item())
            entropy_count += batch.num_graphs * args.num_tasks
        gradient_step(args, loss, model, optimizer, schedule)
        total += loss.item() * batch.num_graphs
        count += batch.num_graphs
    sync_device(args.device)
    if not count:
        raise ValueError("Training set is empty")
    elapsed = time.perf_counter() - start
    return {"loss": total / count, "epoch_seconds": elapsed,
            "batch_seconds": elapsed / len(loader), "num_batches": len(loader), "num_graphs": count,
            "route_entropy": {k: v / entropy_count for k, v in entropy_sum.items()}
                             if entropy_count and args.num_experts > 1 else None}

def predict(args, model, loader, cache):
    model.eval()
    ids, scores = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(args.device)
            pos, neg = batch_nodes(batch, cache)
            scores.append(torch.sigmoid(model(batch, pos, neg)).cpu())
            ids.extend(batch.id.cpu().tolist())
    if not scores:
        raise ValueError("Prediction set is empty")
    return ids, torch.cat(scores).numpy()

def collect_predictions(args, model, loader, cache):

    ids, scores = predict(args, model, loader, cache)
    target_ids, labels = [], []
    for batch in loader:
        target_ids.extend(batch.id.tolist())
        labels.append(batch.y.reshape(-1, args.num_tasks))
    if ids != target_ids:
        raise ValueError("Evaluation loader order is inconsistent")
    labels = torch.cat(labels).numpy()
    return ids, labels, scores

def evaluate(args, model, loader, cache):
    ids, labels, scores = collect_predictions(args, model, loader, cache)
    report = roc_details(labels, scores)

    valid = labels != 0
    if valid.any():
        y = ((labels[valid].astype(np.float64) + 1.0) / 2.0)
        probs = np.clip(scores[valid].astype(np.float64), 1e-7, 1.0 - 1e-7)
        report["bce_loss"] = float(np.mean(-y * np.log(probs) - (1.0 - y) * np.log1p(-probs)))
    else:
        report["bce_loss"] = None
    return report, ids, labels, scores
