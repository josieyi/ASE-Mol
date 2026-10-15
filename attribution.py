#!/usr/bin/env python

import math
import torch
import numpy as np

def stable_descending_indices(values):
    values = np.asarray(values)
    if values.ndim != 2 or not np.issubdtype(values.dtype, np.floating):
        raise ValueError("Attribution scores must be a 2D floating-point array")
    if not np.isfinite(values).all():
        raise FloatingPointError("Attribution scores contain NaN or Inf")
    return np.argsort(-values, axis=0, kind="mergesort")


def isin_1d(elements, candidates):
    if elements.dim() != 1 or candidates.dim() != 1:
        raise ValueError("Membership test expects 1D tensors")
    return (elements.unsqueeze(1) == candidates.unsqueeze(0)).any(dim=1)

def _select(fragment_scores, whole_scores, fragment_ids, rate, directions):
    result_pos, result_neg = [], []
    for i, (sub, whole, fragments) in enumerate(zip(fragment_scores, whole_scores, fragment_ids)):
        count = len(fragments.unique())
        k = max(1, math.floor(rate * count))

        delta = (sub - whole.unsqueeze(0)) * directions[i].unsqueeze(0)
        order = torch.as_tensor(stable_descending_indices(delta.detach().cpu().numpy()),
                                dtype=torch.long, device=delta.device)
        ids = fragments.unique(sorted=True)
        pos, neg = [], []
        for t in range(delta.shape[1]):
            pos.append(torch.nonzero(isin_1d(fragments, ids[order[:k, t]]), as_tuple=True)[0])
            neg.append(torch.nonzero(isin_1d(fragments, ids[order[-k:, t]]), as_tuple=True)[0])
        result_pos.append(pos)
        result_neg.append(neg)
    return result_pos, result_neg

def _prediction_directions(whole):
    return torch.where(whole >= 0, torch.ones_like(whole), -torch.ones_like(whole))

def attribution_node(args, model, loader):
    return _collect(args, model, loader, use_training_labels=False)

def training_attribution_node(args, model, loader):
    return _collect(args, model, loader, use_training_labels=True)

def _collect(args, model, loader, *, use_training_labels):
    was_training = model.training
    pos_all, neg_all = [], []
    model.eval()
    try:
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(args.device)

                whole, fragment_scores, fragment_ids = model.attribution_outputs(
                    batch.x, batch.edge_index, batch.edge_attr, batch.batch, batch.node_batch)
                directions = _prediction_directions(whole)
                if use_training_labels:
                    labels = batch.y.reshape_as(whole)
                    if not torch.all((labels == -1) | (labels == 0) | (labels == 1)):
                        raise ValueError("Labels must use -1/0/1 encoding")
                    directions = torch.where(labels != 0, labels.to(whole.dtype), directions)
                pos, neg = _select(fragment_scores, whole, fragment_ids,
                                   args.topk_rate, directions)
                pos_all.extend([[v.cpu() for v in row] for row in pos])
                neg_all.extend([[v.cpu() for v in row] for row in neg])
    finally:
        model.train(was_training)
    return pos_all, neg_all
