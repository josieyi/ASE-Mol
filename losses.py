#!/usr/bin/env python

import torch
from torch.nn import functional as F

def similarity_scores(whole, pos, neg, temperature=1.0):

    if temperature <= 0:
        raise ValueError("margin_temperature must be positive")
    if whole.ndim != 2 or pos.ndim != 3 or neg.shape != pos.shape:
        raise ValueError("expected whole [B,D] and pos/neg [B,T,D]")
    if whole.shape[0] != pos.shape[0] or whole.shape[1] != pos.shape[2]:
        raise ValueError("motif similarity dimensions do not match")
    expanded = whole.unsqueeze(1)
    pos_logits = (expanded * pos).sum(-1) / temperature
    neg_logits = (expanded * neg).sum(-1) / temperature
    return torch.sigmoid(pos_logits), torch.sigmoid(neg_logits)

def motif_margin_loss(whole, pos, neg, labels, margin=0.5, temperature=1.0):

    sp, sn = similarity_scores(whole, pos, neg, temperature)
    if labels.shape != sp.shape:
        raise ValueError("labels must have shape [B,T]")
    observed = labels != 0
    if not bool(observed.any()):
        return (whole.sum() + pos.sum() + neg.sum()) * 0.0
    losses = F.relu(margin - sp + sn)
    return losses[observed].mean()


def importance_cv2_loss(positive_routes, negative_routes, gamma=0.1):

    if gamma < 0:
        raise ValueError("gamma must be nonnegative")
    if positive_routes.shape != negative_routes.shape or positive_routes.ndim != 3:
        raise ValueError("route scores must have the same shape [B,T,K]")
    result = positive_routes.sum() * 0.0
    for routes in (positive_routes, negative_routes):
        importance = routes.sum(dim=0)
        avg = importance.mean(dim=-1)
        cv2 = importance.var(dim=-1, unbiased=False) / avg.square().clamp_min(1e-12)
        penalty = torch.where(cv2 < gamma, cv2.detach(), cv2)
        result = result + penalty.mean()
    return result
