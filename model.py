#!/usr/bin/env python


import math
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import global_mean_pool
from base_gnn import GNN
from losses import motif_margin_loss, importance_cv2_loss

class GNN_topexpert(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.num_tasks, self.num_experts, self.emb_dim = args.num_tasks, args.num_experts, args.emb_dim
        self.gnn = GNN(args.num_layer, args.emb_dim, args.JK, args.dropout_ratio, args.gnn_type)
        self.pool = global_mean_pool
        self.mlp = nn.Linear(args.emb_dim, args.num_tasks)
        self.pos_experts_w = nn.Parameter(torch.empty(args.num_tasks, args.num_experts, args.emb_dim))
        self.neg_experts_w = nn.Parameter(torch.empty(args.num_tasks, args.num_experts, args.emb_dim))

        self.pos_linear = nn.Linear(args.emb_dim, args.emb_dim)
        self.neg_linear = nn.Linear(args.emb_dim, args.emb_dim)
        self.pos_k_weight = nn.Parameter(torch.empty(args.num_tasks, args.num_experts, args.emb_dim))
        self.neg_k_weight = nn.Parameter(torch.empty(args.num_tasks, args.num_experts, 2 * args.emb_dim))

        self.pred_weight = nn.Parameter(torch.empty(args.num_tasks, 2))
        self.pred_bias = nn.Parameter(torch.zeros(args.num_tasks))
        self.reset_experts()

        for t in range(self.num_tasks):
            nn.init.xavier_normal_(self.pos_k_weight[t])
            nn.init.xavier_normal_(self.neg_k_weight[t])

    def reset_experts(self):

        bound = 1 / math.sqrt(self.emb_dim)
        nn.init.uniform_(self.pos_experts_w, -bound, bound)
        nn.init.uniform_(self.neg_experts_w, -bound, bound)
        nn.init.uniform_(self.pred_weight, -1 / math.sqrt(2), 1 / math.sqrt(2))
        nn.init.zeros_(self.pred_bias)

    def encode(self, data):

        return self.gnn(data.x, data.edge_index, data.edge_attr) * getattr(self.args, "representation_scale", 1.0)

    def attribution_outputs(self, x, edge_index, edge_attr, batch, node_batch):
        """Compute whole-graph and masked BRICS-fragment logits."""
        nodes = self.gnn(x, edge_index, edge_attr) * getattr(self.args, "representation_scale", 1.0)
        whole = self.mlp(self.pool(nodes, batch))
        scores, ids = [], []
        for g in range(whole.shape[0]):
            h = nodes[batch == g]
            fragments = node_batch[batch == g].long()
            if torch.any(fragments < 0):
                raise ValueError("Incomplete BRICS partition; regenerate graph cache")
            masked = []
            for f in fragments.unique(sorted=True):

                masked.append(self.pool(h * (fragments == f).unsqueeze(-1), None).squeeze(0))
            scores.append(self.mlp(torch.stack(masked)))
            ids.append(fragments)
        return whole, scores, ids

    def motif_representations(self, data, pos_node, neg_node):
        """Pool whole-graph, positive-motif and negative-motif embeddings."""
        nodes = self.encode(data)
        whole = self.pool(nodes, data.batch)
        branches = []
        for selections in (pos_node, neg_node):
            per_task = []
            for t in range(self.num_tasks):
                mask = nodes.new_zeros(nodes.shape[0], 1)
                chosen = torch.cat([row[t] for row in selections]).to(nodes.device)
                mask[chosen] = 1
                per_task.append(self.pool(nodes * mask, data.batch))
            branches.append(torch.stack(per_task, dim=1))
        return whole, branches[0], branches[1]

    def pre_forward(self, data, pos_node, neg_node):
        whole, pos, neg = self.motif_representations(data, pos_node, neg_node)
        return whole, pos, neg, self.mlp(whole)

    def compute_r(self, query, keys):
        """Compute task-specific softmax routing with optional training noise."""

        if query.dim() != 3 or keys.dim() != 3 or query.shape[1] != keys.shape[0] or query.shape[2] != keys.shape[2]:
            raise ValueError("Task-specific router dimensions are inconsistent")
        logits = torch.einsum("btd,tkd->btk", query, keys) / self.args.tau
        if self.training and self.args.router_noise_std:
            logits = logits + torch.randn_like(logits) * self.args.router_noise_std
        return torch.softmax(logits, dim=-1)

    def forward_details(self, data, pos_node, neg_node):
        """Run the original full two-branch mixture-of-experts prediction."""
        whole, pos, neg = self.motif_representations(data, pos_node, neg_node)
        zp, zn = self.pos_linear(pos), self.neg_linear(neg)
        rp = self.compute_r(zp, self.pos_k_weight)
        rn = self.compute_r(torch.cat((zp, zn), dim=-1), self.neg_k_weight)
        self.pos_route_score, self.neg_route_score = rp, rn
        ep = torch.einsum("btd,tkd->btk", pos, self.pos_experts_w)
        en = torch.einsum("btd,tkd->btk", neg, self.neg_experts_w)
        op, on = (ep * rp).sum(-1), (en * rn).sum(-1)
        branches = torch.stack((op, on), dim=-1)
        logits = (branches * self.pred_weight).sum(-1) + self.pred_bias
        return logits, whole, pos, neg, op, on, rp, rn

    def forward(self, data, pos_node=None, neg_node=None):
        return self.forward_details(data, pos_node, neg_node)[0]

    def margin_loss(self, whole, pos, neg, labels):
        return motif_margin_loss(
            whole, pos, neg, labels,
            margin=self.args.margin,
            temperature=self.args.margin_temperature,
        )

    def compute_importance_and_loss(self):
        return importance_cv2_loss(
            self.pos_route_score,
            self.neg_route_score,
            gamma=self.args.gamma,
        )

def masked_bce(logits, labels):
    if logits.shape != labels.shape:
        raise ValueError("Logit and label dimensions do not match")
    valid = labels != 0
    if not valid.any():
        return logits.sum() * 0
    return F.binary_cross_entropy_with_logits(logits[valid], (labels[valid].to(logits.dtype) + 1) / 2)
