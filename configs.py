#!/usr/bin/env python

import argparse
import math

TASKS = dict(tox21=12, bace=1, bbbp=1, toxcast=617, sider=27, clintox=2, hiv=1, muv=17)

SPLIT = "scaffold"
SPLIT_SEED = 0

OPTIMIZATION = {
    "recognition": {
        "optimizer": "adam",
        "scheduler": "cosine",
        "warmup_epochs": 5,
        "min_lr_ratio": 0.05,
        "lr_scales": {"encoder": 1.0, "head": 1.0, "router": 1.0},
        "layer_lr_decay": 1.0,
    },
    "expert": {
        "optimizer": "adamw",
        "scheduler": "cosine",
        "warmup_epochs": 10,
        "min_lr_ratio": 0.05,
        "lr_scales": {"encoder": 0.8, "head": 1.0, "router": 0.3},
        "layer_lr_decay": 0.8,
    },
}
NO_DECAY_BIAS_NORM = True
LOSS_POLICY = {
    "margin_loss": "ranking_hinge",
    "importance_loss": "stop_grad",
    "importance_scope": "all",
}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset_dir", default="./data/")
    p.add_argument("--dataset", choices=tuple(TASKS), default="bbbp")
    p.add_argument("--output_dir", default="runs/bbbp_gcn_seed0")
    p.add_argument("--log_file", default="", help="Optional extra formatted training log path")
    p.add_argument("--log_only_run", default="", help="Format a recorded run without training")
    p.add_argument("--seed", type=int, default=0, help="Training seed; does not change the fixed scaffold split")
    p.add_argument("--split", choices=(SPLIT,), default=SPLIT,
                   help="Fixed balanced scaffold algorithm (short CLI name: scaffold)")
    p.add_argument("--split_file", default="", help="Optional fixed scaffold split manifest")
    p.add_argument("--device", default="auto")
    p.add_argument("--device_no", type=int, default=0)
    p.add_argument("--pre_epoch", type=int, default=100)
    p.add_argument("--pre_inner", type=int, default=10)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--num_experts", type=int, default=3)
    p.add_argument("--topk_rate", type=float, default=0.1)
    p.add_argument("--num_layer", type=int, default=5)
    p.add_argument("--emb_dim", type=int, default=300)
    p.add_argument("--dropout_ratio", type=float, default=0.5)
    p.add_argument("--graph_pooling", choices=("mean",), default="mean")
    p.add_argument("--representation_scale", type=float, default=1.0)
    p.add_argument("--JK", choices=("last", "sum", "max"), default="last")
    p.add_argument("--gnn_type", choices=("gcn", "gin", "gat", "graphsage"), default="gcn")
    p.add_argument("--pre_lr", type=float, default=1e-3, help="Recognition-stage base learning rate")
    p.add_argument("--lr", type=float, default=1e-3, help="Expert-stage head base learning rate")
    p.add_argument("--decay", type=float, default=1e-5, help="Weight decay, not learning rate decay")
    p.add_argument("--alpha", dest="pre_cl", type=float, default=0.1)
    p.add_argument("--beta", dest="cl", type=float, default=0.1)
    p.add_argument("--tau", type=float, default=0.1)
    p.add_argument("--gamma", type=float, default=0.1)
    p.add_argument("--margin", type=float, default=0.5)
    p.add_argument("--margin_temperature", type=float, default=1.0)
    p.add_argument("--router_noise_std", type=float, default=0.01)
    p.add_argument("--input_model_file", "-i", default="", help="Strictly compatible GNN encoder state_dict")
    p.add_argument("--ckpt_all", "-c", default="", help="Full checkpoint for evaluation")
    p.add_argument("--validate_only", action="store_true", help="Skip test prediction")
    p.add_argument("--eval_only", action="store_true")
    p.add_argument("--preflight_only", action="store_true", help="Check splits before training")
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--patience", type=int, default=0, help="0 disables early stopping")
    a = p.parse_args(argv)
    a.num_tasks = TASKS[a.dataset]
    # Compatibility and manifest metadata, not separate configurable parameters.
    a.split_seed = SPLIT_SEED
    if not 0 < a.topk_rate <= 0.5:
        p.error("topk_rate must lie in (0, 0.5]")
    if min(a.num_experts, a.emb_dim, a.batch_size, a.epochs, a.pre_inner, a.pre_epoch) < 1 or a.num_layer < 2:
        p.error("Training sizes must be positive; num_layer >= 2")
    if not 0 <= a.dropout_ratio < 1 or a.tau <= 0 or a.router_noise_std < 0:
        p.error("Invalid dropout, temperature, or routing noise")
    if a.representation_scale <= 0 or a.margin_temperature <= 0 or not 0 <= a.margin <= 1:
        p.error("Invalid representation scale or motif margin settings")
    if a.lr <= 0 or a.pre_lr <= 0 or a.decay < 0 or min(a.cl, a.pre_cl, a.gamma) < 0:
        p.error("Learning rates must be positive and regularization nonnegative")
    if any(isinstance(v, float) and not math.isfinite(v) for v in vars(a).values()):
        p.error("Floating-point arguments must be finite")
    if a.grad_clip < 0 or a.patience < 0 or min(a.seed, a.num_workers, a.device_no) < 0:
        p.error("Invalid clipping, patience, seed, or worker settings")
    for stage, epochs in (("recognition", a.pre_epoch), ("expert", a.epochs)):
        warmup = OPTIMIZATION[stage]["warmup_epochs"]
        if warmup >= epochs:
            p.error("%s training requires more than %d epochs for its fixed warmup" % (stage, warmup))
    if a.eval_only and not a.ckpt_all:
        p.error("eval_only requires ckpt_all")
    if a.preflight_only and (a.eval_only or a.validate_only or a.ckpt_all):
        p.error("preflight_only is incompatible with eval_only/validate_only/ckpt_all")
    if a.ckpt_all and not a.eval_only:
        p.error("ckpt_all is evaluation-only; resume training is not supported")
    return a


def load_args(argv=None):
    import torch
    a = parse_args(argv)
    if a.device == "auto":
        a.device = "cuda:" + str(a.device_no) if torch.cuda.is_available() else "cpu"
    a.device = torch.device(a.device)
    return a
