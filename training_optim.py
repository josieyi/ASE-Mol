#!/usr/bin/env python

import math
import torch
from configs import OPTIMIZATION, NO_DECAY_BIAS_NORM


def learning_rate_factor(step, total_steps, warmup_steps=0, min_ratio=0.0):
    if not (0 <= step < total_steps and 0 <= warmup_steps < total_steps and 0 <= min_ratio <= 1):
        raise ValueError("Invalid learning-rate schedule")
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    remaining = total_steps - warmup_steps
    progress = (step - warmup_steps) / (remaining - 1) if remaining > 1 else 0.0
    multiplier = .5 * (1 + math.cos(math.pi * progress))
    return min_ratio + (1 - min_ratio) * multiplier


def parameter_role(name, num_layers):
    parts = name.split(".")
    if parts[0] == "gnn":
        if parts[1] in ("x_embedding1", "x_embedding2"):
            return "encoder", 0
        if parts[1] in ("gnns", "batch_norms"):
            depth = int(parts[2]) + 1
            if not 1 <= depth <= num_layers:
                raise ValueError("GNN layer index out of range: " + name)
            return "encoder", depth
        raise ValueError("Unknown GNN parameter group: " + name)
    if parts[0] in ("pos_linear", "neg_linear", "pos_k_weight", "neg_k_weight"):
        return "router", None
    if parts[0] in ("mlp", "pos_experts_w", "neg_experts_w", "pred_weight", "pred_bias"):
        return "head", None
    raise ValueError("Unknown parameter group: " + name)


def clip_grad_norm_finite(parameters, max_norm):
    if not math.isfinite(max_norm) or max_norm < 0:
        raise ValueError("Gradient clipping threshold must be finite and nonnegative")
    gradients = [p.grad for p in parameters if p.grad is not None]
    if not gradients:
        return torch.tensor(0.)
    with torch.no_grad():
        total = torch.norm(torch.stack([torch.norm(g.detach(), 2) for g in gradients]), 2)
        if not torch.isfinite(total):
            raise RuntimeError("Non-finite gradient norm; update was skipped")
        if max_norm:
            coefficient = (max_norm / (total + 1e-6)).clamp(max=1.)
            for gradient in gradients:
                gradient.mul_(coefficient)
    return total


def make_optimizer(args, model, stage="expert"):
    if stage not in ("recognition", "expert"):
        raise ValueError("Unknown training stage")
    norm_ids = {id(p) for m in model.modules()
                if isinstance(m, (torch.nn.modules.batchnorm._BatchNorm, torch.nn.LayerNorm, torch.nn.GroupNorm))
                for p in m.parameters(recurse=False)}
    groups, manifest = {}, []
    policy = OPTIMIZATION[stage]
    base_lr = args.pre_lr if stage == "recognition" else args.lr
    rates = {role: base_lr * scale for role, scale in policy["lr_scales"].items()}
    layer_decay = policy["layer_lr_decay"]
    optimizer_kind = policy["optimizer"]
    seen = set()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if stage == "recognition" and not (name.startswith("gnn.") or name.startswith("mlp.")):
            continue
        if id(parameter) in seen:
            raise ValueError("Duplicate optimizer parameter: " + name)
        seen.add(id(parameter))
        role, depth = parameter_role(name, args.num_layer)
        rate = rates[role]
        if role == "encoder":
            rate *= layer_decay ** (args.num_layer - depth)
        if rate <= 0:
            raise ValueError("Layer-wise learning rate underflow: " + name)
        no_decay = NO_DECAY_BIAS_NORM and (name.endswith("bias") or id(parameter) in norm_ids)
        decay = 0.0 if no_decay else args.decay
        key = (role, depth, rate, decay)
        if key not in groups:
            groups[key] = {"params": [], "lr": rate, "weight_decay": decay,
                           "group_name": "%s/layer_%s/decay_%g" % (role, depth, decay)}
        groups[key]["params"].append(parameter)
        manifest.append({"parameter": name, "role": role, "depth": depth, "lr": rate,
                         "weight_decay": decay, "numel": parameter.numel()})
    if stage == "recognition":
        expected = {id(p) for name, p in model.named_parameters()
                    if p.requires_grad and (name.startswith("gnn.") or name.startswith("mlp."))}
    else:
        expected = {id(p) for p in model.parameters() if p.requires_grad}
    if seen != expected:
        raise ValueError("Incomplete optimizer parameter coverage")
    cls = torch.optim.Adam if optimizer_kind == "adam" else torch.optim.AdamW
    return cls(list(groups.values())), manifest

class UpdateSchedule:
    def __init__(self, args, optimizer, steps_per_epoch, stage):
        policy = OPTIMIZATION[stage]
        epochs = args.pre_epoch if stage == "recognition" else args.epochs
        self.warmup_steps = policy["warmup_epochs"] * steps_per_epoch
        self.min_ratio = policy["min_lr_ratio"]
        self.total_steps = epochs * steps_per_epoch
        self.optimizer = optimizer
        self.base_rates = [g["lr"] for g in optimizer.param_groups]
        self.step_index = 0

    def before_step(self):
        factor = learning_rate_factor(self.step_index, self.total_steps,
                                      self.warmup_steps, self.min_ratio)
        for group, rate in zip(self.optimizer.param_groups, self.base_rates):
            group["lr"] = rate * factor
        self.step_index += 1

def gradient_step(args, loss, model, optimizer, schedule=None):
    if not torch.isfinite(loss):
        raise FloatingPointError("Loss must be finite")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()

    norm = clip_grad_norm_finite(model.parameters(), args.grad_clip)
    if schedule is not None:
        schedule.before_step()
    optimizer.step()
    return float(norm.detach())

class BatchOrderSampler:
    def __init__(self, base_sampler, size):
        self.base_sampler, self.size = base_sampler, size
        self.epoch = 0
        self.last_stats = None

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        batches = list(self.base_sampler)
        self.last_stats = {"samples": self.size, "batches": len(batches)}
        yield from batches

    def __len__(self):
        return len(self.base_sampler)
