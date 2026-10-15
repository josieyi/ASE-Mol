#!/usr/bin/env python

import hashlib
import inspect
import json
import random
from pathlib import Path
import numpy as np
import torch

def load_trusted_artifact(path, map_location="cpu", tensor_only=True):
    kwargs = {"map_location": map_location}
    if "weights_only" in inspect.signature(torch.load).parameters:
        kwargs["weights_only"] = tensor_only
    return torch.load(path, **kwargs)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def serial_args(args):
    return {k: str(v) if isinstance(v, (torch.device, Path)) else v for k, v in vars(args).items()}

def load_encoder(path, model):
    state = load_trusted_artifact(path, map_location="cpu")
    model.gnn.load_state_dict(state, strict=True)

def cache_for(dataset, pos, neg):
    ids = [int(d.id.item()) for d in dataset]
    if len(set(ids)) != len(ids) or len(ids) != len(pos) or len(pos) != len(neg):
        raise ValueError("Attribution cache IDs are not unique or lengths differ")
    return {i: (p, n) for i, p, n in zip(ids, pos, neg)}

def batch_nodes(batch, cache):
    if cache is None:
        return None, None
    pos, neg = [], []
    for g, item_id in enumerate(batch.id.cpu().tolist()):
        p, n = cache[int(item_id)]
        offset = int(batch.ptr[g])
        pos.append([v.to(batch.x.device) + offset for v in p])
        neg.append([v.to(batch.x.device) + offset for v in n])
    return pos, neg

def sync_device(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
