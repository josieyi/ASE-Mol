#!/usr/bin/env python

import copy
import math
import time
import torch
from pathlib import Path
from torch_geometric.loader import DataLoader
from configs import load_args
from loader import MoleculeDataset
from model import GNN_topexpert
from attribution import attribution_node, training_attribution_node
from splitters import data_split
from train_eval import pre_train, train, evaluate, collect_predictions
from metrics import roc_details, audit_dataset, require_usable_split
from training_optim import make_optimizer, UpdateSchedule, BatchOrderSampler
from util import (set_seed, write_json, serial_args, load_encoder,
                  cache_for, sync_device)

RULE = "=" * 86


def make_loader(args, dataset, shuffle=False):
    generator = torch.Generator().manual_seed(args.seed)
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle,
                      num_workers=args.num_workers, generator=generator)


def node_cache(args, selector, dataset, training=False):
    fn = training_attribution_node if training else attribution_node
    pos, neg = fn(args, selector, make_loader(args, dataset))
    return cache_for(dataset, pos, neg)


def consistency(old, new, dataset):
    equal, total = 0, 0
    for data in dataset:
        key = int(data.id.item())
        for t, y in enumerate(data.y.reshape(-1).tolist()):
            if y == 0:
                continue
            for branch in (0, 1):
                equal += int(torch.equal(old[key][branch][t], new[key][branch][t]))
                total += 1
    if not total:
        raise ValueError("Training set has no observed labels")
    return equal / total


def cpu_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def _log_line(handle, text=""):
    handle.write(str(text) + "\n")
    handle.flush()


def run_training(args):
    """Select the best model by validation ROC-AUC, then evaluate the test split."""
    set_seed(args.seed)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    args.output_dir = str(output)
    print("Run directory: " + str(output.resolve()), flush=True)

    dataset = MoleculeDataset(Path(args.dataset_dir) / args.dataset, args.dataset)
    training, validation, test = data_split(args, dataset)
    audit = audit_dataset(args, dataset, (training, validation, test))
    require_usable_split(audit)

    model = GNN_topexpert(args).to(args.device)
    if args.input_model_file:
        load_encoder(args.input_model_file, model)

    # The configuration has user-facing arguments only, without provenance hashes.
    config = serial_args(args)
    config.pop("split_seed", None)
    write_json(output / "config.json", {"args": config})

    log_file = output / (args.dataset + ".log")
    training_loader = make_loader(args, training, shuffle=True)
    selector, train_cache = None, None

    with log_file.open("x", encoding="utf-8") as log:
        _log_line(log, RULE)
        _log_line(log, "Dataset: %-7s Backbone: %-6s Split: %-10s Seed: %d" %
                  (args.dataset.upper(), args.gnn_type.upper(), args.split, args.seed))
        _log_line(log, "Samples: train=%d    validation=%d    test=%d" %
                  (len(training), len(validation), len(test)))
        _log_line(log, "Hyperparameters: batch=%s  lr=%s  weight_decay=%s  experts=%s  topk_rate=%s" %
                  (args.batch_size, args.lr, args.decay, args.num_experts, args.topk_rate))
        _log_line(log, "                 alpha=%s  beta=%s  pre_lr=%s  pre_epochs=%s  train_epochs=%s" %
                  (args.alpha, args.beta, args.pre_lr, args.pre_epoch, args.epochs))
        _log_line(log, "")
        _log_line(log, "[1] Pre-training")
        _log_line(log, "  Epoch    Loss")
        _log_line(log, "  -----    --------")

        optimizer, _ = make_optimizer(args, model, stage="recognition")
        pre_schedule = UpdateSchedule(args, optimizer, len(training_loader), stage="recognition")
        previous, next_refresh = None, 1
        for epoch in range(1, args.pre_epoch + 1):
            consistency_score = None
            sync_device(args.device)
            start = time.perf_counter()
            if epoch >= next_refresh:
                fresh = node_cache(args, model, training, training=True)
                if previous is None:
                    interval = args.pre_inner
                else:
                    consistency_score = consistency(previous, fresh, training)
                    interval = max(1, math.ceil(args.pre_inner * consistency_score))
                train_cache, previous = fresh, fresh
                next_refresh = epoch + interval
            stats = pre_train(args, model, training_loader, optimizer, train_cache, pre_schedule)
            sync_device(args.device)
            _log_line(log, "  %5d    %8.4f" % (epoch, stats["loss"]))

        selector = copy.deepcopy(model).requires_grad_(False).eval()
        train_cache = node_cache(args, selector, training, training=True)
        model.reset_experts()
        valid_cache = node_cache(args, selector, validation)
        base_loader = make_loader(args, training, shuffle=True)
        batch_sampler = BatchOrderSampler(base_loader.batch_sampler, len(training))
        training_loader = DataLoader(training, batch_sampler=batch_sampler,
                                     num_workers=args.num_workers,
                                     generator=torch.Generator().manual_seed(args.seed))
        optimizer, _ = make_optimizer(args, model, stage="expert")
        scheduler = UpdateSchedule(args, optimizer, len(training_loader), stage="expert")

        _log_line(log, "")
        _log_line(log, "[2] Main training / validation")
        _log_line(log, "  Epoch    Train loss    Valid loss    Valid score    Selection")
        _log_line(log, "  -----    ----------    ----------    -----------    ----------")
        best, best_epoch, stale = float("-inf"), 0, 0
        best_state = None
        for epoch in range(1, args.epochs + 1):
            batch_sampler.set_epoch(epoch)
            stats = train(args, model, training_loader, optimizer, train_cache, scheduler)
            metrics, _, _, _ = evaluate(args, model, make_loader(args, validation), valid_cache)
            score = metrics["roc_auc"]
            print("epoch=%d validation_auc=%.4f" % (epoch, score), flush=True)
            selected = score > best
            if selected:
                best, best_epoch, stale = score, epoch, 0
                best_state = cpu_state(model)
            else:
                stale += 1
            _log_line(log, "  %5d    %10.4f    %10.4f    %11.4f    %s" %
                      (epoch, stats["loss"], metrics["bce_loss"], score,
                       "best" if selected else "-"))
            if args.patience and stale >= args.patience:
                break

        if best_state is None:
            raise RuntimeError("No validation checkpoint was selected")
        model.load_state_dict(best_state, strict=True)
        model.eval()
        del best_state

        _log_line(log, "")
        _log_line(log, "[3] Final model selection / test")
        _log_line(log, "  Selected epoch       : %d" % best_epoch)
        _log_line(log, "  Best validation score: %.4f" % best)

        test_cache = node_cache(args, selector, test)
        _, labels, scores = collect_predictions(
            args, model, make_loader(args, test), test_cache)
        report = roc_details(labels, scores)
        _log_line(log, "  Corresponding test   : %.4f" % report["roc_auc"])
        _log_line(log, RULE)

    summary = {
        "best_validation_auc": best,
        "best_epoch": best_epoch,
        "test_roc_auc": report["roc_auc"],
    }
    write_json(output / "summary.json", summary)
    print("test_auc=%.4f" % report["roc_auc"], flush=True)
    print("Artifacts: config.json, summary.json, %s.log" % args.dataset, flush=True)
    return summary


def main(args):
    return run_training(args)


if __name__ == "__main__":
    main(load_args())
