#!/usr/bin/env python


import copy
import hashlib
import sys
import json
import math
import platform
import time
from pathlib import Path
from types import SimpleNamespace
import torch
from torch_geometric.loader import DataLoader
from configs import load_args, OPTIMIZATION, SPLIT_SEED, NO_DECAY_BIAS_NORM, LOSS_POLICY
from loader import MoleculeDataset, FEATURE_VERSION
from model import GNN_topexpert
from attribution import attribution_node, training_attribution_node
from splitters import data_split
from train_eval import pre_train, train, evaluate, collect_predictions
from metrics import UndefinedROCError, save_test_result, audit_dataset, require_usable_split
from training_optim import make_optimizer, UpdateSchedule, BatchOrderSampler
from util import (set_seed, write_json, serial_args, sha256, load_encoder,
                  cache_for, sync_device, load_trusted_artifact)

RULE = "=" * 86

def _read_json(path):
    path = Path(path)
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))

def _sha16(values):
    payload = json.dumps(values, ensure_ascii=False, sort_keys=False,
                         separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]

def _score(value):
    return "%0.4f" % value if value is not None else "N/A"

def render_run_log(output_dir):
    """Format an existing completed or partially completed run; no retraining."""
    root = Path(output_dir)
    conf = _read_json(root / "config.json")
    settings = conf.get("args", {})
    split = _read_json(root / "split.json")
    summary = _read_json(root / "summary.json")
    if not settings or not split:
        raise FileNotFoundError("config.json / split.json not found in " + str(root))
    records = []
    with (root / "epochs.jsonl").open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError("Invalid epochs.jsonl line %d" % line_number) from exc
    pre = [x for x in records if x.get("stage") == "recognition"]
    expert = [x for x in records if x.get("stage") == "expert"]
    dataset, backbone = settings["dataset"], settings["gnn_type"]
    train_ids = split["train_ids"]
    valid_ids = split["valid_ids"]
    test_ids = split["test_ids"]
    lines = [RULE,
        "Dataset: %-7s Backbone: %-6s Split: %-17s Seed: %d" %
        (dataset.upper(), backbone.upper(), settings["split"], settings["seed"]),
        "Scaffold variant: %s  Fixed split seed: %s" %
        (split.get("scaffold_variant", "balanced"), split.get("split_seed", "N/A")),
        "Samples: train=%d    validation=%d    test=%d" %
        (len(train_ids), len(valid_ids), len(test_ids)),
        "Split fingerprints: train=%s  valid=%s  test=%s" %
        (_sha16(train_ids), _sha16(valid_ids), _sha16(test_ids)),
        "Hyperparameters: batch=%s  lr=%s  weight_decay=%s  experts=%s  topk_rate=%s" %
        (settings["batch_size"], settings["lr"], settings["decay"],
         settings["num_experts"], settings["topk_rate"]),
        "                 pre_lr=%s  split_seed=%s" % (settings.get("pre_lr", "N/A"), split.get("split_seed", "N/A")),
        "                 alpha=%s  beta=%s  pre_epochs=%s  train_epochs=%s  selection_start=1" %
        (settings["pre_cl"], settings["cl"], settings["pre_epoch"], settings["epochs"]),
        "                 margin_loss=%s  importance_loss=%s  margin_temperature=%s  importance_scope=%s" %
        (LOSS_POLICY["margin_loss"], LOSS_POLICY["importance_loss"],
         settings.get("margin_temperature", 1.0), LOSS_POLICY["importance_scope"]),
        "", "[1] Pre-training", "  Epoch    Loss", "  -----    --------"]
    for row in pre:
        lines.append("  %5d    %8.4f" % (row["epoch"], row["loss"]))
    lines.extend(["", "[2] Main training / validation",
                  "  Epoch    Train loss    Valid loss    Valid score    Checkpoint",
                  "  -----    ----------    ----------    -----------    ----------"])
    best_score = float("-inf")
    best_epoch = summary.get("best_epoch")
    if best_epoch is None and expert:
        best_epoch = max(expert, key=lambda r: r["validation"]["roc_auc"])["epoch"]
    for row in expert:
        score = row["validation"]["roc_auc"]
        saved = score > best_score
        if saved:
            best_score = score
        mark = "*" if best_epoch == row["epoch"] else ("saved" if saved else "-")
        loss = row.get("validation_loss", row.get("validation", {}).get("bce_loss"))
        valid_text = "%10.4f" % loss if loss is not None else "%10s" % "N/A"
        lines.append("  %5d    %10.4f    %s    %11.4f    %s" %
                     (row["epoch"], row["loss"], valid_text, score, mark))
    lines.extend(["  * validation-selected epoch", "",
                  "[3] Final model selection / test",
                  "  Selected epoch       : %s" % (best_epoch if best_epoch is not None else "N/A"),
                  "  Best validation score: %s" % _score(summary.get("best_validation_auc", best_score if expert else None))])
    test = _read_json(root / "test_metrics.json")
    if summary.get("test_evaluated") and test.get("status") == "ok":
        lines.append("  Corresponding test   : %s" % _score(test.get("roc_auc")))
    else:
        lines.append("  Corresponding test   : NOT EVALUATED (%s)" %
                     summary.get("test_status", "run incomplete"))
    lines.extend(["  Selected checkpoint epoch: %s" %
                 (best_epoch if best_epoch is not None else "N/A"),
                 "  Selection rule: validation score only; no test-based checkpoint selection.",
                 "  Retention: checkpoint used internally; only three report files retained.", RULE])
    return "\n".join(lines) + "\n"

SCHEMA = "ase-corrected-loss-v1"

def make_loader(args, dataset, shuffle=False):
    generator = torch.Generator().manual_seed(args.seed)
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle,
                      num_workers=args.num_workers, generator=generator)

def node_cache(args, selector, dataset, training=False):
    """Cache motif indices. Ground-truth directions are training-only."""
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

def parameter_counts(model):
    return {"total": sum(p.numel() for p in model.parameters()),
            "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "expert_weights": model.pos_experts_w.numel() + model.neg_experts_w.numel(),
            "router_keys": model.pos_k_weight.numel() + model.neg_k_weight.numel()}

def load_run(cli):
    checkpoint = load_trusted_artifact(cli.ckpt_all, map_location="cpu")
    if checkpoint.get("schema") != SCHEMA:
        raise ValueError("Checkpoint schema mismatch; refusing incompatible weights")
    args = SimpleNamespace(**checkpoint["args"])
    args.device, args.dataset_dir, args.output_dir = cli.device, cli.dataset_dir, cli.output_dir
    dataset = MoleculeDataset(Path(args.dataset_dir) / args.dataset, args.dataset)
    if dataset.raw_sha256 != checkpoint["raw_sha256"]:
        raise ValueError("Checkpoint raw-data fingerprint mismatch")
    model = GNN_topexpert(args).to(args.device)
    model.load_state_dict(checkpoint["model"], strict=True)
    selector = None
    if checkpoint["selector"] is not None:
        selector = GNN_topexpert(args).to(args.device)
        selector.load_state_dict(checkpoint["selector"], strict=True)
        selector.requires_grad_(False).eval()
    lookup = {int(d.id.item()): i for i, d in enumerate(dataset)}
    manifest = checkpoint["split_manifest"]
    parts = [[lookup[v] for v in manifest[k]] for k in ("train_ids", "valid_ids", "test_ids")]
    flat = [i for p in parts for i in p]
    if sorted(flat) != list(range(len(dataset))) or any(not p for p in parts):
        raise ValueError("Checkpoint split is incomplete or overlapping")
    subsets = tuple(dataset[torch.tensor(p, dtype=torch.long)] for p in parts)
    return args, model.eval(), selector, subsets, checkpoint

def final_evaluation(args, model, selector, test_dataset, output):
    """Evaluate the held-out test split after checkpoint selection."""
    cache = node_cache(args, selector, test_dataset)
    ids, labels, scores = collect_predictions(args, model, make_loader(args, test_dataset), cache)
    return save_test_result(output, ids, labels, scores)

def run_training(args):
    """Train the full model; select checkpoints by validation ROC-AUC."""
    set_seed(args.seed)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    args.output_dir = str(output)
    print("Run directory: " + str(output.resolve()), flush=True)
    if args.eval_only:
        args, model, selector, subsets, checkpoint = load_run(args)
        report = final_evaluation(args, model, selector, subsets[2], output)
        print(json.dumps(report, ensure_ascii=False))
        return
    dataset = MoleculeDataset(Path(args.dataset_dir) / args.dataset, args.dataset)
    subsets = data_split(args, dataset)
    training, validation, test = subsets
    audit = audit_dataset(args, dataset, subsets)
    write_json(output / "data_audit.json", audit)
    require_usable_split(audit, output / "data_audit.json")
    if args.preflight_only:
        print(json.dumps(audit, ensure_ascii=False))
        return
    model = GNN_topexpert(args).to(args.device)
    if args.input_model_file:
        load_encoder(args.input_model_file, model)
    import torch_geometric
    import rdkit
    config = serial_args(args)
    split_manifest = json.loads((output / "split.json").read_text(encoding="utf-8"))
    provenance = {"schema": SCHEMA, "args": config, "raw_sha256": dataset.raw_sha256,
                  "feature_version": FEATURE_VERSION, "python": platform.python_version(),
                  "torch": str(torch.__version__), "torch_geometric": torch_geometric.__version__,
                  "rdkit": rdkit.__version__,
                  "source_sha256": {p.name: sha256(p) for p in (
                       Path(__file__).parent / name for name in (
                           "main.py", "attribution.py", "base_gnn.py", "configs.py",
                           "loader.py", "losses.py", "metrics.py", "model.py",
                           "splitters.py", "train_eval.py",
                           "training_optim.py", "util.py")) if p.is_file()},
                  "pretrained_sha256": sha256(args.input_model_file) if args.input_model_file else None,
                  "parameters": parameter_counts(model),
                  "optimization_policy": OPTIMIZATION,
                  "loss_policy": LOSS_POLICY,
                  "fixed_split_seed": SPLIT_SEED,
                  "no_decay_bias_norm": NO_DECAY_BIAS_NORM}
    write_json(output / "config.json", provenance)
    training_loader = make_loader(args, training, shuffle=True)
    selector, train_cache = None, None
    with (output / "epochs.jsonl").open("x", encoding="utf-8") as log:
        optimizer, parameter_groups = make_optimizer(args, model, stage="recognition")
        write_json(output / "optimizer_recognition.json", parameter_groups)
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
            stats.update(stage="recognition", epoch=epoch, next_refresh=next_refresh,
                         consistency=consistency_score, epoch_seconds=time.perf_counter() - start,
                         optimizer_steps=pre_schedule.step_index,
                         learning_rates=[g["lr"] for g in optimizer.param_groups])
            log.write(json.dumps(stats, ensure_ascii=False) + "\n")
            log.flush()
        selector = copy.deepcopy(model).requires_grad_(False).eval()
        train_cache = node_cache(args, selector, training, training=True)
        model.reset_experts()
        valid_cache = node_cache(args, selector, validation)
        base_loader = make_loader(args, training, shuffle=True)
        batch_sampler = BatchOrderSampler(base_loader.batch_sampler, len(training))
        training_loader = DataLoader(training, batch_sampler=batch_sampler, num_workers=args.num_workers,
                                     generator=torch.Generator().manual_seed(args.seed))
        optimizer, parameter_groups = make_optimizer(args, model, stage="expert")
        write_json(output / "optimizer_expert.json", parameter_groups)
        scheduler = UpdateSchedule(args, optimizer, len(training_loader), stage="expert")
        best, best_epoch, stale = float("-inf"), 0, 0
        for epoch in range(1, args.epochs + 1):
            batch_sampler.set_epoch(epoch)
            stats = train(args, model, training_loader, optimizer, train_cache, scheduler)
            metrics, _, _, _ = evaluate(args, model, make_loader(args, validation), valid_cache)
            score = metrics["roc_auc"]
            stats.update(stage="expert", epoch=epoch, validation=metrics, validation_loss=metrics["bce_loss"],
                         learning_rates=[g["lr"] for g in optimizer.param_groups],
                         optimizer_steps=scheduler.step_index, batch_order=batch_sampler.last_stats)
            log.write(json.dumps(stats, ensure_ascii=False) + "\n")
            log.flush()
            print("epoch=%d validation_auc=%.4f" % (epoch, score), flush=True)
            if score > best:
                best, best_epoch, stale = score, epoch, 0
                torch.save({"schema": SCHEMA, "args": config, "model": cpu_state(model),
                            "selector": cpu_state(selector) if selector is not None else None,
                            "raw_sha256": dataset.raw_sha256, "split_manifest": split_manifest,
                            "best_validation_auc": best, "best_epoch": best_epoch,
                            "provenance": provenance}, output / "best.pt")
            else:
                stale += 1
            if args.patience and stale >= args.patience:
                break
    summary = {"schema": SCHEMA, "best_validation_auc": best, "best_epoch": best_epoch,
               "test_evaluated": False, "test_attempted": False,
               "test_status": "not_requested" if args.validate_only else "pending"}
    write_json(output / "summary.json", summary)
    if not args.validate_only:
        checkpoint = load_trusted_artifact(output / "best.pt", map_location=args.device)
        model.load_state_dict(checkpoint["model"], strict=True)
        summary.update(test_attempted=True, test_status="running")
        write_json(output / "summary.json", summary)
        try:
            report = final_evaluation(args, model, selector, test, output)
        except UndefinedROCError:
            summary.update(test_evaluated=True, test_status="undefined_auc", test_metric_defined=False)
            write_json(output / "summary.json", summary)
            raise
        except Exception as exc:
            summary.update(test_evaluated=False, test_status="failed", test_metric_defined=False,
                           test_error={"type": type(exc).__name__, "message": str(exc)})
            write_json(output / "summary.json", summary)
            raise
        summary.update(test_evaluated=True, test_status="complete", test_metric_defined=True)
        write_json(output / "summary.json", summary)
        print("test_auc=%.4f" % report["roc_auc"])

def save_formatted_log(run_dir, extra_path=""):
    """Render a readable log from this run's recorded JSON artifacts."""
    run_dir = Path(run_dir)
    log_text = render_run_log(run_dir)
    settings = _read_json(run_dir / "config.json").get("args", {})
    target = run_dir / (settings.get("dataset", "run") + ".log")
    target.write_text(log_text, encoding="utf-8")
    print("Formatted training log: " + str(target.resolve()), flush=True)
    if extra_path:
        additional = Path(extra_path)
        additional.parent.mkdir(parents=True, exist_ok=True)
        with additional.open("x", encoding="utf-8") as handle:
            handle.write(log_text)

def main(args):
    """Keep only config.json, summary.json, and <dataset>.log for each completed run."""
    if args.log_only_run:
        raise ValueError('--log_only_run requires intermediate epochs.jsonl/split.json; '
                         'use the existing <dataset>.log in the three-file package')
    if args.eval_only:
        raise ValueError('--eval_only requires a retained checkpoint; '
                         'three-file mode intentionally does not retain best.pt')
    if args.log_file:
        raise ValueError('--log_file would create another output file; '
                         'three-file mode writes <dataset>.log to its run directory')

    run_training(args)
    if args.preflight_only:
        return

    output = Path(args.output_dir)
    save_formatted_log(output)
    config = _read_json(output / 'config.json')
    summary = _read_json(output / 'summary.json')
    split = _read_json(output / 'split.json')
    if not config or not summary or not split:
        raise RuntimeError('Missing config/summary/split; refusing an incomplete run')

    # Keep the original split IDs and final test result in the three retained files.
    config['split_manifest'] = split
    config['split_fingerprints'] = {
        key: _sha16(split[key]) for key in ('train_ids', 'valid_ids', 'test_ids')}
    config['output_policy'] = 'three-files; intermediate files removed after completion'
    test_report = _read_json(output / 'test_metrics.json')
    if summary.get('test_evaluated') and test_report.get('status') == 'ok':
        summary['test_roc_auc'] = test_report['roc_auc']
        summary['test_evaluated_tasks'] = test_report.get('evaluated_tasks')
        summary['test_excluded_tasks'] = test_report.get('excluded_tasks', [])
    else:
        summary['test_roc_auc'] = None

    write_json(output / 'config.json', config)
    write_json(output / 'summary.json', summary)
    retained = {'config.json', 'summary.json', args.dataset + '.log'}
    for path in output.iterdir():
        if path.name not in retained:
            if not path.is_file():
                raise RuntimeError('Unexpected directory in run output: %s' % path)
            path.unlink()
    print('Run directory: ' + str(output.resolve()), flush=True)
    print('Artifacts: config.json, summary.json, %s.log' % args.dataset, flush=True)


if __name__ == '__main__':
    main(load_args())
