#!/usr/bin/env python

from collections import defaultdict
from pathlib import Path
import json
import random
import torch
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from configs import SPLIT, SPLIT_SEED
from util import write_json


def generate_scaffold(smiles, include_chirality=False):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Cannot compute scaffold: " + str(smiles))
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)


def scaffold_indices(smiles, seed=SPLIT_SEED):
    """Shuffle large and small scaffold groups separately, without using labels."""
    groups = defaultdict(list)
    for i, text in enumerate(smiles):
        groups[generate_scaffold(text)].append(i)
    ordered = list(groups.values())
    n = len(smiles)
    big = [g for g in ordered if len(g) > 0.1 * n / 2]
    small = [g for g in ordered if len(g) <= 0.1 * n / 2]
    rng = random.Random(seed)
    rng.shuffle(big)
    rng.shuffle(small)
    ordered = big + small
    train, valid, test = [], [], []
    for group in ordered:
        if len(train) + len(group) <= 0.8 * n:
            train.extend(group)
        elif len(valid) + len(group) <= 0.1 * n:
            valid.extend(group)
        else:
            test.extend(group)
    return train, valid, test


def data_split(args, dataset):
    if args.split != SPLIT or args.split_seed != SPLIT_SEED:
        raise ValueError("Only the fixed balanced scaffold algorithm (CLI: scaffold) is supported")
    smiles = [d.smiles for d in dataset]
    ids = [int(d.id.item()) for d in dataset]
    if len(ids) != len(set(ids)):
        raise ValueError("Dataset IDs must be unique")
    if args.split_file:
        manifest = json.loads(Path(args.split_file).read_text(encoding="utf-8"))
        if manifest["raw_sha256"] != dataset.raw_sha256:
            raise ValueError("Split manifest has a different raw-data fingerprint")
        if manifest["split"] != SPLIT or manifest["split_seed"] != SPLIT_SEED:
            raise ValueError("Split manifest does not match fixed scaffold policy")
        if manifest.get("scaffold_variant", "balanced") != "balanced":
            raise ValueError("Split manifest uses a different scaffold algorithm")
        lookup = {item_id: i for i, item_id in enumerate(ids)}
        parts = [[lookup[v] for v in manifest[k]] for k in ("train_ids", "valid_ids", "test_ids")]
    else:
        parts = scaffold_indices(smiles, SPLIT_SEED)
        manifest = {"version": "ase-reconstruction-v2", "raw_sha256": dataset.raw_sha256,
                    "split": SPLIT, "split_seed": SPLIT_SEED, "rule": "chemprop-v1.6.1",
                    "scaffold_variant": "balanced"}
        for key, indices in zip(("train_ids", "valid_ids", "test_ids"), parts):
            manifest[key] = [ids[i] for i in indices]
    flat = [i for p in parts for i in p]
    if len(flat) != len(dataset) or set(flat) != set(range(len(dataset))):
        raise ValueError("Split must be a non-overlapping, exhaustive partition")
    if any(not part for part in parts):
        raise ValueError("Split produced an empty subset")
    sets = [{generate_scaffold(smiles[i]) for i in p} for p in parts]
    if any(sets[i] & sets[j] for i in range(3) for j in range(i)):
        raise ValueError("Scaffold overlap across partitions")
    output = Path(args.output_dir) / "split.json"
    if not output.exists():
        write_json(output, manifest)
    elif json.loads(output.read_text(encoding="utf-8")) != manifest:
        raise ValueError("Existing split manifest differs from current split")
    return tuple(dataset[torch.tensor(p, dtype=torch.long)] for p in parts)
