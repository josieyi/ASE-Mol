# ASE-Mol: Attribution-guided Substructure-aware Mixture-of-Experts for Molecular Property Prediction

Implementation for paper: **Attribution-guided Substructure-aware Mixture-of-Experts for Molecular Property Prediction**

This repository contains the full ASE-Mol model for binary molecular property prediction. The method uses BRICS fragments, attribution-guided positive/negative motif recognition, and task-specific positive/negative mixture-of-experts routing. Training has two stages: motif recognition and expert learning.

## Requirements

- Python == 3.8
- PyTorch == 1.8.1
- torch_geometric == 2.2.0
- torch_scatter == 2.0.8
- rdkit == 2024.3.2
- numpy == 1.24.3
- scikit-learn == 1.3.2

## Usage

The main running parameters are as follows:

| Parameter | Description | Default Value | Choices                                                |
| --- | --- | --- |--------------------------------------------------------|
| `dataset` | MoleculeNet classification dataset | `bace` | `bace, bbbp, sider, clintox, tox21, hiv, muv, toxcast` |
| `gnn_type` | Molecular graph encoder | `gcn` | `gcn, gin, gat, graphsage`                             |
| `seed` | Training random seed (fixed scaffold split seed is 0) | `0` | Nonnegative integer                                    |
| `device_no` | CUDA device index | `0` | Available device index                                 |
| `pre_lr` | Recognition-stage base learning rate | `1e-3` | Positive float                                         |
| `lr` | Expert-stage base learning rate | `1e-3` | Positive float                                         |
| `batch_size` | Molecular graphs per batch | `128` | `128,256,512`                                          |
| `num_experts` | Experts per task and motif branch | `3` | `1,3,5,7,9`                                            |
| `topk_rate` | Proportion of fragments selected per branch | `0.1` | `0.1,0.2,0.3`                                          |
| `alpha` | Margin regularization weight | `0.1` | `0.01,0.1,1,5`                                         |
| `beta` | Expert importance regularization weight | `0.1` | `0.01,0.1,1,5`                                      |

### Run molecular property prediction

```bash
python main.py \
  --dataset_dir ./data --dataset {dataset} --device_no {device_no} \
  --gnn_type {gnn_type} --split scaffold --seed {seed} \
  --pre_lr {pre_lr} --lr {lr} --batch_size {batch_size} \
  --num_experts {num_experts} --topk_rate {topk_rate} --alpha {alpha} --beta {beta} \
  --output_dir runs/{dataset}_{gnn_type}_seed{seed}
```

For example, to run **BBBP with GCN** using the main-experiment settings:

```bash
python main.py \
  --dataset_dir ./data --dataset bbbp --device_no 0 \
  --gnn_type gcn --split scaffold --seed 0 \
  --pre_lr 1e-3 --lr 1e-3 --batch_size 128 \
  --num_experts 3 --topk_rate 0.3 --alpha 1 --beta 0.1 \
  --output_dir runs/bbbp_gcn_seed0
```

