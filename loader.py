#!/usr/bin/env python

from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, InMemoryDataset
from rdkit import Chem
from rdkit.Chem import BRICS, AllChem
from util import sha256, write_json
from util import load_trusted_artifact

FEATURE_VERSION = "ase-mol"
allowable_features = {
    "possible_atomic_num_list": list(range(1, 119)),
    "possible_chirality_list": [Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW, Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.rdchem.ChiralType.CHI_OTHER],
    "possible_bonds": [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC, Chem.rdchem.BondType.DATIVE],
    "possible_bond_dirs": [Chem.rdchem.BondDir.NONE, Chem.rdchem.BondDir.ENDUPRIGHT,
                          Chem.rdchem.BondDir.ENDDOWNRIGHT],
}

def return_brics_leaf_structure(mol):
    if mol is None or mol.GetNumAtoms() == 0:
        raise ValueError("Molecule is empty")
    cuts = [tuple(pair) for pair, _ in BRICS.FindBRICSBonds(mol)]
    reduced = Chem.RWMol(mol)
    for u, v in cuts:
        reduced.RemoveBond(u, v)
    fragments = Chem.GetMolFrags(reduced.GetMol(), asMols=False, sanitizeFrags=False)
    covered = [i for f in fragments for i in f]
    if sorted(covered) != list(range(mol.GetNumAtoms())):
        raise ValueError("BRICS fragments do not cover all atoms")
    return {"substructure": {i: list(f) for i, f in enumerate(fragments)},
            "substructure_bond": [set(pair) for pair in cuts]}

def mol_to_graph_data_obj_simple(mol, smile=None):
    parts = return_brics_leaf_structure(mol)["substructure"]
    node_batch = torch.empty(mol.GetNumAtoms(), dtype=torch.long)
    for key, atom_ids in parts.items():
        node_batch[atom_ids] = key
    x = torch.tensor([[allowable_features["possible_atomic_num_list"].index(a.GetAtomicNum()),
                       allowable_features["possible_chirality_list"].index(a.GetChiralTag())]
                      for a in mol.GetAtoms()], dtype=torch.long)
    edges, attrs = [], []
    for bond in mol.GetBonds():
        u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        attr = [allowable_features["possible_bonds"].index(bond.GetBondType()),
                allowable_features["possible_bond_dirs"].index(bond.GetBondDir())]
        edges.extend([(u, v), (v, u)])
        attrs.extend([attr, attr])
    edge_index = torch.tensor(edges, dtype=torch.long).reshape(-1, 2).T.contiguous()
    edge_attr = torch.tensor(attrs, dtype=torch.long).reshape(-1, 2)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, node_batch=node_batch)
    if smile is not None:
        data.smiles = smile
    return data

class MoleculeDataset(InMemoryDataset):
    def __init__(self, root, dataset):
        self.dataset = dataset
        raw = Path(root) / "raw"
        files = sorted(p for p in raw.iterdir() if p.is_file() and (p.name.endswith(".csv") or p.name.endswith(".csv.gz")))
        if len(files) != 1:
            raise ValueError(str(raw) + " must contain exactly one raw CSV or CSV.GZ file")
        self._raw_name = files[0].name
        self.raw_sha256 = sha256(files[0])
        super().__init__(str(root))

        self.data, self.slices = load_trusted_artifact(self.processed_paths[0], map_location="cpu", tensor_only=False)

    @property
    def raw_file_names(self):
        return [self._raw_name]

    @property
    def processed_file_names(self):
        return [FEATURE_VERSION + "-" + self.raw_sha256 + ".pt"]

    def download(self):
        raise FileNotFoundError("Provide the original MoleculeNet CSV; automatic download is disabled")

    def process(self):
        loaders = {"tox21": _load_tox21_dataset, "bace": _load_bace_dataset,
                   "bbbp": _load_bbbp_dataset, "hiv": _load_hiv_dataset,
                   "clintox": _load_clintox_dataset, "muv": _load_muv_dataset,
                   "sider": _load_sider_dataset, "toxcast": _load_toxcast_dataset}
        result = loaders[self.dataset](self.raw_paths[0])
        smiles, mols, labels = result[0], result[1], result[-1]

        source_frame = pd.read_csv(self.raw_paths[0])
        smiles = source_frame["mol" if self.dataset == "bace" else "smiles"].tolist()
        records, excluded = [], []
        for i, (text, mol, target) in enumerate(zip(smiles, mols, labels)):
            if mol is None or mol.GetNumAtoms() == 0:
                excluded.append({"id": i, "reason": "invalid_or_empty_molecule"})
                continue
            if any(a.GetAtomicNum() == 0 for a in mol.GetAtoms()):
                excluded.append({"id": i, "reason": "dummy_atom"})
                continue
            target = np.asarray(target).reshape(-1)
            if not np.isin(target, [-1, 0, 1]).all():
                raise ValueError("Invalid label at original row " + str(i))
            record = mol_to_graph_data_obj_simple(mol, text)
            record.id = torch.tensor([i], dtype=torch.long)
            record.y = torch.tensor(target, dtype=torch.float32).unsqueeze(0)
            records.append(record)
        if not records:
            raise ValueError("No valid molecules remain after preprocessing")
        data, slices = self.collate(records)
        torch.save((data, slices), self.processed_paths[0])
        write_json(Path(self.processed_paths[0]).with_suffix(".json"),
                   {"raw_sha256": self.raw_sha256, "feature_version": FEATURE_VERSION,
                    "raw_rows": len(mols), "kept": len(records), "excluded": excluded})

def _load_tox21_dataset(input_path):
    """

    :param input_path:
    :return: list of smiles, list of rdkit mol obj, np.array containing the
    labels
    """
    input_df = pd.read_csv(input_path, sep=',')
    smiles_list = input_df['smiles']
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    tasks = ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD',
             'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53']
    labels = input_df[tasks]

    labels = labels.replace(0, -1)

    labels = labels.fillna(0)
    assert len(smiles_list) == len(rdkit_mol_objs_list)
    assert len(smiles_list) == len(labels)
    return smiles_list, rdkit_mol_objs_list, labels.values

def _load_hiv_dataset(input_path):
    """
    :param input_path:
    :return: list of smiles, list of rdkit mol obj, np.array containing the
    labels
    """
    input_df = pd.read_csv(input_path, sep=',')
    smiles_list = input_df['smiles']
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    labels = input_df['HIV_active']

    labels = labels.replace(0, -1)

    assert len(smiles_list) == len(rdkit_mol_objs_list)
    assert len(smiles_list) == len(labels)
    return smiles_list, rdkit_mol_objs_list, labels.values

def _load_bace_dataset(input_path):
    """

    :param input_path:
    :return: list of smiles, list of rdkit mol obj, np.array
    containing indices for each of the 3 folds, np.array containing the
    labels
    """
    input_df = pd.read_csv(input_path, sep=',')
    smiles_list = input_df['mol']
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    labels = input_df['Class']

    labels = labels.replace(0, -1)

    folds = input_df['Model']
    folds = folds.replace('Train', 0)
    folds = folds.replace('Valid', 1)
    folds = folds.replace('Test', 2)
    assert len(smiles_list) == len(rdkit_mol_objs_list)
    assert len(smiles_list) == len(labels)
    assert len(smiles_list) == len(folds)
    return smiles_list, rdkit_mol_objs_list, folds.values, labels.values

def _load_bbbp_dataset(input_path):
    """

    :param input_path:
    :return: list of smiles, list of rdkit mol obj, np.array containing the
    labels
    """
    input_df = pd.read_csv(input_path, sep=',')
    smiles_list = input_df['smiles']
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]

    preprocessed_rdkit_mol_objs_list = [m if m != None else None for m in rdkit_mol_objs_list]

    preprocessed_smiles_list = [smiles_list[m_i] if m != None else None for m_i, m in enumerate(rdkit_mol_objs_list)]
    labels = input_df['p_np']

    labels = labels.replace(0, -1)

    assert len(smiles_list) == len(preprocessed_rdkit_mol_objs_list)
    assert len(smiles_list) == len(preprocessed_smiles_list)
    assert len(smiles_list) == len(labels)
    return preprocessed_smiles_list, preprocessed_rdkit_mol_objs_list, labels.values

def _load_clintox_dataset(input_path):
    """

    :param input_path:
    :return: list of smiles, list of rdkit mol obj, np.array containing the
    labels
    """
    input_df = pd.read_csv(input_path, sep=',')
    smiles_list = input_df['smiles']
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]

    preprocessed_rdkit_mol_objs_list = [m if m != None else None for m in rdkit_mol_objs_list]
    preprocessed_smiles_list = [AllChem.MolToSmiles(m) if m != None else None for m in preprocessed_rdkit_mol_objs_list]
    tasks = ['FDA_APPROVED', 'CT_TOX']
    labels = input_df[tasks]

    labels = labels.replace(0, -1)

    labels = labels.fillna(0)
    assert len(smiles_list) == len(preprocessed_rdkit_mol_objs_list)
    assert len(smiles_list) == len(preprocessed_smiles_list)
    assert len(smiles_list) == len(labels)
    return preprocessed_smiles_list, preprocessed_rdkit_mol_objs_list, labels.values

def _load_muv_dataset(input_path):
    """

    :param input_path:
    :return: list of smiles, list of rdkit mol obj, np.array containing the
    labels
    """
    input_df = pd.read_csv(input_path, sep=',')
    smiles_list = input_df['smiles']
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    tasks = ['MUV-466', 'MUV-548', 'MUV-600', 'MUV-644', 'MUV-652', 'MUV-689',
             'MUV-692', 'MUV-712', 'MUV-713', 'MUV-733', 'MUV-737', 'MUV-810',
             'MUV-832', 'MUV-846', 'MUV-852', 'MUV-858', 'MUV-859']
    labels = input_df[tasks]

    labels = labels.replace(0, -1)

    labels = labels.fillna(0)
    assert len(smiles_list) == len(rdkit_mol_objs_list)
    assert len(smiles_list) == len(labels)
    return smiles_list, rdkit_mol_objs_list, labels.values

def _load_sider_dataset(input_path):
    """

    :param input_path:
    :return: list of smiles, list of rdkit mol obj, np.array containing the
    labels
    """
    input_df = pd.read_csv(input_path, sep=',')
    smiles_list = input_df['smiles']
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    tasks = ['Hepatobiliary disorders',
             'Metabolism and nutrition disorders', 'Product issues', 'Eye disorders',
             'Investigations', 'Musculoskeletal and connective tissue disorders',
             'Gastrointestinal disorders', 'Social circumstances',
             'Immune system disorders', 'Reproductive system and breast disorders',
             'Neoplasms benign, malignant and unspecified (incl cysts and polyps)',
             'General disorders and administration site conditions',
             'Endocrine disorders', 'Surgical and medical procedures',
             'Vascular disorders', 'Blood and lymphatic system disorders',
             'Skin and subcutaneous tissue disorders',
             'Congenital, familial and genetic disorders',
             'Infections and infestations',
             'Respiratory, thoracic and mediastinal disorders',
             'Psychiatric disorders', 'Renal and urinary disorders',
             'Pregnancy, puerperium and perinatal conditions',
             'Ear and labyrinth disorders', 'Cardiac disorders',
             'Nervous system disorders',
             'Injury, poisoning and procedural complications']
    labels = input_df[tasks]

    labels = labels.replace(0, -1)
    labels = labels.fillna(0)
    assert len(smiles_list) == len(rdkit_mol_objs_list)
    assert len(smiles_list) == len(labels)
    return smiles_list, rdkit_mol_objs_list, labels.values

def _load_toxcast_dataset(input_path):
    """

    :param input_path:
    :return: list of smiles, list of rdkit mol obj, np.array containing the
    labels
    """

    input_df = pd.read_csv(input_path, sep=',')
    smiles_list = input_df['smiles']
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]

    preprocessed_rdkit_mol_objs_list = [m if m != None else None for m in rdkit_mol_objs_list]
    preprocessed_smiles_list = [AllChem.MolToSmiles(m) if m != None else None for m in preprocessed_rdkit_mol_objs_list]
    tasks = list(input_df.columns)[1:]
    labels = input_df[tasks]

    labels = labels.replace(0, -1)

    labels = labels.fillna(0)
    assert len(smiles_list) == len(preprocessed_rdkit_mol_objs_list)
    assert len(smiles_list) == len(preprocessed_smiles_list)
    assert len(smiles_list) == len(labels)
    return preprocessed_smiles_list, preprocessed_rdkit_mol_objs_list, labels.values
