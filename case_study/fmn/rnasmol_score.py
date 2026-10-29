#!/usr/bin/env python
"""
RNAsmol scoring interface for the 1FMN / FMN case study.

RNAsmol is a *sequence-based* RNA-ligand binder classifier:
  - RNA  : the aptamer nucleotide sequence (A/C/G/U), encoded as ints,
           padded/truncated to 500 residues (CNN "protein" branch).
  - Ligand: a 2D molecular graph built from SMILES (87-feature atoms,
            GCN branch with Personalized-PageRank edge weights).
The model (MCNN_GCN) outputs 2 logits (class 0 = non-binder, 1 = binder).
We report P(binder) = softmax(logits)[1].

The graph construction below is copied VERBATIM from RNAsmol/rnasmol/preprocessing.py
(including the original adj_matrix behaviour) so that the input distribution
matches what the released saved_model weights were trained on.

Usage
-----
    conda activate rnasmol
    cd case_study/fmn
    python rnasmol_score.py --ligand FMN_ligand.sdf
    python rnasmol_score.py --ligand generated.sdf --model RNAsmol/saved_model/pdb_bothaug_molperturbation.pt
    # batch: a multi-record SDF -> prints one score per molecule
"""
import os
import sys
import argparse

# --- RNAsmol code on path (model.py) -------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
RNASMOL_ROOT = os.path.join(HERE, "RNAsmol")
RNASMOL_PKG = os.path.join(RNASMOL_ROOT, "rnasmol")
if RNASMOL_PKG not in sys.path:
    sys.path.insert(0, RNASMOL_PKG)

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from rdkit import Chem
import networkx as nx

from model import MCNN_GCN  # noqa: E402

# Default RNA target = the 1FMN FMN-binding aptamer (35 nt, MODEL 1)
FMN_APTAMER_SEQ = "GGCGUGUAGGAUAUGCUUCGGCAGAAGGACACGCC"
DEFAULT_MODEL = os.path.join(RNASMOL_ROOT, "saved_model",
                             "pdb_bothaug_molperturbation.pt")

# =========================================================================
# Graph construction -- copied verbatim from preprocessing.py
# (do not "fix" the adj_matrix loop: the saved_model was trained with it)
# =========================================================================
VOCAB_PROTEIN = {"A": 1, "C": 2, "G": 3, "T": 4, "U": 4, "a": 5, "c": 6,
                 "g": 7, "t": 8, "u": 8, "D": 9, "d": 9, "B": 9, "b": 9,
                 "F": 9, "f": 9, "I": 9, "i": 9, "H": 9, "h": 9, "K": 9,
                 "k": 9, "M": 9, "m": 9, "L": 9, "l": 9, "O": 9, "o": 9,
                 "N": 9, "n": 9, "Q": 9, "q": 9, "P": 9, "p": 9, "S": 9,
                 "s": 9, "R": 9, "r": 9, "E": 9, "e": 9, "W": 9, "w": 9,
                 "V": 9, "v": 9, "Y": 9, "y": 9, "X": 9, "x": 9, "Z": 9,
                 "z": 9}


def seqs2int(target):
    # unknown nucleotides -> 9 (matches training vocab)
    return [VOCAB_PROTEIN.get(s, 9) for s in target]


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def atom_features(atom):
    encoding = one_of_k_encoding_unk(atom.GetSymbol(),
        ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca',
         'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag',
         'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni',
         'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'Unknown'])
    encoding += one_of_k_encoding(atom.GetDegree(),
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) + one_of_k_encoding_unk(
        atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    encoding += one_of_k_encoding_unk(atom.GetImplicitValence(),
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    encoding += one_of_k_encoding_unk(atom.GetHybridization(), [
        Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2, 'other'])
    encoding += [atom.GetIsAromatic()]
    try:
        encoding += one_of_k_encoding_unk(atom.GetProp('_CIPCode'),
            ['R', 'S']) + [atom.HasProp('_ChiralityPossible')]
    except Exception:
        encoding += [0, 0] + [atom.HasProp('_ChiralityPossible')]
    return np.array(encoding)


def mol_to_graph(mol):
    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append(feature / np.sum(feature))
    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    if len(edges) == 0:
        return features, [[0, 0]]
    g = nx.Graph(edges).to_directed()
    edge_index = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])
    adj_matrix = np.zeros(shape=(len(g.nodes), len(g.nodes)))
    for i, j in zip(edge_index[0], edge_index[1]):
        adj_matrix[i - 1, j - 1] = 1
    return features, edge_index, adj_matrix


def get_ppr_matrix(adj_matrix, alpha=0.1):
    num_nodes = adj_matrix.shape[0]
    A_tilde = adj_matrix + np.eye(num_nodes)
    D_tilde = np.diag(1 / np.sqrt(A_tilde.sum(axis=1)))
    H = D_tilde @ A_tilde @ D_tilde
    return alpha * np.linalg.inv(np.eye(num_nodes) - (1 - alpha) * H)


def get_top_k_matrix(A, k=5):
    num_nodes = A.shape[0]
    row_idx = np.arange(num_nodes)
    A[A.argsort(axis=0)[:num_nodes - k], row_idx] = 0.
    norm = A.sum(axis=0)
    norm[norm <= 0] = 1
    return A / norm


def build_data_from_mol(mol, rna_seq, alpha=0.1, k=5, target_len=500):
    """Build a torch_geometric Data object matching RNAsmol's preprocessing."""
    x_feats, edge_index_raw, adj_matrix = mol_to_graph(mol)
    ppr_matrix = get_ppr_matrix(adj_matrix, alpha=alpha)
    ppr_matrix = get_top_k_matrix(ppr_matrix, k=k)

    target = seqs2int(rna_seq)
    if len(target) < target_len:
        target = np.pad(target, (0, target_len - len(target)))
    else:
        target = target[:target_len]

    edges_i, edges_j, edge_attr = [], [], []
    for i in range(ppr_matrix.shape[0]):
        for j in np.where(ppr_matrix[i] > 0)[0]:
            edges_i.append(i)
            edges_j.append(j)
            edge_attr.append(ppr_matrix[i, j])
    edge_index = [edges_i, edges_j]

    return Data(
        x=torch.FloatTensor(np.array(x_feats)),
        edge_index=torch.LongTensor(edge_index),
        edge_attr=torch.FloatTensor(edge_attr),
        y=torch.FloatTensor([0]),
        target=torch.LongTensor([target]),
    )


def load_model(model_path, device):
    model = MCNN_GCN(3, 25 + 1, embedding_size=96, filter_num=32,
                     out_dim=2, ban_heads=2).to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


@torch.no_grad()
def score_data(model, data, device):
    data = data.to(device)
    data.batch = torch.zeros(data.x.size(0), dtype=torch.long, device=device)
    ligand_x, protein_x, att, score = model(data)
    prob = F.softmax(score, dim=-1)
    p_binder = prob[:, 1].item()
    pred_cls = int(torch.argmax(score, dim=-1).item())
    return p_binder, pred_cls, score.cpu().numpy().ravel().tolist()


def ligands_from_file(path):
    """Yield (name, rdkit Mol) from a .sdf or .smi file."""
    if path.lower().endswith(".sdf"):
        suppl = Chem.SDMolSupplier(path, removeHs=True, sanitize=True)
        for i, m in enumerate(suppl):
            if m is None:
                print(f"[warn] SDF record {i} could not be parsed, skipped",
                      file=sys.stderr)
                continue
            name = m.GetProp("_Name") if m.HasProp("_Name") else f"mol{i}"
            yield name, m
    else:  # treat as smiles (.smi / .csv): first column = SMILES
        with open(path) as fh:
            for i, ln in enumerate(fh):
                ln = ln.strip()
                if not ln or ln.lower().startswith("smiles"):
                    continue
                smi = ln.split(",")[0].split()[0]
                m = Chem.MolFromSmiles(smi)
                if m is None:
                    print(f"[warn] bad SMILES on line {i}: {smi}", file=sys.stderr)
                    continue
                yield f"mol{i}", m


def main():
    ap = argparse.ArgumentParser(description="RNAsmol binder scoring (1FMN case)")
    ap.add_argument("--ligand", required=True,
                    help="ligand .sdf (3D or 2D) or .smi/.csv (SMILES col)")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="path to a RNAsmol saved_model .pt")
    ap.add_argument("--rna-seq", default=FMN_APTAMER_SEQ,
                    help="RNA aptamer sequence (default: 1FMN aptamer, 35 nt)")
    ap.add_argument("--device", default="auto",
                    help="auto|cpu|cuda:N (default auto: first CUDA GPU)")
    args = ap.parse_args()

    if args.device == "auto":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    elif args.device.startswith("cuda"):
        idx = 0 if args.device == "cuda" else int(args.device.split(":")[1])
        device = torch.device(f"cuda:{idx}")
    else:
        device = torch.device("cpu")
    print(f"[rnasmol] device = {device}")
    print(f"[rnasmol] model  = {args.model}")
    print(f"[rnasmol] RNA seq ({len(args.rna_seq)} nt) = {args.rna_seq}")

    model = load_model(args.model, device)

    print(f"\n{'name':<20} {'P(binder)':>10} {'pred':>5} {'logits':>20}")
    print("-" * 60)
    for name, mol in ligands_from_file(args.ligand):
        data = build_data_from_mol(mol, args.rna_seq)
        p_binder, pred, logits = score_data(model, data, device)
        print(f"{name:<20} {p_binder:10.4f} {pred:>5} {str(logits):>20}")


if __name__ == "__main__":
    main()
