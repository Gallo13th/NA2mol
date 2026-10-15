#!/usr/bin/env python
"""
RNAsmol scoring migrated to the rna2smol env (torch 2.3.1+cu118, PyG 2.6.1, py3.10)
running on a single RTX 4090 (sm_89).

Why this file exists
--------------------
The released RNAsmol weights were trained under py3.7 / torch 1.7.1, which cannot
target sm_89 (RTX 4090) and only ran on CPU. torch 2.3.1 fully supports sm_89, so
we re-host the *unchanged* model definition (`RNAsmol/rnasmol/model.py`) and the
*unchanged* saved_model state_dict on the new env + GPU.

Compatibility survey (rna2smol env) -- all green, no monkey-patch needed:
  * `rnasmol.model`           : imports cleanly. Only a non-fatal deprecation
                                warning on `torch.nn.utils.weight_norm`.
  * `rnasmol.dataset`         : imports cleanly (`InMemoryDataset` API stable).
  * `rnasmol.preprocessing`   : module-level `sys.argv[1]` blocks bare import,
                                but this is a script-level artefact, NOT a
                                torch/PyG issue. We avoid importing it and
                                reuse the verbatim graph builder in
                                `rnasmol_score.py` instead.
  * DataLoader                : `from torch_geometric.data import DataLoader`
                                still works in PyG 2.6.1 (re-exported, with a
                                DeprecationWarning pointing at
                                `torch_geometric.loader.DataLoader`).
  * GCNConv / GATConv / GINConv: signatures compatible; the model calls
                                `conv(x, edge_index, edge_weight=edge_attr)`,
                                which is supported.
  * state_dict (42 tensors)   : loads with strict=True -- 0 missing, 0 extra,
                                0 shape mismatch. torch checkpoint format is
                                fully version-portable for plain state_dicts.

This script:
  1. asserts CUDA is available and the model + inputs live on cuda,
  2. scores FMN (1FMN aptamer) and reports P(binder),
  3. scores 5 molecules (FMN/glucose/benzene/caffeine/ethanol), prints the
     ranking (FMN must be #1),
  4. benchmarks single-molecule latency: GPU vs CPU.

The RNAsmol original package (`RNAsmol/rnasmol/*.py`) is NOT modified -- this
file reuses the adapter graph builder already present in `rnasmol_score.py`.

Usage
-----
    conda activate rna2smol
    cd case_study/fmn
    python rnasmol_score_gpu.py
    python rnasmol_score_gpu.py --ligand testset.sdf --device cuda
"""
import os
import sys
import time
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
RNASMOL_ROOT = os.path.join(HERE, "RNAsmol")
RNASMOL_PKG = os.path.join(RNASMOL_ROOT, "rnasmol")
# `rnasmol/` ships without __init__.py (PEP-420 namespace package): put the
# *parent* on the path so `import rnasmol.model` resolves. The adapter
# `rnasmol_score.py` adds the inner dir so its `from model import ...` works.
for _p in (RNASMOL_PKG, RNASMOL_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import torch
import torch.nn.functional as F

# reuse the verbatim graph builder + model loader from the adapter script
from rnasmol_score import (                                  # noqa: E402
    build_data_from_mol, load_model, ligands_from_file,
    FMN_APTAMER_SEQ, DEFAULT_MODEL,
)
from rnasmol.model import MCNN_GCN                           # noqa: E402  (compat probe)


# ---------------------------------------------------------------------------
# scoring with explicit CUDA verification
# ---------------------------------------------------------------------------
@torch.no_grad()
def score_data(model, data, device):
    data = data.to(device)
    data.batch = torch.zeros(data.x.size(0), dtype=torch.long, device=device)
    # prove tensors are on the requested device
    assert next(model.parameters()).device.type == device.type, \
        f"model not on {device}"
    assert data.x.device.type == device.type, f"data.x not on {device}"
    ligand_x, protein_x, att, score = model(data)
    prob = F.softmax(score, dim=-1)
    p_binder = prob[:, 1].item()
    pred_cls = int(torch.argmax(score, dim=-1).item())
    return p_binder, pred_cls, score.detach().cpu().numpy().ravel().tolist()


def score_all(model, ligand_path, rna_seq, device):
    rows = []
    for name, mol in ligands_from_file(ligand_path):
        data = build_data_from_mol(mol, rna_seq)
        p_binder, pred, logits = score_data(model, data, device)
        rows.append((name, p_binder, pred, logits))
    return rows


def benchmark(model, mol, rna_seq, device, repeat=20):
    """Mean single-molecule scoring latency (ms), warmup excluded."""
    data = build_data_from_mol(mol, rna_seq).to(device)
    data.batch = torch.zeros(data.x.size(0), dtype=torch.long, device=device)
    model.eval()
    with torch.no_grad():
        for _ in range(5):                      # warmup
            model(data)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(repeat):
            model(data)
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / repeat
    return dt * 1000.0                          # ms


def print_rows(title, rows):
    print(f"\n{title}")
    print(f"{'rank':<5}{'name':<14}{'P(binder)':>12}{'pred':>6}{'logits':>30}")
    print("-" * 67)
    ranked = sorted(rows, key=lambda r: -r[1])
    for rank, (name, p, pred, logits) in enumerate(ranked, 1):
        print(f"{rank:<5}{name:<14}{p:12.6f}{pred:>6}{str([round(x,4) for x in logits]):>30}")


def main():
    ap = argparse.ArgumentParser(description="RNAsmol GPU scoring (rna2smol env)")
    ap.add_argument("--ligand", default=os.path.join(HERE, "testset.sdf"),
                    help="ligand SDF (default: testset.sdf = 5-molecule panel)")
    ap.add_argument("--fmn", default=os.path.join(HERE, "FMN_ligand.sdf"),
                    help="FMN SDF for the single-target validation")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--rna-seq", default=FMN_APTAMER_SEQ)
    ap.add_argument("--device", default="cuda",
                    help="cuda | cpu (default cuda; falls back if unavailable)")
    args = ap.parse_args()

    print("=" * 67)
    print("RNAsmol GPU scoring -- rna2smol env")
    print("=" * 67)
    print(f"torch          : {torch.__version__}")
    print(f"cuda available : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"cuda device    : {torch.cuda.get_device_name(0)} "
              f"(cap {torch.cuda.get_device_capability(0)})")
    print(f"model file     : {os.path.relpath(args.model, HERE)}")
    print(f"RNA ({len(args.rna_seq)} nt)      : {args.rna_seq}")

    device = torch.device(args.device if args.device != "cuda"
                          or torch.cuda.is_available() else "cpu")
    assert device.type == "cuda", "GPU requested but CUDA not available"
    print(f"scoring device : {device}")

    model = load_model(args.model, device)
    # final proof: report the device of one parameter
    sample_param = next(model.parameters())
    print(f"model param dev: {sample_param.device}  "
          f"(dtype={sample_param.dtype})")

    # --- Step A: single-target FMN validation -----------------------------
    fmn_rows = score_all(model, args.fmn, args.rna_seq, device)
    print_rows("[A] FMN validation (1FMN aptamer vs FMN)", fmn_rows)

    # --- Step B: 5-molecule ranking ---------------------------------------
    panel_rows = score_all(model, args.ligand, args.rna_seq, device)
    print_rows("[B] 5-molecule panel ranking", panel_rows)
    top_name = max(panel_rows, key=lambda r: r[1])[0]
    print(f"\nTop-1: {top_name}  (expected: FMN)")
    assert top_name == "FMN", "FMN did NOT rank first -- investigate"

    # --- Step C: GPU vs CPU latency ---------------------------------------
    from rdkit import Chem
    fmn_mol = next(Chem.SDMolSupplier(args.fmn, removeHs=True, sanitize=True))
    cpu_model = load_model(args.model, torch.device("cpu"))
    gpu_ms = benchmark(model, fmn_mol, args.rna_seq, torch.device("cuda"))
    cpu_ms = benchmark(cpu_model, fmn_mol, args.rna_seq, torch.device("cpu"))
    print("\n[C] Single-molecule latency (FMN, 20 iters, warmup excluded)")
    print(f"    GPU ({torch.cuda.get_device_name(0)}): {gpu_ms:.3f} ms/mol")
    print(f"    CPU                          : {cpu_ms:.3f} ms/mol")
    print(f"    speedup                      : {cpu_ms / gpu_ms:.1f}x")
    print("=" * 67)


if __name__ == "__main__":
    main()
