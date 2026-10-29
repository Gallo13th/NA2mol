#!/usr/bin/env python
"""6-version RNAsmol ensemble scoring for top100(SAS-optimal) FMN binders.

Inputs
------
- final_binders_top100_sas.sdf : 100 molecules (docked + rdock + SAS + QED + MW +
                                 single-version rnasmol from pdb_bothaug_molperturbation)
- RNAsmol/saved_model/*.pt     : 6 model variants
  pdb_*  : pdb_bothaug_{mol,net,rna}perturbation
  robin_*: robin_bothaug_{mol,net,rna}perturbation
- FMN_ligand.sdf               : native cofactor reference

Outputs
-------
- final_binders_top100_ensemble.sdf : 100 mols + docked + rdock + 6-version rnasmol
                                      + ensemble_P + ensemble_logit + multi_gate + SAS/QED/MW
- final_binders_top100_ensemble.tsv : same, tabular (sorted, with rank)
- fmn_ensemble_reference.json       : FMN scores for the 6 versions + ensemble
- ensemble_scoring_summary.json     : per-version P distribution + ensemble stats

GPU (rna2smol env). Reuses rnasmol_score.py graph builder verbatim.
"""
import os
import sys
import json
import time
import warnings

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
RNASMOL_ROOT = os.path.join(HERE, "RNAsmol")
RNASMOL_PKG = os.path.join(RNASMOL_ROOT, "rnasmol")
for _p in (RNASMOL_PKG, RNASMOL_ROOT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import torch
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

from rnasmol_score import (                                       # noqa: E402
    build_data_from_mol, load_model, FMN_APTAMER_SEQ,
)
from rnasmol_score_gpu import score_data                          # noqa: E402

# ---------------------------------------------------------------------------
IN_SDF = os.path.join(HERE, "final_binders_top100_sas.sdf")
FMN_SDF = os.path.join(HERE, "FMN_ligand.sdf")
OUT_SDF = os.path.join(HERE, "final_binders_top100_ensemble.sdf")
OUT_TSV = os.path.join(HERE, "final_binders_top100_ensemble.tsv")
OUT_FMN_JSON = os.path.join(HERE, "fmn_ensemble_reference.json")
OUT_SUMMARY = os.path.join(HERE, "ensemble_scoring_summary.json")

# 6 model variants — short_key used for SDF prop names
MODELS = [
    ("pdb_mol",  "pdb_bothaug_molperturbation.pt"),
    ("pdb_net",  "pdb_bothaug_netperturbation.pt"),
    ("pdb_rna",  "pdb_bothaug_rnaperturbation.pt"),
    ("rob_mol",  "robin_bothaug_molperturbation.pt"),
    ("rob_net",  "robin_bothaug_netperturbation.pt"),
    ("rob_rna",  "robin_bothaug_rnaperturbation.pt"),
]

POSITIVE_PROB_THRESHOLD = 0.5


def score_one_model(model, mols_data, device):
    """Return list of (P_binder, logit_diff) for each pre-built data object."""
    out = []
    with torch.no_grad():
        for data in mols_data:
            p, _, logits = score_data(model, data, device)
            # logit = logit_binder - logit_non_binder (signed strength)
            logit_diff = float(logits[1] - logits[0])
            out.append((p, logit_diff))
    return out


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert device.type == "cuda", "GPU required"
    print(f"[ensemble] device = {device} ({torch.cuda.get_device_name(0)})")
    print(f"[ensemble] RNA ({len(FMN_APTAMER_SEQ)} nt) = {FMN_APTAMER_SEQ}")

    # --- load molecules ----------------------------------------------------
    suppl = Chem.SDMolSupplier(IN_SDF, removeHs=False, sanitize=False)
    raw = [m for m in suppl if m is not None]
    # we need sanitized mols (no H) for the graph builder
    suppl_s = Chem.SDMolSupplier(IN_SDF, removeHs=True, sanitize=True)
    mols = list(suppl_s)
    assert len(mols) == 100, f"expected 100 mols, got {len(mols)}"
    print(f"[ensemble] loaded {len(mols)} molecules (+FMN)")

    fmn_mol = next(Chem.SDMolSupplier(FMN_SDF, removeHs=True, sanitize=True))

    # --- pre-build graphs (do this once — graph build is the slow part) ----
    print("[ensemble] building graphs (100 mols + FMN) ...")
    t0 = time.time()
    mols_data = [build_data_from_mol(m, FMN_APTAMER_SEQ) for m in mols]
    fmn_data = build_data_from_mol(fmn_mol, FMN_APTAMER_SEQ)
    print(f"          done in {time.time() - t0:.2f}s")

    # --- score with each model --------------------------------------------
    # per_model_scores[key] = list[(P, logit)] aligned with mols
    per_model_scores = {}
    fmn_scores = {}
    for key, fname in MODELS:
        path = os.path.join(RNASMOL_ROOT, "saved_model", fname)
        print(f"[ensemble] scoring with {key} ({fname}) ...")
        t0 = time.time()
        model = load_model(path, device)
        torch.cuda.synchronize()
        # warmup
        _ = score_data(model, fmn_data, device)
        torch.cuda.synchronize()
        t1 = time.time()
        per_model_scores[key] = score_one_model(model, mols_data, device)
        torch.cuda.synchronize()
        # FMN
        p, _, logits = score_data(model, fmn_data, device)
        fmn_scores[key] = (p, float(logits[1] - logits[0]))
        print(f"          100 mols in {time.time() - t1:.2f}s "
              f"| P range {min(s[0] for s in per_model_scores[key]):.4f}"
              f"..{max(s[0] for s in per_model_scores[key]):.4f}"
              f" | FMN P={p:.4f}")
        del model
        torch.cuda.empty_cache()

    # --- ensemble aggregation ---------------------------------------------
    print("[ensemble] aggregating (mean P, vote count, mean logit) ...")
    ensemble_records = []  # list aligned with mols
    for i in range(len(mols)):
        ps = [per_model_scores[k][i][0] for k, _ in MODELS]
        ls = [per_model_scores[k][i][1] for k, _ in MODELS]
        ensemble_p = float(np.mean(ps))
        ensemble_logit = float(np.mean(ls))
        multi_gate = int(sum(1 for p in ps if p > POSITIVE_PROB_THRESHOLD))
        ensemble_records.append({
            "per_p": dict(zip([k for k, _ in MODELS], ps)),
            "per_logit": dict(zip([k for k, _ in MODELS], ls)),
            "ensemble_P": ensemble_p,
            "ensemble_logit": ensemble_logit,
            "multi_gate": multi_gate,
        })

    fmn_ps = [fmn_scores[k][0] for k, _ in MODELS]
    fmn_ls = [fmn_scores[k][1] for k, _ in MODELS]
    fmn_record = {
        "per_p": dict(zip([k for k, _ in MODELS], fmn_ps)),
        "per_logit": dict(zip([k for k, _ in MODELS], fmn_ls)),
        "ensemble_P": float(np.mean(fmn_ps)),
        "ensemble_logit": float(np.mean(fmn_ls)),
        "multi_gate": int(sum(1 for p in fmn_ps if p > POSITIVE_PROB_THRESHOLD)),
    }

    # --- write back properties to molecules -------------------------------
    print("[ensemble] writing SDF with full attributes ...")
    # Use raw (with Hs + docked coords + existing props) as base.
    assert len(raw) == len(ensemble_records)
    for m, rec in zip(raw, ensemble_records):
        # clear stale single-version props to avoid confusion (optional, keep)
        # per-version P
        for k, _ in MODELS:
            m.SetProp(f"rnasmol_p_{k}", f"{rec['per_p'][k]:.6f}")
            m.SetProp(f"rnasmol_logit_{k}", f"{rec['per_logit'][k]:.4f}")
        m.SetProp("rnasmol_ensemble_P", f"{rec['ensemble_P']:.6f}")
        m.SetProp("rnasmol_ensemble_logit", f"{rec['ensemble_logit']:.4f}")
        m.SetProp("rnasmol_multi_gate", f"{rec['multi_gate']}/6")

    # --- sort: multi_gate desc -> ensemble_P desc -> rdock_INTER asc ------
    def _inter(m):
        try:
            return float(m.GetProp("rdock_INTER"))
        except Exception:
            return 0.0

    def _gate(m):
        try:
            return int(m.GetProp("rnasmol_multi_gate").split("/")[0])
        except Exception:
            return 0

    def _ep(m):
        try:
            return float(m.GetProp("rnasmol_ensemble_P"))
        except Exception:
            return 0.0

    order = sorted(
        range(len(raw)),
        key=lambda i: (-_gate(raw[i]), -_ep(raw[i]), _inter(raw[i])),
    )
    sorted_mols = [raw[i] for i in order]
    sorted_recs = [ensemble_records[i] for i in order]

    # assign rank as a prop
    for r, (m, rec) in enumerate(zip(sorted_mols, sorted_recs), 1):
        m.SetProp("ensemble_rank", str(r))
        m.SetProp("rnasmol_ensemble_P", f"{rec['ensemble_P']:.6f}")  # rewrite to be safe

    writer = Chem.SDWriter(OUT_SDF)
    for m in sorted_mols:
        writer.write(m)
    writer.close()
    print(f"[ensemble] wrote {OUT_SDF}")

    # --- TSV ---------------------------------------------------------------
    print("[ensemble] writing TSV ...")
    header = [
        "rank", "gen_idx", "rdock_INTER", "rdock_SCORE",
        "rnasmol_ensemble_P", "rnasmol_ensemble_logit", "rnasmol_multi_gate",
    ] + [f"rnasmol_p_{k}" for k, _ in MODELS] + [f"rnasmol_logit_{k}" for k, _ in MODELS] + [
        "SAS", "QED", "MW", "canonical_smiles",
    ]
    rows = []
    for r, (m, rec) in enumerate(zip(sorted_mols, sorted_recs), 1):
        d = m.GetPropsAsDict()
        row = [
            r,
            d.get("gen_idx", ""),
            f"{_inter(m):.4f}",
            f"{float(d.get('rdock_SCORE', 0.0)):.4f}",
            f"{rec['ensemble_P']:.6f}",
            f"{rec['ensemble_logit']:.4f}",
            f"{rec['multi_gate']}/6",
        ]
        for k, _ in MODELS:
            row.append(f"{rec['per_p'][k]:.6f}")
        for k, _ in MODELS:
            row.append(f"{rec['per_logit'][k]:.4f}")
        row.append(f"{float(d.get('SAS', 0.0)):.3f}")
        row.append(f"{float(d.get('QED', 0.0)):.4f}")
        row.append(f"{float(d.get('MW', 0.0)):.2f}")
        row.append(d.get("canonical_smiles", Chem.MolToSmiles(m)))
        rows.append(row)
    with open(OUT_TSV, "w") as fh:
        fh.write("\t".join(header) + "\n")
        for r in rows:
            fh.write("\t".join(str(x) for x in r) + "\n")
    print(f"[ensemble] wrote {OUT_TSV}")

    # --- FMN reference JSON -----------------------------------------------
    with open(OUT_FMN_JSON, "w") as fh:
        json.dump({
            "RNA_seq": FMN_APTAMER_SEQ,
            "models": [k for k, _ in MODELS],
            "fmn_smiles": Chem.MolToSmiles(fmn_mol),
            "fmn_scores": fmn_record,
        }, fh, indent=2)
    print(f"[ensemble] wrote {OUT_FMN_JSON}")

    # --- summary JSON ------------------------------------------------------
    def _stats(xs):
        xs = np.array(xs, dtype=float)
        return dict(
            min=float(xs.min()), median=float(np.median(xs)),
            mean=float(xs.mean()), max=float(xs.max()),
            p25=float(np.percentile(xs, 25)), p75=float(np.percentile(xs, 75)),
        )

    summary = {
        "n_mols": len(mols),
        "threshold_P": POSITIVE_PROB_THRESHOLD,
        "per_model_P_stats": {k: _stats([per_model_scores[k][i][0]
                                          for i in range(len(mols))])
                              for k, _ in MODELS},
        "ensemble_P_stats": _stats([r["ensemble_P"] for r in ensemble_records]),
        "ensemble_logit_stats": _stats([r["ensemble_logit"] for r in ensemble_records]),
        "multi_gate_distribution": {
            str(g): int(sum(1 for r in ensemble_records if r["multi_gate"] == g))
            for g in range(7)
        },
        "fmn": fmn_record,
    }
    with open(OUT_SUMMARY, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"[ensemble] wrote {OUT_SUMMARY}")

    # --- console report ----------------------------------------------------
    print("\n" + "=" * 70)
    print("Per-model P(binder) distribution over 100 mols")
    print("=" * 70)
    print(f"{'model':<10}{'min':>9}{'median':>9}{'mean':>9}{'max':>9}   FMN_P")
    for k, _ in MODELS:
        s = summary["per_model_P_stats"][k]
        print(f"{k:<10}{s['min']:>9.4f}{s['median']:>9.4f}"
              f"{s['mean']:>9.4f}{s['max']:>9.4f}   {fmn_record['per_p'][k]:.4f}")
    print("\nEnsemble P  min/median/max: "
          f"{summary['ensemble_P_stats']['min']:.4f} / "
          f"{summary['ensemble_P_stats']['median']:.4f} / "
          f"{summary['ensemble_P_stats']['max']:.4f}")
    print("Ensemble logit min/median/max: "
          f"{summary['ensemble_logit_stats']['min']:.3f} / "
          f"{summary['ensemble_logit_stats']['median']:.3f} / "
          f"{summary['ensemble_logit_stats']['max']:.3f}")
    print("multi_gate distribution:")
    for g in range(6, -1, -1):
        n = summary["multi_gate_distribution"].get(str(g), 0)
        bar = "#" * n
        print(f"  {g}/6: {n:>4}  {bar}")
    print(f"\nFMN ensemble_P = {fmn_record['ensemble_P']:.4f}  "
          f"multi_gate = {fmn_record['multi_gate']}/6  "
          f"ensemble_logit = {fmn_record['ensemble_logit']:.3f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
