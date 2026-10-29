#!/usr/bin/env python
"""Targeted chemical-space audit for the RNA2smol manuscript.

The analysis answers three manuscript-level questions:
1. Are known RNA ligands and protein ligands mostly overlapping or separable?
2. Do generated molecules from RNA and protein benchmarks occupy different
   RNA-like or protein-like regions of ligand space?
3. How much chemical space do screened and strict survivor sets explore?

The script uses ECFP4 fingerprints and balanced BindingDB RNA/protein reference
sets to avoid the much larger protein reference set dominating nearest-neighbour
similarity estimates.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Crippen, Descriptors, QED
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parent.parent
NB = ROOT / "data" / "benchmark"
CACHE = NB / "output" / "cache_chemspace"
OUT = NB / "output" / "chemspace_targeted"
OUT.mkdir(parents=True, exist_ok=True)

RNG = np.random.default_rng(42)
N_BIAS_SAMPLE = 5000
N_DIVERSITY_SAMPLE = 2000

SMRT_METHOD_ORDER = [
    "RNA2smol (Only General Feat)",
    "RNA2smol (Baseline)",
    "MolCRAFT",
    "Pocket2Mol",
    "ED2Mol",
]

DUDE_METHOD_ORDER = [
    "RNA2smol",
    "MolCRAFT",
    "Pocket2Mol",
    "ed2mol",
]

METHOD_LABELS = {
    "RNA2smol (Only General Feat)": "RNA2smol",
    "RNA2smol": "RNA2smol",
    "RNA2smol (Baseline)": "RNA2smol (Baseline)",
    "MolCRAFT": "MolCRAFT",
    "Pocket2Mol": "Pocket2Mol",
    "ED2Mol": "ED2Mol",
    "ed2mol": "ED2Mol",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--controls-only",
        action="store_true",
        help="Only refresh benchmark-positive control ligand-space bias tables.",
    )
    return parser.parse_args()


def unique_csv_values(path: Path, column: str, chunksize: int = 100_000) -> list[str]:
    values: set[str] = set()
    for chunk in pd.read_csv(path, usecols=[column], chunksize=chunksize):
        values.update(chunk[column].dropna().astype(str).unique().tolist())
    return sorted(values)


def canonical_smiles(smi: str) -> str | None:
    mol = Chem.MolFromSmiles(str(smi))
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def is_druglike_smiles(smi: str) -> bool:
    mol = Chem.MolFromSmiles(str(smi))
    if mol is None:
        return False
    try:
        mw = Descriptors.MolWt(mol)
        logp = Crippen.MolLogP(mol)
        qed = QED.qed(mol)
    except Exception:
        return False
    return 150 <= mw <= 700 and -5 <= logp <= 10 and qed > 0.2


def fp_array_from_smiles(smiles: list[str]) -> tuple[list[str], np.ndarray]:
    valid_smiles: list[str] = []
    fps = []
    for smi in smiles:
        can = canonical_smiles(smi)
        if can is None:
            continue
        mol = Chem.MolFromSmiles(can)
        arr = np.zeros(2048, dtype=np.float32)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        DataStructs.ConvertToNumpyArray(fp, arr)
        valid_smiles.append(can)
        fps.append(arr)
    if not fps:
        return [], np.empty((0, 2048), dtype=np.float32)
    return valid_smiles, np.asarray(fps, dtype=np.float32)


def jaccard_distance_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    inter = a.astype(np.float32) @ b.astype(np.float32).T
    suma = a.sum(axis=1, keepdims=True)
    sumb = b.sum(axis=1, keepdims=True).T
    union = np.maximum(suma + sumb - inter, 1.0)
    return 1.0 - inter / union


def max_tanimoto_to_ref(query: np.ndarray, ref: np.ndarray, batch_size: int = 512) -> np.ndarray:
    out = np.zeros(len(query), dtype=np.float32)
    if len(query) == 0 or len(ref) == 0:
        return out
    for start in range(0, len(query), batch_size):
        d = jaccard_distance_matrix(query[start : start + batch_size], ref)
        out[start : start + batch_size] = 1.0 - d.min(axis=1)
    return out


def internal_diversity(fps: np.ndarray, sample_size: int = N_DIVERSITY_SAMPLE) -> float:
    if len(fps) < 2:
        return float("nan")
    n = min(len(fps), sample_size)
    idx = RNG.choice(len(fps), n, replace=False)
    d = jaccard_distance_matrix(fps[idx], fps[idx])
    tri = np.triu_indices(n, k=1)
    return float(d[tri].mean())


def murcko_scaffold(smi: str) -> str | None:
    mol = Chem.MolFromSmiles(str(smi))
    if mol is None:
        return None
    try:
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    except Exception:
        return None
    return scaffold or None


def ensure_sei(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "SEI" not in out.columns:
        out["SEI"] = -out["affinity"] / out["tpsa"].replace(0, np.nan)
    out["SEI"] = out["SEI"].replace([np.inf, -np.inf], np.nan).fillna(0)
    return out


def select_rows(df: pd.DataFrame, group: str) -> pd.DataFrame:
    if group == "broad":
        return df
    if group == "druglike":
        return df[(df["qed"] > 0.6) & (df["sa_score"] < 4)]
    if group == "screened":
        return df[
            (df["affinity"] < -8)
            & (df["qed"] > 0.5)
            & (df["sa_score"] < 5)
            & (df["SEI"] > 0.1)
        ]
    if group == "strict":
        if "strain" in df.columns:
            return df[
                (df["affinity"] < -9)
                & (df["qed"] > 0.6)
                & (df["sa_score"] < 4)
                & (df["SEI"] > 0.2)
                & (df["strain"] < 200)
            ]
        return df[
            (df["affinity"] < -9)
            & (df["rmsd"] < 1.0)
            & (df["qed"] > 0.6)
            & (df["sa_score"] < 4)
            & (df["SEI"] > 0.2)
        ]
    raise ValueError(group)


def load_reference() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rna_smiles = np.load(CACHE / "smiles_bkg_na.npy", allow_pickle=True)
    protein_smiles = np.load(CACHE / "smiles_bkg_protein.npy", allow_pickle=True)
    rna_fps = np.load(CACHE / "bkg_na_fps.npy")
    protein_fps = np.load(CACHE / "bkg_protein_fps.npy")
    return rna_smiles, rna_fps, protein_smiles, protein_fps


def reference_overlap_and_separability(rna_fps: np.ndarray, protein_fps: np.ndarray) -> dict:
    n = min(len(rna_fps), len(protein_fps))
    rna_idx = RNG.choice(len(rna_fps), n, replace=False)
    protein_idx = RNG.choice(len(protein_fps), n, replace=False)
    rna_bal = rna_fps[rna_idx]
    protein_bal = protein_fps[protein_idx]

    split = n // 2
    rna_q, rna_ref = rna_bal[:split], rna_bal[split:]
    protein_q, protein_ref = protein_bal[:split], protein_bal[split:]

    rna_to_rna = max_tanimoto_to_ref(rna_q, rna_ref)
    rna_to_protein = max_tanimoto_to_ref(rna_q, protein_ref)
    protein_to_protein = max_tanimoto_to_ref(protein_q, protein_ref)
    protein_to_rna = max_tanimoto_to_ref(protein_q, rna_ref)

    x = np.vstack([rna_bal, protein_bal])
    y = np.array([1] * len(rna_bal) + [0] * len(protein_bal))
    aucs = []
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for train, test in cv.split(x, y):
        model = LogisticRegression(max_iter=2000, solver="liblinear", class_weight="balanced")
        model.fit(x[train], y[train])
        scores = model.predict_proba(x[test])[:, 1]
        aucs.append(roc_auc_score(y[test], scores))

    rows = []
    for label, same, cross in [
        ("RNA query", rna_to_rna, rna_to_protein),
        ("Protein query", protein_to_protein, protein_to_rna),
    ]:
        rows.append(
            {
                "query_class": label,
                "n_query": len(same),
                "median_same_class_max_tanimoto": float(np.median(same)),
                "median_cross_class_max_tanimoto": float(np.median(cross)),
                "cross_ge_0.30_fraction": float((cross >= 0.30).mean()),
                "cross_ge_0.50_fraction": float((cross >= 0.50).mean()),
                "mean_bias_same_minus_cross": float((same - cross).mean()),
            }
        )
    ref_table = pd.DataFrame(rows)
    ref_table["balanced_classifier_auc_mean"] = float(np.mean(aucs))
    ref_table["balanced_classifier_auc_sd"] = float(np.std(aucs, ddof=1))
    ref_table.to_csv(OUT / "reference_overlap.csv", index=False)

    return {
        "n_balanced_per_class": int(n),
        "auc_mean": float(np.mean(aucs)),
        "auc_sd": float(np.std(aucs, ddof=1)),
        "rna_cross_median": float(np.median(rna_to_protein)),
        "rna_cross_ge_030": float((rna_to_protein >= 0.30).mean()),
        "rna_cross_ge_050": float((rna_to_protein >= 0.50).mean()),
        "protein_cross_median": float(np.median(protein_to_rna)),
        "protein_cross_ge_030": float((protein_to_rna >= 0.30).mean()),
        "protein_cross_ge_050": float((protein_to_rna >= 0.50).mean()),
    }


def bias_summary_from_smiles(
    smiles: list[str],
    rna_ref: np.ndarray,
    protein_ref: np.ndarray,
    sample_limit: int | None = None,
) -> tuple[list[str], dict]:
    smiles = pd.Series(smiles).dropna().astype(str).unique().tolist()
    smiles = pd.Series(smiles).map(canonical_smiles).dropna().unique().tolist()
    smiles = [s for s in smiles if is_druglike_smiles(s)]
    if sample_limit is not None and len(smiles) > sample_limit:
        smiles = list(RNG.choice(np.asarray(smiles, dtype=object), sample_limit, replace=False))
    smiles, fps = fp_array_from_smiles(smiles)
    if len(fps) == 0:
        return smiles, {
            "n_unique": 0,
            "mean_bias": np.nan,
            "rna_leaning_fraction": np.nan,
            "intermediate_fraction": np.nan,
            "protein_leaning_fraction": np.nan,
            "median_max_tanimoto_to_rna": np.nan,
            "median_max_tanimoto_to_protein": np.nan,
        }

    t_rna = max_tanimoto_to_ref(fps, rna_ref)
    t_protein = max_tanimoto_to_ref(fps, protein_ref)
    bias = t_rna - t_protein
    return smiles, {
        "n_unique": len(fps),
        "mean_bias": float(bias.mean()),
        "rna_leaning_fraction": float((bias > 0.05).mean()),
        "intermediate_fraction": float((np.abs(bias) <= 0.05).mean()),
        "protein_leaning_fraction": float((bias < -0.05).mean()),
        "median_max_tanimoto_to_rna": float(np.median(t_rna)),
        "median_max_tanimoto_to_protein": float(np.median(t_protein)),
    }


def model_bias_and_diversity(
    benchmark: str,
    gen: pd.DataFrame,
    method_order: list[str],
    rna_ref: np.ndarray,
    protein_ref: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    bias_rows = []
    diversity_rows = []

    for mode in method_order:
        sub = gen[gen["mode"] == mode].copy()
        if sub.empty:
            continue

        for group in ["broad", "screened", "strict"]:
            g = select_rows(sub, group)
            sample_limit = N_BIAS_SAMPLE if group == "broad" else None
            _, stats = bias_summary_from_smiles(
                g["SMILES"].dropna().tolist(),
                rna_ref,
                protein_ref,
                sample_limit=sample_limit,
            )
            bias_rows.append(
                {
                    "benchmark": benchmark,
                    "method": METHOD_LABELS[mode],
                    "set": group,
                    **stats,
                }
            )

        for group in ["screened", "strict"]:
            g = select_rows(sub, group)
            raw_smiles = pd.Series(g["SMILES"].dropna().astype(str).unique()).tolist()
            smiles = pd.Series(raw_smiles).map(canonical_smiles).dropna().unique().tolist()
            smiles, fps = fp_array_from_smiles(smiles)
            scaffolds = [murcko_scaffold(s) for s in smiles]
            scaffolds = [s for s in scaffolds if s]
            diversity_rows.append(
                {
                    "benchmark": benchmark,
                    "method": METHOD_LABELS[mode],
                    "set": group,
                    "n_unique_smiles": len(smiles),
                    "n_unique_bm_scaffolds": len(set(scaffolds)),
                    "bm_scaffold_per_molecule": len(set(scaffolds)) / len(smiles) if smiles else np.nan,
                    "internal_ecfp4_diversity": internal_diversity(fps),
                    "n_targets": int(g["pdb_id"].nunique()) if "pdb_id" in g else 0,
                }
            )

    bias_table = pd.DataFrame(bias_rows)
    div_table = pd.DataFrame(diversity_rows)
    return bias_table, div_table


def control_ligand_bias(rna_ref: np.ndarray, protein_ref: np.ndarray) -> pd.DataFrame:
    controls = []

    smiles = unique_csv_values(NB / "output" / "smrtbench_combined_results.csv", "active_smi")
    _, stats = bias_summary_from_smiles(smiles, rna_ref, protein_ref, sample_limit=N_BIAS_SAMPLE)
    controls.append({"control_set": "SMRT-Net positive ligands", **stats})

    smiles = unique_csv_values(NB / "DUDE_active_compounds.csv", "canonical_smiles")
    _, stats = bias_summary_from_smiles(smiles, rna_ref, protein_ref, sample_limit=N_BIAS_SAMPLE)
    controls.append({"control_set": "DUD-E active ligands", **stats})
    table = pd.DataFrame(controls)
    table.to_csv(OUT / "control_ligand_space_bias.csv", index=False)
    return table


def update_control_tables_only() -> None:
    print("Loading reference chemical-space cache...")
    _, rna_fps, _, protein_fps = load_reference()
    print(f"  BindingDB RNA: {len(rna_fps):,}; Protein: {len(protein_fps):,}")

    # Keep the RNG trajectory consistent with the full workflow before sampling
    # the balanced reference sets used by the control bias calculation.
    reference_overlap_and_separability(rna_fps, protein_fps)

    n_ref = min(len(rna_fps), len(protein_fps))
    rna_ref_idx = RNG.choice(len(rna_fps), n_ref, replace=False)
    protein_ref_idx = RNG.choice(len(protein_fps), n_ref, replace=False)
    rna_ref = rna_fps[rna_ref_idx]
    protein_ref = protein_fps[protein_ref_idx]

    control_table = control_ligand_bias(rna_ref, protein_ref)
    summary_path = OUT / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    else:
        summary = {}
    summary["controls"] = control_table.to_dict(orient="records")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nControl ligand-space bias:")
    print(control_table.to_string(index=False))
    print(f"\nWrote controls to {OUT / 'control_ligand_space_bias.csv'}")


def main() -> None:
    args = parse_args()
    if args.controls_only:
        update_control_tables_only()
        return

    print("Loading reference chemical-space cache...")
    rna_smiles, rna_fps, protein_smiles, protein_fps = load_reference()
    print(f"  BindingDB RNA: {len(rna_fps):,}; Protein: {len(protein_fps):,}")

    ref_summary = reference_overlap_and_separability(rna_fps, protein_fps)

    n_ref = min(len(rna_fps), len(protein_fps))
    rna_ref_idx = RNG.choice(len(rna_fps), n_ref, replace=False)
    protein_ref_idx = RNG.choice(len(protein_fps), n_ref, replace=False)
    rna_ref = rna_fps[rna_ref_idx]
    protein_ref = protein_fps[protein_ref_idx]

    print("Loading generated molecules...")
    smrt = pd.read_csv(NB / "output" / "smrtbench_combined_results.csv")
    smrt = ensure_sei(smrt[smrt["mode"].isin(SMRT_METHOD_ORDER)].copy())
    print(f"  SMRTnet RNA rows: {len(smrt):,}")
    smrt_bias, smrt_div = model_bias_and_diversity(
        "SMRTnet RNA", smrt, SMRT_METHOD_ORDER, rna_ref, protein_ref
    )

    dude = pd.read_csv(NB / "dude_stat_all.csv")
    dude = ensure_sei(dude[dude["mode"].isin(DUDE_METHOD_ORDER)].copy())
    print(f"  DUD-E protein rows: {len(dude):,}")
    dude_bias, dude_div = model_bias_and_diversity(
        "DUD-E protein", dude, DUDE_METHOD_ORDER, rna_ref, protein_ref
    )

    bias_table = pd.concat([smrt_bias, dude_bias], ignore_index=True)
    div_table = pd.concat([smrt_div, dude_div], ignore_index=True)
    bias_table.to_csv(OUT / "model_ligand_space_bias.csv", index=False)
    div_table.to_csv(OUT / "model_diversity.csv", index=False)
    control_table = control_ligand_bias(rna_ref, protein_ref)

    summary = {
        "reference": ref_summary,
        "bias": bias_table.to_dict(orient="records"),
        "diversity": div_table.to_dict(orient="records"),
        "controls": control_table.to_dict(orient="records"),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nReference overlap:")
    print(pd.read_csv(OUT / "reference_overlap.csv").to_string(index=False))
    print("\nModel ligand-space bias:")
    print(bias_table.to_string(index=False))
    print("\nModel diversity:")
    print(div_table.to_string(index=False))
    print("\nControl ligand-space bias:")
    print(control_table.to_string(index=False))
    print(f"\nWrote outputs to {OUT}")


if __name__ == "__main__":
    main()
