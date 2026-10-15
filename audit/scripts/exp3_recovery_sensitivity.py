#!/usr/bin/env python
"""Experiment III: is the published ECFP4-TS recovery robust to overlap?

Reproduces the target-level recovery (mean over tau in [0.25,0.5] of the
fraction of targets with >=1 molecule reaching ECFP4-TS >= tau) and repeats it
after excluding reference actives that overlap the fine-tuning training set or
the pretraining library. All methods are scored on a common eligible-target set.
The permutation null keeps molecule-level similarities and shuffles the
molecule-to-target labels (documented definition).

Outputs
  tables/recovery_original_vs_overlap_filtered.csv
  tables/recovery_curves.csv
"""
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator

sys.path.insert(0, str(Path(__file__).resolve().parent))
from target_ids import validate_join_keys
from lka_common import AUDIT

RDLogger.DisableLog("rdApp.*")
GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
TAU = np.linspace(0.25, 0.5, 201)
SPAN = 0.25
N_PERM = 300
N_BOOT = 2000
SEED = 0

FILTERS = ["original", "ft_exact", "ft_T0.5", "ft_T0.6", "ft_T0.7",
           "pretrain_exact", "pretrain_T0.5"]
METHODS = {
    "SMRT-Net": ["RNA2smol", "MolCRAFT", "Pocket2Mol", "ED2Mol"],
    "DUD-E": ["RNA2smol", "MolCRAFT", "Pocket2Mol", "ED2Mol"],
}


def ffp(smi):
    m = Chem.MolFromSmiles(smi)
    return GEN.GetFingerprint(m) if m is not None else None


def keep_mask(ov, filt):
    if filt == "original":
        return np.ones(len(ov), dtype=bool)
    if filt == "ft_exact":
        return ~ov["exact_ft_train"].to_numpy(bool)
    if filt == "pretrain_exact":
        return ~ov["exact_pretrain"].to_numpy(bool)
    if filt.startswith("ft_T"):
        return ov["maxts_ft_train"].to_numpy(float) < float(filt.split("T")[1])
    if filt.startswith("pretrain_T"):
        return ov["maxts_pretrain"].to_numpy(float) < float(filt.split("T")[1])
    raise ValueError(filt)


def target_work(target):
    bench, tid = target["benchmark"], target["target_id"]
    act, ov, gen = target["actives"], target["overlap"], target["gen"]
    afps = [f for f in (ffp(s) for s in act["canonical"]) if f is not None]
    if not afps:
        return None
    uniq = list(dict.fromkeys(gen["canonical"].tolist()))
    gfps = {s: ffp(s) for s in uniq}
    uniq = [s for s in uniq if gfps[s] is not None]
    if not uniq:
        return None
    S = np.zeros((len(uniq), len(afps)), dtype=np.float32)
    for i, s in enumerate(uniq):
        S[i] = DataStructs.BulkTanimotoSimilarity(gfps[s], afps)
    idx = {s: i for i, s in enumerate(uniq)}
    mrows = {m: np.array([idx[s] for s in gen[gen["method"] == m]["canonical"] if s in idx])
             for m in METHODS[bench]}
    out = {}
    for filt in FILTERS:
        keep = keep_mask(ov, filt)
        if keep.sum() == 0 or not all(len(v) for v in mrows.values()):
            out[filt] = None
            continue
        tmax = S[:, keep].max(axis=1)
        out[filt] = {m: tmax[rows].astype(np.float32) for m, rows in mrows.items()}
    return (bench, tid, out)


def curve(labels, sims):
    if len(sims) == 0:
        return np.nan
    g = pd.Series(sims).groupby(labels).max().to_numpy(float)
    r = (g[None, :] >= TAU[:, None]).mean(axis=1)
    auc = np.trapezoid(r, TAU) if hasattr(np, "trapezoid") else np.trapz(r, TAU)
    return float(auc / SPAN)


def main():
    ref = pd.read_csv(AUDIT / "tables" / "reference_actives.csv", dtype={"target_id": str})
    ref = ref[ref.parse_ok]
    ov = pd.read_csv(AUDIT / "tables" / "reference_active_overlap.csv", dtype={"target_id": str})
    gen = pd.read_csv(AUDIT / "tables" / "generated_long.csv", dtype={"target_id": str})
    validate_join_keys(ref, ov, gen)
    ref["_k"] = ref["benchmark"] + "|" + ref["target_id"].astype(str) + "|" + ref["canonical"]
    ov["_k"] = ov["benchmark"] + "|" + ov["target_id"].astype(str) + "|" + ov["canonical"]
    key = ["exact_ft_train", "exact_pretrain", "maxts_ft_train", "maxts_pretrain"]
    ref = ref.merge(ov[key + ["_k"]].drop_duplicates("_k"), on="_k", how="left")

    targets = []
    for (bench, tid), act in ref.groupby(["benchmark", "target_id"]):
        g = gen[(gen.benchmark == bench) & (gen.target_id == tid)]
        if len(g) and len(act):
            targets.append(dict(benchmark=bench, target_id=tid, actives=act,
                                overlap=act.reset_index(drop=True), gen=g))
    print("targets", len(targets), flush=True)
    with Pool(12) as p:
        res = [r for r in p.map(target_work, targets, chunksize=2) if r]

    rng = np.random.default_rng(SEED)

    def trapz(y, x, axis=-1):
        return np.trapezoid(y, x, axis=axis) if hasattr(np, "trapezoid") else np.trapz(y, x, axis=axis)

    rows, curve_rows = [], []
    for bench in ["SMRT-Net", "DUD-E"]:
        rbench = [r for r in res if r[0] == bench]
        methods = METHODS[bench]
        for filt in FILTERS:
            ok = {tid: out[filt] for (_, tid, out) in rbench if out[filt] is not None}
            per_method = {}
            for m in methods:
                labs, sims, elig = [], [], set()
                for tid, pm in ok.items():
                    v = pm[m]
                    if len(v):
                        elig.add(tid)
                        labs.append(np.full(len(v), tid, dtype=object))
                        sims.append(v)
                if labs:
                    per_method[m] = (np.concatenate(labs), np.concatenate(sims), elig)
            if not per_method:
                continue
            common = set.intersection(*[per_method[m][2] for m in per_method])
            if not common:
                continue
            ut = np.array(sorted(common), dtype=object)
            for m in methods:
                if m not in per_method:
                    continue
                labs, sims, _ = per_method[m]
                sel = np.isin(labs, ut)
                labs, sims = labs[sel], sims[sel]
                ti = np.searchsorted(ut, labs.astype(str))
                T = len(ut)
                g = np.full(T, -np.inf)
                np.maximum.at(g, ti, sims)
                # observed = target maxima above tau
                G = (g[:, None] >= TAU[None, :])          # T x n_tau
                r = G.mean(axis=0)
                a_obs = float(trapz(r, TAU) / SPAN)
                # bootstrap over targets (target-level uncertainty)
                bidx = rng.integers(0, T, size=(N_BOOT, T))
                boot = trapz(G[bidx].mean(axis=1), TAU, axis=1) / SPAN
                curve_rows.append(dict(benchmark=bench, filter=filt, method=m,
                                       target_id="|".join(map(str, ut)),
                                       n_eligible_targets=T,
                                       target_max="|".join(f"{x:.4f}" for x in g)))
                rows.append(dict(benchmark=bench, filter=filt, method=m,
                                 n_eligible_targets=T,
                                 a_obs=round(a_obs, 4),
                                 a_obs_boot_lo=round(float(np.percentile(boot, 2.5)), 4),
                                 a_obs_boot_hi=round(float(np.percentile(boot, 97.5)), 4)))
    out = pd.DataFrame(rows)
    # Attach the manuscript's published permutation-gain values for the original
    # analysis (the null requires the full molecule x target matrix; it is not
    # recomputed for the filtered reference sets - see the audit summary).
    pub = {"SMRT-Net": AUDIT.parent / "notebooks" / "output" / "tables" / "smrtbench" / "ecfp4_permutation_gain.csv",
           "DUD-E": AUDIT.parent / "notebooks" / "output" / "tables" / "dude" / "ecfp4_permutation_gain.csv"}
    name_map = {"RNA2smol (Only General Feat)": "RNA2smol", "RNA2smol": "RNA2smol",
                "MolCRAFT": "MolCRAFT", "Pocket2Mol": "Pocket2Mol",
                "ED2Mol": "ED2Mol", "ed2mol": "ED2Mol"}
    frames = []
    for bench, path in pub.items():
        if path.exists():
            p = pd.read_csv(path)
            p["method"] = p["method"].map(name_map)
            p["benchmark"] = bench
            frames.append(p)
    if frames:
        pdf = pd.concat(frames)[["benchmark", "method", "ecfp4_A_obs", "ecfp4_A_rand",
                                 "max_ecfp4_tanimoto_raw"]].rename(
            columns={"ecfp4_A_obs": "published_A_obs", "ecfp4_A_rand": "published_A_rand",
                     "max_ecfp4_tanimoto_raw": "published_gain_S"})
        out = out.merge(pdf, on=["benchmark", "method"], how="left")
    out.to_csv(AUDIT / "tables" / "recovery_original_vs_overlap_filtered.csv", index=False)
    pd.DataFrame(curve_rows).to_csv(AUDIT / "tables" / "recovery_curves.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
