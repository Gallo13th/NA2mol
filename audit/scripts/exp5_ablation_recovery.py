#!/usr/bin/env python
"""Experiment V - ablation-ladder recovery on SMRT-Net.

Adds the no-pretraining baseline (RNA2smol (Baseline), pretraining omitted) and
the intermediate configurations (Pretrain Only, Pretrain+Constrain) to the
recovery-sensitivity and fixed-target analyses, so the contribution of
electron-density pretraining to the ECFP4 recovery (and to its sensitivity to
the reference filters) can be separated from the shared representation.

Configurations (generated_long tags):
  ablation:RNA2smol (Baseline)   no pretraining (lambda_cont = lambda_constrain = 0)
  ablation:Pretrain Only         pretraining, no regularisation
  ablation:Pretrain+Constrain    pretraining + constrained learning
  RNA2smol                       full model (pretraining + constraint + contrastive)

Outputs
  tables/recovery_ablation_ladder.csv
  tables/recovery_ablation_coverage.csv
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
N_BOOT = 2000
SEED = 0

FILTERS = ["original", "ft_exact", "ft_T0.5", "ft_T0.6", "ft_T0.7",
           "pretrain_exact", "pretrain_T0.5"]
CONFIGS = ["ablation:RNA2smol (Baseline)", "ablation:Pretrain Only",
           "ablation:Pretrain+Constrain", "RNA2smol"]


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
    tid = target["target_id"]
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
    out = {}
    for m in CONFIGS:
        rows = np.array([idx[s] for s in gen[gen["method"] == m]["canonical"] if s in idx])
        if len(rows) == 0:
            continue
        d = {"original": S[rows, :].max(axis=1).astype(np.float32)}
        for filt in FILTERS[1:]:
            keep = keep_mask(ov, filt)
            d[filt] = (S[rows][:, keep].max(axis=1).astype(np.float32) if keep.sum() else None)
        out[m] = d
    return (tid, out)


def trapz(y, x, axis=-1):
    return np.trapezoid(y, x, axis=axis) if hasattr(np, "trapezoid") else np.trapz(y, x, axis=axis)


def curve(g):
    g = np.asarray(g, float)
    g = g[np.isfinite(g)]
    if len(g) == 0:
        return np.nan
    r = (g[:, None] >= TAU[None, :]).mean(axis=0)
    return float(trapz(r, TAU) / SPAN)


def boot_ci(g, rng):
    g = np.asarray(g, float)
    g = g[np.isfinite(g)]
    if len(g) == 0:
        return (np.nan, np.nan)
    G = (g[:, None] >= TAU[None, :])
    bidx = rng.integers(0, len(g), size=(N_BOOT, len(g)))
    b = trapz(G[bidx].mean(axis=1), TAU, axis=1) / SPAN
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def main():
    ref = pd.read_csv(AUDIT / "tables" / "reference_actives.csv", dtype={"target_id": str})
    ref = ref[ref.parse_ok]
    ov = pd.read_csv(AUDIT / "tables" / "reference_active_overlap.csv", dtype={"target_id": str}, low_memory=False)
    gen = pd.read_csv(AUDIT / "tables" / "generated_long.csv", dtype={"target_id": str})
    gen = gen[gen.benchmark == "SMRT-Net"]
    ref = ref[ref.benchmark == "SMRT-Net"]
    validate_join_keys(ref, ov, gen)
    ref["_k"] = ref["benchmark"] + "|" + ref["target_id"].astype(str) + "|" + ref["canonical"]
    ov["_k"] = ov["benchmark"] + "|" + ov["target_id"].astype(str) + "|" + ov["canonical"]
    key = ["exact_ft_train", "exact_pretrain", "maxts_ft_train", "maxts_pretrain"]
    ref = ref.merge(ov[key + ["_k"]].drop_duplicates("_k"), on="_k", how="left")

    targets = []
    for tid, act in ref.groupby("target_id"):
        g = gen[gen.target_id == tid]
        if len(g) and len(act):
            targets.append(dict(target_id=tid, actives=act,
                                overlap=act.reset_index(drop=True), gen=g))
    print("SMRT-Net targets", len(targets), flush=True)
    with Pool(12) as p:
        res = [r for r in p.map(target_work, targets, chunksize=2) if r]

    data = {}   # config -> filt -> {tid: g}
    for tid, out in res:
        for m, d in out.items():
            for filt, arr in d.items():
                if arr is not None:
                    data.setdefault(m, {}).setdefault(filt, {})[tid] = float(np.nanmax(arr))

    # coverage
    cov = [dict(config=m, targets_covered=len(data.get(m, {}).get("original", {})))
           for m in CONFIGS]
    cover = set.intersection(*[set(data.get(m, {}).get("original", {})) for m in CONFIGS])
    cov.append(dict(config="COMMON", targets_covered=len(cover)))
    pd.DataFrame(cov).to_csv(AUDIT / "tables" / "recovery_ablation_coverage.csv", index=False)

    rng = np.random.default_rng(SEED)
    rows = []
    for filt in FILTERS:
        elig = None
        for m in CONFIGS:
            s = set(data.get(m, {}).get(filt, {}).keys())
            elig = s if elig is None else (elig & s)
        elig = sorted(elig or set())
        for m in CONFIGS:
            gf = np.array([data[m][filt][t] for t in elig])
            go = np.array([data[m]["original"][t] for t in elig])
            a_f = curve(gf)
            a_o = curve(go)
            lo, hi = boot_ci(gf, rng)
            rows.append(dict(filter=filt, config=m, n_eligible=len(elig),
                             A_original_subset=round(a_o, 4),
                             A_filtered_subset=round(a_f, 4),
                             delta=round(a_f - a_o, 4),
                             A_filtered_boot_lo=round(lo, 4),
                             A_filtered_boot_hi=round(hi, 4)))
    out = pd.DataFrame(rows)
    out.to_csv(AUDIT / "tables" / "recovery_ablation_ladder.csv", index=False)

    # paired target-bootstrap intervals: Delta_A per config and the paired
    # contrasts Delta_Delta_A (full - no-pretrain) and filtered model contrast.
    full, base = "RNA2smol", "ablation:RNA2smol (Baseline)"
    paired = []
    for filt in FILTERS:
        elig = sorted(set.intersection(*[set(data[m][filt]) for m in CONFIGS]))
        n = len(elig)
        Dc = {m: np.array([data[m][filt][t] for t in elig]) for m in CONFIGS}
        Oc = {m: np.array([data[m]["original"][t] for t in elig]) for m in CONFIGS}
        for m in CONFIGS:
            est = curve(Dc[m]) - curve(Oc[m])
            bs = np.array([curve(Dc[m][ix]) - curve(Oc[m][ix])
                           for ix in (rng.integers(0, n, n) for _ in range(N_BOOT))])
            paired.append(dict(filter=filt, comparison=m, kind="delta_A", n_eligible=n,
                               estimate=round(est, 4),
                               ci_lo=round(float(np.percentile(bs, 2.5)), 4),
                               ci_hi=round(float(np.percentile(bs, 97.5)), 4)))
        dd = (curve(Dc[full]) - curve(Oc[full])) - (curve(Dc[base]) - curve(Oc[base]))
        bs = np.array([(curve(Dc[full][ix]) - curve(Oc[full][ix]))
                       - (curve(Dc[base][ix]) - curve(Oc[base][ix]))
                       for ix in (rng.integers(0, n, n) for _ in range(N_BOOT))])
        paired.append(dict(filter=filt, comparison="full-minus-nopretrain",
                           kind="delta_delta_A", n_eligible=n, estimate=round(dd, 4),
                           ci_lo=round(float(np.percentile(bs, 2.5)), 4),
                           ci_hi=round(float(np.percentile(bs, 97.5)), 4)))
        est = curve(Dc[full]) - curve(Dc[base])
        bs = np.array([curve(Dc[full][ix]) - curve(Dc[base][ix])
                       for ix in (rng.integers(0, n, n) for _ in range(N_BOOT))])
        paired.append(dict(filter=filt, comparison="full-minus-nopretrain",
                           kind="filtered_contrast", n_eligible=n, estimate=round(est, 4),
                           ci_lo=round(float(np.percentile(bs, 2.5)), 4),
                           ci_hi=round(float(np.percentile(bs, 97.5)), 4)))
    pd.DataFrame(paired).to_csv(AUDIT / "tables" / "recovery_ablation_paired_ci.csv", index=False)
    print("\n=== paired target-bootstrap (Delta_A and Delta_Delta_A) ===")
    pr = pd.DataFrame(paired)
    print(pr[pr.kind.isin(["delta_delta_A", "filtered_contrast"])].to_string(index=False))

    print("\n=== SMRT-Net ablation ladder: A_obs(ECFP4) per filter ===")
    piv = out.pivot_table(index="filter", columns="config", values="A_filtered_subset")
    print(piv.to_string())
    print("\n=== same-target original vs filtered (delta) ===")
    print(out.pivot_table(index="filter", columns="config", values="delta").to_string())


if __name__ == "__main__":
    main()
