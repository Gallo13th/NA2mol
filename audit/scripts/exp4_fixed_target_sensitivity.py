#!/usr/bin/env python
"""Experiment IV - (a) reconcile the manuscript's original recovery values and
(b) fixed-target sensitivity: for each reference filter, compare the filtered
A_obs against the ORIGINAL A_obs recomputed on exactly the same eligible-target
subset, so that the target-composition and reference-chemistry effects are
separated.

Outputs
  tables/recovery_reproduction_check.csv
  tables/recovery_fixed_target_sensitivity.csv
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

FILTERS = ["ft_exact", "ft_T0.5", "ft_T0.6", "ft_T0.7",
           "pretrain_exact", "pretrain_T0.5"]
METHODS = {"SMRT-Net": ["RNA2smol", "MolCRAFT", "Pocket2Mol", "ED2Mol"],
           "DUD-E": ["RNA2smol", "MolCRAFT", "Pocket2Mol", "ED2Mol"]}


def ffp(smi):
    m = Chem.MolFromSmiles(smi)
    return GEN.GetFingerprint(m) if m is not None else None


def keep_mask(ov, filt):
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
    for m, rows in mrows.items():
        if len(rows) == 0:
            continue
        d = {"full": S[rows, :].max(axis=1).astype(np.float32)}
        for filt in FILTERS:
            keep = keep_mask(ov, filt)
            d[filt] = (S[rows][:, keep].max(axis=1).astype(np.float32)
                       if keep.sum() else None)
        out[m] = d
    return (bench, tid, out, {m: len(r) for m, r in mrows.items()})


def trapz(y, x, axis=-1):
    return np.trapezoid(y, x, axis=axis) if hasattr(np, "trapezoid") else np.trapz(y, x, axis=axis)


def curve_from_g(g, T):
    """A_obs over tau in [0.25,0.5] from a per-target max vector g (NaN-excluded)."""
    g = g[np.isfinite(g)]
    if len(g) == 0:
        return np.nan
    r = (g[:, None] >= TAU[None, :]).mean(axis=0)
    return float(trapz(r, TAU) / SPAN)


def boot_ci(g, rng):
    g = g[np.isfinite(g)]
    if len(g) == 0:
        return (np.nan, np.nan)
    G = (g[:, None] >= TAU[None, :])
    bidx = rng.integers(0, len(g), size=(N_BOOT, len(g)))
    boot = trapz(G[bidx].mean(axis=1), TAU, axis=1) / SPAN
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def main():
    ref = pd.read_csv(AUDIT / "tables" / "reference_actives.csv", dtype={"target_id": str})
    ref = ref[ref.parse_ok]
    ov = pd.read_csv(AUDIT / "tables" / "reference_active_overlap.csv", dtype={"target_id": str}, low_memory=False)
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

    # per-target, per-method max vectors
    data = {}   # bench -> method -> {filt -> {tid: g}}
    cover = {}  # bench -> method -> set(tid)
    for bench, tid, out, counts in res:
        cover.setdefault(bench, {})
        for m, d in out.items():
            dm = data.setdefault(bench, {}).setdefault(m, {})
            dm.setdefault("full", {})[tid] = float(np.nanmax(d["full"]))
            cover[bench].setdefault(m, set()).add(tid)
            for filt in FILTERS:
                if d[filt] is not None:
                    dm.setdefault(filt, {})[tid] = float(np.nanmax(d[filt]))

    # reference actives per target (for coverage diagnostics)
    ref_t = ref.groupby(["benchmark", "target_id"]).size()

    # ----- (a) reproduction check: per-method on its own covered targets -----
    published = {"SMRT-Net": {"RNA2smol": 0.215463, "MolCRAFT": 0.059630,
                             "Pocket2Mol": 0.080556, "ED2Mol": 0.069352},
                 "DUD-E": {"RNA2smol": 0.648267, "MolCRAFT": 0.350825,
                           "Pocket2Mol": 0.395930, "ED2Mol": 0.476211}}
    rng = np.random.default_rng(SEED)
    rows = []
    for bench in ["SMRT-Net", "DUD-E"]:
        for m in METHODS[bench]:
            all_t = sorted(data.get(bench, {}).get(m, {}).get("full", {}).keys())
            g = np.array([data[bench][m]["full"][t] for t in all_t])
            a = curve_from_g(g, len(all_t))
            lo, hi = boot_ci(g, rng)
            rows.append(dict(benchmark=bench, method=m, n_targets=len(all_t),
                             A_obs_recomputed=round(a, 4),
                             A_obs_boot_lo=round(lo, 4), A_obs_boot_hi=round(hi, 4),
                             A_obs_published=published[bench][m],
                             difference=round(a - published[bench][m], 4)))
    rep = pd.DataFrame(rows)

    # manuscript's own per-threshold target-success tables (native A_obs)
    note = AUDIT.parent / "notebooks" / "output" / "tables"
    manu_tab = {
        "DUD-E": note / "dude" / "d2_ecfp4_target_success_by_threshold.csv",
        "SMRT-Net": note / "smrtbench" / "t1b_ecfp4_target_success_by_threshold.csv",
    }
    manu_name = {
        "DUD-E": {"RNA2smol": "RNA2smol", "MolCRAFT": "MolCRAFT",
                  "Pocket2Mol": "Pocket2Mol", "ED2Mol": "ed2mol"},
        "SMRT-Net": {"RNA2smol": "RNA2smol (Only General Feat)", "MolCRAFT": "MolCRAFT",
                     "Pocket2Mol": "Pocket2Mol", "ED2Mol": "ED2Mol"},
    }
    manu_a = {}
    for bench, path in manu_tab.items():
        if not path.exists():
            continue
        t = pd.read_csv(path)
        for m, mname in manu_name[bench].items():
            s = t[(t["method"] == mname) & (t["tau"] >= 0.25) & (t["tau"] <= 0.5)].sort_values("tau")
            if len(s) >= 2:
                ratio = (s["successful_targets_ge1_molecule"] / s["total_evaluable_targets"]).to_numpy(float)
                tt = s["tau"].to_numpy(float)
                manu_a[(bench, m)] = (round(float(trapz(ratio, tt) / 0.25), 4),
                                      int(s["total_evaluable_targets"].iloc[0]))
    rep["A_obs_manuscript_table"] = [manu_a.get((r.benchmark, r.method), (np.nan, np.nan))[0]
                                     for r in rep.itertuples()]
    rep["n_targets_manuscript_table"] = [manu_a.get((r.benchmark, r.method), (np.nan, np.nan))[1]
                                         for r in rep.itertuples()]
    rep.to_csv(AUDIT / "tables" / "recovery_reproduction_check.csv", index=False)
    print("=== (a) reproduction check (per-method own target set) ===")
    print(rep.to_string(index=False))

    # ----- (b) fixed-target sensitivity -----
    rows = []
    for bench in ["SMRT-Net", "DUD-E"]:
        methods = METHODS[bench]
        # full common set (all methods cover the target)
        full_common = set.intersection(*[cover.get(bench, {}).get(m, set()) for m in methods])
        for filt in FILTERS:
            # eligible = target has a filtered-active molecule for every method
            elig = None
            for m in methods:
                s = set(data.get(bench, {}).get(m, {}).get(filt, {}).keys())
                elig = s if elig is None else (elig & s)
            elig = sorted(elig or set())
            for m in methods:
                gf = np.array([data[bench][m][filt][t] for t in elig])
                go_sub = np.array([data[bench][m]["full"][t] for t in elig])
                go_full = np.array([data[bench][m]["full"][t] for t in sorted(full_common)])
                a_filt = curve_from_g(gf, len(elig))
                a_orig_sub = curve_from_g(go_sub, len(elig))
                a_orig_full = curve_from_g(go_full, len(full_common))
                n = len(elig)
                bd = np.array([curve_from_g(gf[ix], n) - curve_from_g(go_sub[ix], n)
                               for ix in (rng.integers(0, n, n) for _ in range(N_BOOT))])
                rows.append(dict(benchmark=bench, filter=filt, method=m,
                                 n_eligible=n,
                                 A_original_subset=round(a_orig_sub, 4),
                                 A_filtered_subset=round(a_filt, 4),
                                 delta_filter_effect=round(a_filt - a_orig_sub, 4),
                                 delta_ci_lo=round(float(np.percentile(bd, 2.5)), 4),
                                 delta_ci_hi=round(float(np.percentile(bd, 97.5)), 4),
                                 A_original_full=round(a_orig_full, 4),
                                 delta_target_composition=round(a_orig_sub - a_orig_full, 4)))
    fix = pd.DataFrame(rows)
    fix.to_csv(AUDIT / "tables" / "recovery_fixed_target_sensitivity.csv", index=False)
    print("\n=== (b) fixed-target sensitivity ===")
    print(fix.to_string(index=False))

    # coverage diagnostics
    diag = []
    for bench in ["SMRT-Net", "DUD-E"]:
        n_ref = ref_t.loc[bench].shape[0]
        diag.append(dict(benchmark=bench, n_targets_with_actives=n_ref))
        for m in METHODS[bench]:
            diag[-1][f"{m}_covered"] = len(cover.get(bench, {}).get(m, set()))
        diag[-1]["common_full"] = len(set.intersection(
            *[cover.get(bench, {}).get(m, set()) for m in METHODS[bench]]))
    pd.DataFrame(diag).to_csv(AUDIT / "tables" / "recovery_target_coverage.csv", index=False)
    print("\n=== coverage ===")
    print(pd.DataFrame(diag).to_string(index=False))


if __name__ == "__main__":
    main()
