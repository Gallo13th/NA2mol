# Reproduction validation

Result of re-running the pipeline end to end from this checkout, rather than trusting
that the scripts exist. Priority was the statistical tables and figures.

**Date:** 16 October 2026 · **HEAD:** `1719430` + the fixes in this commit

---

## Summary

| Layer | Result |
|---|---|
| Static path resolution | **PASS** — every `ROOT / "..."` data path in a figure script resolves |
| Numerical tables | **40 of 44 regenerate byte-identically**; 4 radar tables differ (see §2) |
| Figures | **6 of 7 PASS**; Supplementary Fig. 4 blocked on absent panel assets (§4) |
| Training scripts | compile; `pretrain_ed.py` now points at the verified LMDB |

The most important outcome is **not** the pass count — it is that running the pipeline
surfaced four defects that static checks could not see. They are listed in §3.

---

## 1. Method

The naive check — re-run the exporter and diff — is **invalid here**: several blocks in
`export_tables.py` are guarded by `if <cache>.exists():`, so a missing cache skips the
block and leaves the *previous* run's file in place, which then compares equal and looks
reproducible. That is exactly what happened on the first attempt.

`reproduction/scripts/verify_tables.sh` therefore **deletes the tables first**, regenerates
from scratch, and reports anything that could not be produced. `export_tables.py` now does
the same clearing itself and exits non-zero with the missing tables named.

Figures were verified by `reproduction/scripts/verify_figures.sh`, which runs each script
and requires a real, non-blank, multi-page PDF — a script that exits 0 without producing a
figure counts as a failure.

---

## 2. Tables: 40 identical, 4 differ

Byte-identical across a from-scratch regeneration:

- all headline statistics — `t1_main_comparison`, `t1b_ecfp4_*`, `t1d_per_target_paired_stats`
- the ablation tables — `t2_ablation_comparison`, `t2b_ablation_paired_stats`
- both screening funnels — `t6`, `t6b`, `d7`, `d7b`
- per-method/per-target heatmaps and property statistics
- the DUD-E equivalents

**Four tables differ: `t7a`, `t7b`, `t7c`, `t8`** (the radar profile and its rank/min-max
normalisations).

Cause: the `bond_len_js` / `bond_angle_js` metrics come from 3D conformer embedding, which
is **not numerically stable across RDKit versions**. The sampling *is* seeded
(`RandomState(42)`, `EmbedMolecule(randomSeed=42)`), so this is not run-to-run randomness —
it is a version-dependent numerical difference in bond lengths and angles, propagated into a
histogram JS divergence.

**Consequence, and why it matters:** rank normalisation amplifies small numeric differences
into a **rank swap**:

| Method | `bond_len_js_rank_norm` shipped | regenerated |
|---|---|---|
| NA2smol (baseline) | 0.625 | **0.4375** |
| ED2Mol | 0.4375 | **0.625** |

That is a visible change in the radar panel of Main Fig. 2. The shipped tables have been
restored so the figure remains faithful to the published version, but the metric should be
treated as version-sensitive: **record the RDKit version used for the paper** and expect
small radar-panel differences on other versions. Values elsewhere on that axis agree to
~1–5 % (e.g. MolCRAFT 0.146261 vs 0.147350).

---

## 3. Defects found by running the pipeline

All four were invisible to static checks and are now fixed.

1. **Tables were written to the wrong directory.** `export_tables.py` had
   `OUT_DIR = Path(__file__).resolve().parent / 'output' / 'tables'` — the restructure
   rewrote the other data paths but missed this one, so tables went to `analysis/output/`
   while everything downstream read `data/benchmark/output/`. The same bug was in
   `prepare_strain_energy.py`. *This is also why the first "byte-for-byte" claim was wrong:
   the check hashed one directory while the script wrote to another.*

2. **The cleanup had removed load-bearing code.** `visualize_methods.py` and
   `dude_eval_viz.py` were quarantined as "exploratory", but they are the **only**
   implementations of the bond-JS metrics that `export_tables.py` reads from cache. With
   them gone, `d8b`/`d8c` were never written and Main Fig. 2 failed with
   `FileNotFoundError`. Restored as `analysis/prepare_bond_js.py` and
   `analysis/prepare_bond_js_dude.py`.

3. **A missing input failed silently.** Because the radar blocks are guarded by
   `if cache.exists()`, an absent cache produced a *partial* table set with exit code 0;
   the failure only appeared later, in a figure script. `export_tables.py` now clears its
   output, checks a list of 15 required tables, and exits 2 naming what is missing and
   which producer to run.

4. **Two more `__file__`-relative data paths** survived the restructure
   (`Path(__file__).parent / 'dude_stat_all.csv'` and an `OUT_DIR` in the restored scripts).

---

## 4. Figures: 6 of 7

| Figure | Status |
|---|---|
| Main Fig. 2 — paired benchmark | **PASS** (8 panels, no text overlaps) |
| Main Fig. 3 — four-configuration ablation | **PASS** |
| Main Fig. 4 — KRAS G4 | **PASS** |
| SI Fig. 1 — chemical-space audit | **PASS** |
| SI Fig. 2 — MolCRAFT mapping control | **PASS** |
| SI Fig. 3 — property landscape | **PASS** |
| SI Fig. 4 — KRAS ranking | **BLOCKED** |
| Main Fig. 4g — ThT panel | **BLOCKED** (Origin export pending) |

### Supplementary Fig. 4 — missing panel assets

`case_study/kras_g4/assemble_extended_data_fig_kras.py` composes two panels from
`case_study/kras_g4/fig5_panels/{c_relaxed,d_triage}.{pdf,svg}`. **That directory does not
exist, and no script in the repository or in the quarantine backup produces it.** The
generator was part of the pre-extension Fig. 4 pipeline that was replaced.

Either supply those four files, or identify the script that produced them. Until then
Supplementary Fig. 4 cannot be regenerated.

### Main Fig. 4g — ThT panel

Needs the Origin per-sheet export in `data/tht/export/`. `analysis/fig4g_tht_panel.py` and
`reproduction/scripts/check_tht_origin.py` are ready; see `data/tht/README.md`.

---

## 5. Required caches

`export_tables.py` reads four caches from `data/benchmark/output/`. They are **not in git**
and two of them are load-bearing:

| Cache | Producer | If absent |
|---|---|---|
| `cache_bond_js.pkl` | `analysis/prepare_bond_js.py` | t7a–t7c, t8 lose the bond-JS metrics |
| `cache_dude_bond_js.pkl` | `analysis/prepare_bond_js_dude.py` | **d8b/d8c not written → Main Fig. 2 fails** |
| `cache_strain_energy.pkl` | `analysis/prepare_strain_energy.py` | strain column falls back |
| `cache_ecfp4_enrich.csv` | top-N enrichment step | ECFP4 top-N tables skipped |

Build both bond-JS caches in one step (≈10 min; RDKit embedding):

```bash
python3 reproduction/scripts/build_bond_js_caches.py
```

---

## 6. How to re-verify

```bash
bash reproduction/scripts/smoke_test.sh          # static: paths, schema, syntax
bash reproduction/scripts/verify_tables.sh       # honest table regeneration
bash reproduction/scripts/verify_figures.sh      # run every figure, validate output
```

Run the long ones under `tmux`.
