# Reproduction validation

Result of re-running the pipeline end to end from this checkout, rather than trusting
that the scripts exist. Priority was the statistical tables and figures.

**Date:** 16 October 2026

> **Correction (see §0).** An earlier revision of this report recorded "Main Fig. 2 PASSES".
> That was wrong in the way that matters. The harness checked that a valid figure was
> *produced*, not that it was the *paper's* figure. The figure script renders a different
> chart type, in a different panel order, with different labels and a different
> normalisation from the manuscript. Section 0 documents this.

---

## 0. The figure scripts do not implement the published profile panels

The manuscript's Fig. 2 caption specifies panels a and d precisely:

> "a, DUD-E multi-metric profile across **seven metrics** (Dock., RMSD, QED, SA, SEI,
> ECFP4, Strain); d, the corresponding RNA profile. Docking, RMSD, QED, SA, SEI and Strain
> use the **normalised integral of the molecule-retention curve** over their threshold
> ranges (log scale for Strain); ECFP4 similarity uses the recovery gain relative to
> randomly reassigned target labels. **The radial scale spans $r=1$ to $r=3$.**"

Fig. 3a is specified as using "the same radar scheme".

The published figure — verified by rendering the manuscript PDF — shows **radar (spider)
charts** in panels a and d, with QED, SEI, ECFP4, Dock, RMSD, SA and Strain at the
vertices, titled "DUD-E protein" (a) and "RNA test set" (d).

What the repository actually contains:

| Aspect | Manuscript | Repository code |
|---|---|---|
| Chart type | **radar / spider** | **chart-enhanced table with data bars** (`plot_profile_table`, `bar_mode="rank"`) |
| Metrics in the profile | 7 | 7 **plotted**, but the source table carries 12 (adds LLE, LBE, rotatable\_bonds, bond\_len\_js, bond\_angle\_js) |
| Normalisation | normalised **retention-curve integral** over threshold ranges | **rank normalisation** with floor 0.25 (range 0.25–1) |
| Radial scale | **$r=1$ to $r=3$** | not applicable; bars, not radii |
| Panel order | a = DUD-E, d = RNA test set | a = RNA, **d = DUD-E** (swapped) |
| Panel title | "RNA test set" | "SMRT-Net RNA" |

Supporting checks:

- No code in `analysis/` computes a retention-curve integral. `trapz` appears only in
  `audit/scripts/exp3–exp5`, which integrate the **ECFP4-Tanimoto recovery curve**, not the
  six per-metric retention curves.
- Polar-axes code exists only in `analysis/prepare_bond_js.py` and `prepare_bond_js_dude.py`
  — the legacy per-method visualisation scripts — and it draws an **11-metric
  Rank+Floor / MinMax+Floor** radar, which is not the manuscript's scheme either.
- No analysis script uses the title "RNA test set"; the figures still say "SMRT-Net RNA".

**Conclusion: the repository does not contain the implementation that produced the published
Fig. 2/Fig. 3 profile panels.** The scripts present are an earlier or divergent generation:
they render a rank-normalised table where the paper has a retention-integral radar.

### Consequence for the earlier radar finding

The four radar tables whose values differ (`t7a`–`t7c`, `t8`) feed this non-matching
implementation. Their instability is real, but since `bond_len_js`/`bond_angle_js` **do not
appear in the published figure at all**, that finding is irrelevant to reproducing the paper.
It should not have been presented as a paper-reproducibility issue. The relevant problem is
that the correct implementation is absent.

---

## 1. What does reproduce

| Layer | Result |
|---|---|
| Static path resolution | **PASS** — every declared data path resolves |
| Numerical tables | **40 of 44 regenerate byte-identically** |
| Figures that do not use the profile panel | **PASS** — Fig. 2 panels b,c,e,f,g,h; Fig. 4; SI Figs. 1–3 |
| Training scripts | compile; `pretrain_ed.py` points at the verified LMDB (2,368,986 entries) |

Byte-identical across a from-scratch regeneration: all headline statistics
(`t1_main_comparison`, `t1b_ecfp4_*`, `t1d_per_target_paired_stats`), both ablation tables,
both screening funnels (`t6`, `t6b`, `d7`, `d7b`), the heatmaps, the property statistics and
their DUD-E equivalents. Independently cross-checked earlier: the KRAS G4 funnel
(10,077 → 9,278 → 717 → 1,588 → 216,092 → 9,812 → 1,721 → 785 → 728 → 293; CID 74755) and
Main Table 2 (baseline −0.077/−0.077 … full model −0.077/−0.198, −0.121 over baseline).

### Why the naive table check is invalid

Several blocks in `export_tables.py` are guarded by `if <cache>.exists():`. A missing cache
skips the block and leaves the **previous** run's file in place, which then compares equal.
`verify_tables.sh` therefore deletes the tables first. `export_tables.py` now does the same
and exits 2 naming any of 15 required tables that are missing.

---

## 2. Defects found by running the pipeline

1. **Tables written to the wrong directory.** `OUT_DIR` still used `Path(__file__).parent`
   after the restructure moved other paths to `DATA_DIR`, so tables went to
   `analysis/output/` while everything downstream read `data/benchmark/output/`. Same bug in
   `prepare_strain_energy.py`. *This is also why the first "byte-for-byte" claim was wrong —
   the check hashed one directory while the script wrote to another.*
2. **The cleanup removed load-bearing code.** `visualize_methods.py` / `dude_eval_viz.py`
   were quarantined as exploratory, but they are the only implementations of the bond-JS
   metrics read from cache. Restored as `prepare_bond_js.py` / `prepare_bond_js_dude.py`.
3. **A missing input failed silently** — partial tables with exit code 0. Now gated.
4. **Two more `__file__`-relative data paths** survived the restructure.

---

## 3. Figures: status

| Figure | Status |
|---|---|
| Main Fig. 2 | **MISMATCH** — panels b,c,e,f,g,h reproduce; a,d are the wrong chart type, order, labels and normalisation (§0) |
| Main Fig. 3 | **MISMATCH** — panel a uses the same non-matching profile renderer |
| Main Fig. 4 | PASS |
| SI Figs. 1–3 | PASS |
| SI Fig. 4 | **BLOCKED** — needs `case_study/kras_g4/fig5_panels/{c_relaxed,d_triage}.{pdf,svg}`, which no script in the repo or backup produces |
| Main Fig. 4g (ThT) | **BLOCKED** — Origin per-sheet export pending |

---

## 4. What is needed

1. **The figure script that produced the published radar panels** — the 7-metric
   retention-curve-integral radar with radial scale $r=1$ to $r=3$, panel order DUD-E first.
   Nothing in the repository or the quarantine backup implements it.
2. **`fig5_panels/{c_relaxed,d_triage}.{pdf,svg}** for Supplementary Fig. 4.
3. **The Origin ThT export** for Main Fig. 4g.
4. A decision on the label: the figure titles still read "SMRT-Net RNA" where the manuscript
   says "RNA test set".

Until (1) is supplied, Main Fig. 2 and Fig. 3 cannot be called reproducible, whatever the
harness reports.

---

## 5. Verification harness — and its limitation

```bash
bash reproduction/scripts/smoke_test.sh      # static: paths, schema, syntax
bash reproduction/scripts/verify_tables.sh   # honest table regeneration (clears first)
bash reproduction/scripts/verify_figures.sh  # runs each figure, validates the output PDF
```

**Limitation.** `verify_figures.sh` checks that a figure is produced and is a valid,
non-blank, multi-page PDF. It cannot tell whether the figure *matches the manuscript*. That
gap is exactly how a non-matching Fig. 2 was reported as PASS. Any automated check of figure
correctness needs a reference comparison against the published artwork, which this harness
does not do.
