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

## 0b. Radar panels reimplemented (16 October 2026)

The gap in section 0 has been closed. `analysis/radar.py` and
`analysis/radar_profiles.py` now implement the manuscript's scheme exactly, and
`fig2_cross_benchmark.py` / `fig3_ablation.py` draw it for their profile panels.

**Verified against the manuscript, RNA test set** (the panels used by Fig. 2d and 3a):

| method | observed recovery | manuscript |
|---|---|---|
| NA2smol (full) | 0.2176 | 0.2175 |
| NA2smol (baseline) | 0.1407 | 0.141 |
| Pretrain Only | 0.0346 | 0.034 |
| Pretrain+Constrain | 0.1316 | 0.132 |

Universe: 27 targets, matching "27 shared evaluable targets". The DUD-E side reports
101 targets with reference actives, matching the manuscript's "fixed universe of 101".

**Implementation notes.**

* The retention integral has an exact closed form — for one molecule it is
  ``clip((u-v)/(u-l), 0, 1)`` (lower-is-better) or ``clip((v-l)/(u-l), 0, 1)``
  (higher-is-better) — so ``s_m`` is a mean over molecules with no numerical
  integration. ``radar.integral_form`` evaluates the manuscript's integral directly
  and the two agree to 1e-6 across six distributions and the edge cases.
* Strain uses log10 over [1, 5]; non-finite or non-positive strain scores 0 (such a
  molecule fails every threshold). TPSA = 0 gives SEI = 0 rather than +inf.
* ECFP4 similarity is computed vectorised from bit matrices; verified bit-identical
  to the per-molecule RDKit reference.
* `pdb_id` must be read as ``str``: ``1e66`` and ``3e37`` are otherwise parsed as
  scientific notation and stop matching the actives file.
* DUD-E uses ``DUDE_active_compounds.csv`` — the manuscript's reference set (101
  targets, 38,759 unique actives after dedup) — not the per-row ``active_smi`` in
  ``dude_stat_all.csv``, which holds only 42–332 unique actives per target.

**Residual discrepancy, DUD-E (open).** Observed recoveries computed here differ from
the manuscript's `A_original_full`:

| method | computed | manuscript | delta |
|---|---|---|---|
| NA2smol | 0.656 | 0.6918 | 0.036 |
| MolCRAFT | 0.331 | 0.3452 | 0.014 |
| Pocket2Mol | 0.417 | 0.5023 | 0.085 |
| ed2mol | 0.483 | 0.5138 | 0.031 |

Averaging over the targets each method actually produced output for instead of the
fixed 101-target universe brings three of the four much closer (MolCRAFT 0.3443 vs
0.3452, Pocket2Mol 0.5008 vs 0.5023, ed2mol 0.5126 vs 0.5138 — all within 0.0015),
which suggests the manuscript's DUD-E figures were computed on a per-method eligible
set. NA2smol remains off by 0.016 under either convention. The written Methods
specify a fixed universe with zero for missing output, so that convention is what
the code implements; the remaining gap most likely reflects a different DUD-E
molecule snapshot for that method. It cannot be closed from the artifacts in this
checkout, and it is recorded rather than smoothed over.

Consequence for the DUD-E radar: because the shuffled null is high for the external
generators (e.g. MolCRAFT null 0.483 vs observed 0.331), their ECFP4 score clips to
0. If the manuscript's per-method eligible-set convention is adopted, those scores
would be small but non-zero.

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

---

## 7. OpenBabel is a hard dependency and is not installed

`util/mol_utils.py::reconstruct_from_generated` is the only route from a generated atom cloud to a
molecular graph, and it requires the OpenBabel Python bindings:

```
build OpenBabel molecule (all atoms temporarily carbon, uniform covalent radii)
  -> connect_the_dots: bond if d(i,j) < 1.3 * (r_cov(i) + r_cov(j))
     max valence = min(OpenBabel max bonds, RDKit default); sulfone exception (S with >=2 O -> 6)
     delete bonds between two max-valence-1 atoms
  -> AddPolarHydrogens, PerceiveBondOrders        (OpenBabel assigns bond orders)
  -> restore original element types
  -> _adjust_heteroatom_charges, _neutralize_or_reject
  -> postprocess_rd_mol_1 / postprocess_rd_mol_2, RDKit sanitisation
```

`setup_env.sh` already lists `openbabel`, but `import openbabel` currently fails in this environment
(`ModuleNotFoundError`). Molecules already generated and stored in the benchmark tables are unaffected,
but **no new molecule can be reconstructed until it is installed** in the live environment.

This is also why the Methods previously said only "standard bond inference": the routine spans OpenBabel
and RDKit and is not a one-line call. The Methods now name `reconstruct_from_generated`, the 1.3 covalent
factor, the valence caps and the OpenBabel bond-order perception.
