# Reproducibility guide

This document maps every display item in the manuscript to the data it needs and the command that
regenerates it. It is the authoritative reference for reviewers and for anyone rebuilding the results.

Everything below is repository-relative. Nothing depends on a machine-specific path.

---

## 1. Environment

**The conda environment is required, not optional.** The system Python cannot run this project:
`model/bfn4edmol.py` imports `torch_scatter` and `torch_geometric`, and
`util/mol_utils.py::reconstruct_from_generated` imports OpenBabel. Neither is available outside the
environment, so the model and molecule reconstruction fail with `ModuleNotFoundError` there.

```bash
bash setup_env.sh            # creates the `rna2smol` env from the committed environment.yml
conda activate rna2smol
python -c "import torch, torch_scatter, torch_geometric, openbabel; print('ok')"
```

`environment.yml` is committed and pins the full specification (exported with
`conda env export --no-builds`), rather than being generated at setup time as in earlier versions.
`setup_env.sh` reads it; set `RECREATE_ENV=y` to rebuild an existing environment non-interactively.

If you are invoking scripts without activating the environment, use its interpreter directly:

```bash
/path/to/envs/rna2smol/bin/python analysis/fig2_cross_benchmark.py
```

Analyses that only need pandas/matplotlib/RDKit (`analysis/`, `audit/`) do run under a plain Python 3.9,
which is why the missing `torch_scatter` is easy to overlook — anything touching the model does not.

Verified versions at the time of writing:

| Package | Version |
|---|---|
| python | 3.9 |
| torch | 2.4.0+cu124 |
| rdkit | 2025.09.2 |
| pandas / numpy / scipy | 2.3.3 / 1.26.4 / 1.13.1 |
| matplotlib | 3.9.4 |

External binaries and packages:

* **smina** — `util/smina.static` is bundled.
* **rDock 2013.1** — set `RBT_ROOT`; required only for the case study.
* **OpenBabel** (`openbabel` Python bindings) — **required for every generated molecule.** Listed in
  `setup_env.sh`; missing from the live environment.
* **rsvg-convert** (librsvg) and **PyMuPDF** — only for Fig. 4. Panel a there is designed
  artwork (`case_study/kras_g4/assets/figure4a.svg`); rsvg-convert turns it into a PDF and
  PyMuPDF places it over the panel-a slot, keeping it vector. **pdftocairo** (poppler) then
  re-derives the png/svg/eps from the composited PDF. Without these, Fig. 4 cannot be rebuilt.
  `util/mol_utils.py::reconstruct_from_generated` is the only path from a generated atom cloud to a
  molecular graph. It builds an OpenBabel molecule, infers bonds from covalent radii, calls
  `PerceiveBondOrders`, and only then converts to RDKit. `setup_env.sh` already lists it, but it is
  **not installed in the current environment**, so `import openbabel` fails and no molecule can be
  reconstructed until it is added.

> **Rendering on a shared filesystem.** Matplotlib's font-cache build is extremely slow on network
> mounts (observed: minutes of stalled I/O). All rendering entry points pin `MPLCONFIGDIR` to local
> storage, default `/tmp/na2smol_mplconfig`, overridable with `NA2SMOL_MPLCONFIGDIR`, and limit BLAS
> threads via `OMP_NUM_THREADS`. If a plot appears to hang, this is why.

---

## 2. Repository layout

```
rna2smol/
├── README.md                  project overview
├── REPRODUCIBILITY.md         this guide
├── setup_env.sh               environment bootstrap
├── config.yaml                model/network configuration (single canonical config)
│
├── model/                     BFN score model and samplers
├── dataloader/                featurisation, datasets, atom-count tables
├── util/                      molecule reconstruction, refinement, scoring
│
├── training/                  every training entry point
│   ├── pretrain_ed.py             ligand-only electron-density pretraining
│   ├── finetune_full.py           mixed-domain fine-tuning (released checkpoint)
│   ├── baseline_scratch.py        no-pretraining baseline
│   ├── ablation_pretrain_only.py  Pretrain Only
│   ├── ablation_pretrain_constrain.py  Pretrain+Constrain
│   ├── prepare_filter_expdb.py    DUD-E pocket-similarity filter
│   ├── prepare_crossdock_indices.py
│   └── launchers/             shell launchers for the above
│
├── generation/                library generation from a pocket
├── analysis/                  evaluation metrics, tables and every figure script
├── audit/                     ligand-leakage and recovery audit
├── case_study/kras_g4/        KRAS promoter G-quadruplex (BER102) case study
├── reproduction/              validators, manifests, canonical entry points
│
├── data/                      all inputs
│   ├── benchmark/             benchmark result tables and the Origin ThT projects
│   └── ...                    training corpora (LMDB), index pickles
├── paperdata/                 archived benchmark input archives
├── ckpt/                      model checkpoints
└── results/, logs/            generated outputs
```

`analysis/` contains exactly the scripts that produce published numbers or figures. Historical
exploratory scripts are deliberately **not** in the repository.

---

## 3. Data layout and what must be supplied

| Path | Content | In git? |
|---|---|---|
| `data/benchmark/dude_stat_all.csv` | DUD-E per-molecule metrics, 438,543 rows × 5 methods | no — deposit with release |
| `data/benchmark/smrtnet_stat_all.csv` | RNA test-set per-molecule metrics | no |
| `data/benchmark/DUDE_active_compounds.csv` | DUD-E reference actives | no |
| `data/benchmark/output/smrtbench_combined_results.csv` | merged 392,887-row benchmark table, all configurations | no |
| `data/benchmark/output/tables/{smrtbench,dude}/` | derived per-target tables | no |
| `data/benchmark/output/chemspace_targeted/` | chemical-space audit tables | no |
| `data/pretrain/merged/` | ligand-only pretraining LMDB (29.5 GB `data.mdb`) | no |
| `data/tht/*.opj`, `*.opju` | authoritative ThT assay data (Origin) | yes |
| `data/tht/export/` | per-sheet CSV export of the Origin projects | yes |
| `paperdata/*.tar.gz` | benchmark input archives (pockets, references) | yes |
| `ckpt/*.pth` | released checkpoints | no — deposit with release |

Run the validator to see exactly what is missing:

```bash
python3 reproduction/scripts/validate_inputs.py
```

It exits non-zero for any gap and never silently skips.

### Pretraining corpus (verified)

`data/pretrain.tar.gz` (5,916,109,016 bytes, SHA256
`b1f834c60c6329ac3318d0a86c8c0acf2649f15ac6653c04df5211bd386a860e`) unpacks via
`reproduction/scripts/finalize_pretrain.sh` into `data/pretrain/merged/`. That LMDB holds
**2,368,986 entries**, consistent with the ~2.4 M ligand-only library cited in the manuscript.
`training/pretrain_ed.py` reads it from that path.

Run long unpacks under `tmux`: an SSH drop will otherwise kill a 20-minute extraction.

---

## 4. Full reproduction sequence

### Step 0 — prerequisites

```bash
python3 reproduction/scripts/validate_inputs.py
python3 reproduction/scripts/check_figure_paths.py   # every figure script's inputs resolve
```

### Step 1 — training (optional; released checkpoints make this skippable)

| Step | Command | Output |
|---|---|---|
| 1a. Ligand-only ED pretraining | `python training/pretrain_ed.py --config_file config.yaml` | teacher checkpoint |
| 1b. Mixed-domain fine-tuning | `python training/finetune_full.py --config_file config.yaml` | released checkpoint |
| 1c. No-pretraining baseline | `python training/baseline_scratch.py` | baseline checkpoint |
| 1d. Pretrain Only | `python training/ablation_pretrain_only.py` | ablation checkpoint |
| 1e. Pretrain+Constrain | `python training/ablation_pretrain_constrain.py` | ablation checkpoint |

Launchers in `training/launchers/` set the documented hyperparameters.

> **Ablation semantics.** `ablation_pretrain_only.py` takes `--ckpt_path_teacher`. Leave it unset for
> Pretrain Only (no teacher is loaded and the alignment/contrastive terms are zeroed);
> `ablation_pretrain_constrain.py` sets it to the pretrained teacher checkpoint.

> **Known limitation.** `training/baseline_scratch.py` calls `baseline_loss_one_step`, which is not
> defined in the released `model/bfn4edmol.py`. Baseline *retraining* is therefore not reproducible as
> shipped; the baseline's reported metrics are reproducible from its frozen outputs. This is listed as a
> blocker rather than hidden.

Data preparation, if rebuilding the corpora:

```bash
python training/prepare_filter_expdb.py      # DUD-E pocket-similarity filter (removes 2,827 of 48,698 PDB entries)
python training/prepare_crossdock_indices.py # CrossDocked index rebuild
```

### Step 2 — generation

```bash
python generation/genbypkt.py --ckpt ckpt/<checkpoint>.pth --pocket <pocket.pdb> --output_prefix <name>
```

### Step 3 — evaluation and tables

```bash
python analysis/prepare_benchmark_table.py   # per-molecule metrics for the ablation configurations
python analysis/export_tables.py             # every published table, from the merged benchmark CSV
```

`export_tables.py` is the single source of all numerical tables. Run it after any change to
`prepare_*`, because it rewrites the CSVs. It clears its output directory first and exits
non-zero if any of 15 required tables is missing — a missing cache used to produce a partial
table set silently.

**Bond-JS caches.** The radar tables need two caches that are not in git:

```bash
python3 reproduction/scripts/build_bond_js_caches.py   # ~10 min, RDKit 3D embedding
```

Without `cache_dude_bond_js.pkl`, `d8b`/`d8c` are not written and Main Fig. 2 fails.

**Known numerical caveat.** The `bond_len_js`/`bond_angle_js` metrics are derived from 3D
conformer embedding and are not stable across RDKit versions. Rank normalisation amplifies
this into a rank swap between NA2smol (baseline) and ED2Mol in the radar panel. Record the
RDKit version used for the paper; expect small radar differences on other versions.

### Step 4 — figures

See §5.

---

## 5. Display item → command

| Display item | Command | Primary inputs |
|---|---|---|
| **Main Fig. 1** workflow | *not script-generated* | `reproduction/assets/main_workflow_rev2.pdf` (hand-laid-out) |
| **Main Fig. 2** paired RNA/DUD-E benchmark | `python analysis/fig2_cross_benchmark.py` | merged benchmark CSV, `output/tables/{smrtbench,dude}/`, `chemspace_targeted/scaffold_breadth_table.csv` |
| **Main Fig. 3** four-configuration ablation | `python analysis/fig3_ablation.py` | merged benchmark CSV (must contain all four configurations) |
| **Main Fig. 4** KRAS G4 case study | `python case_study/kras_g4/polish_figure4.py` | `case_study/kras_g4/denovo2/*.json`, `g4_tier1_order_table.csv` |
| **Main Fig. 4g** ThT panel | `python analysis/fig4g_tht_panel.py` | `data/tht/export/*_sheet2.csv` |
| **Main Table 1** training composition | `python reproduction/scripts/export_available_sources.py` | split manifests |
| **Main Table 2** recovery sensitivity | *see §6* | `audit/tables/recovery_manuscript_authoritative.csv` |
| **SI Fig. 1** chemical-space audit | `python analysis/prepare_chemical_space.py && python analysis/sfig1_chemical_space.py` | `output/chemspace_targeted/`, `cache_chemspace/*.npy` |
| **SI Fig. 2** MolCRAFT mapping control | `python reproduction/scripts/render_mapping_control.py` | `dude_stat_all.csv` |
| **SI Fig. 3** property landscape | `python analysis/sfig3_property_landscape.py` | merged benchmark CSV + `dude_stat_all.csv` |
| **SI Fig. 4** KRAS ranking | `python case_study/kras_g4/assemble_extended_data_fig_kras.py` | `case_study/kras_g4/denovo2/*.json` |
| **SI Tables 1–11** | `python analysis/export_tables.py` + `audit/` | see §6 |

All figures are written to `reproduction/figures/` as PDF + PNG (SVG/EPS/TIFF optional).

---

## 6. Audit tables

The leakage/recovery audit produces the overlap and recovery tables behind Main Table 2 and
Supplementary Tables 2, 4, 10 and 11.

```bash
cd audit/scripts && bash run_chain.sh     # full chain
```

Individual steps: `exp1_reference_overlap.py` (reference-vs-training overlap),
`exp2*` (generated-molecule overlap), `exp3_recovery_sensitivity.py` (analogue-filter sensitivity,
the main result), `exp4_fixed_target_sensitivity.py`, `exp5_ablation_recovery.py` (per-configuration),
then `make_supp_recovery_table.py` to emit the LaTeX table.

**Independently verified.** `audit/tables/recovery_manuscript_authoritative.csv` reproduces Main Table 2
exactly: recovery change under the fine-tuning and pretraining analogue filters is baseline
−0.077/−0.077, Pretrain Only −0.017/−0.014, Pretrain+Constrain −0.016/−0.099, full model −0.077/−0.198,
with the full model's pretraining-filter effect exceeding the baseline by −0.121; eligible-target counts
24/18/19/19/26/14 (RNA test set) and 84/81 (DUD-E).

---

## 7. Case study (KRAS promoter G-quadruplex, BER102, 7X8M)

```bash
python case_study/kras_g4/run_generation_ber102.py       # 4 streams x 2,500 -> 10,077 unique
python case_study/kras_g4/denovo2/run_chain_ber102.py    # fragment chain -> Tier-1
```

Requires rDock 2013.1 (`RBT_ROOT`) and the RNAsmol ensemble weights under
`case_study/fmn/RNAsmol/saved_model/` (not tracked; see `case_study/fmn/RNAsmol/README.md`).

**Independently verified.** `case_study/kras_g4/denovo2/paper_funnel.json` reproduces every funnel
number in the Fig. 4 caption and Methods: 10,077 cold-start unique → 9,278 fragments → 717 pooled →
1,588 variants → 216,092 port candidates → 9,812 assemblies → 1,721 unique products → 785 QED/SA pass →
728 resolved → 293 Tier-1; 34 PubChem-registered, 14 purchasable, candidate CID 74755.

### ThT assay data

The authoritative experimental data are the two Origin projects in `data/tht/`. Origin's binary formats
cannot be read by Python, so the plotting code consumes a per-sheet CSV export:

```bash
python3 reproduction/scripts/check_tht_origin.py     # reports what is present and validates the schema
```

`data/tht/README.md` gives the export procedure. The validator checks each export file for the schema the
panel script reads (`skiprows=1`, `S1 / R1` at column index 4, ≥3 numeric wells) and exits non-zero until
the export is complete.

> `荧光数据汇总.csv` is **not** used. It aggregates every sheet into one table with an extra leading
> `文件` column, which shifts the column index and silently reads the wrong values.

---

## 8. Verification

The results of actually re-running the pipeline are in
[`reproduction/manifests/VALIDATION.md`](reproduction/manifests/VALIDATION.md): 40 of 44
tables regenerate byte-identically, 6 of 7 figures pass, and four defects that static
checks could not see were found and fixed.

```bash
bash reproduction/scripts/smoke_test.sh      # static: paths, schema, syntax
bash reproduction/scripts/verify_tables.sh   # honest table regeneration (clears first)
bash reproduction/scripts/verify_figures.sh  # run every figure, validate the output
```

Three checks, each of which fails loudly:

1. `validate_inputs.py` — every declared prerequisite exists and has the expected schema/methods.
2. `check_figure_paths.py` — every `ROOT / "..."` data path declared in a figure script resolves;
   distinguishes genuinely missing inputs from runtime-created output directories.
3. `py_compile` over every script.

Expected output when the release is complete: all checks pass. Any failure names the exact missing path.

Additional manifests under `reproduction/manifests/`: `inputs.json` (what each display item needs),
`data_files.json` (bulk inputs with SHA256), `display_items.yaml` (display-item inventory),
`READINESS.md` (current status and blockers).

---

## 9. Terminology

- **`SMRTnet`** is a *method* (Fei et al., Nat Biotechnol 2026, `10.1038/s41587-025-02942-z`).
- **`RNA test set`** is the evaluation *data*, assembled from SMRTnet's evaluation set. The two must not
  be conflated in text, figure labels or variable names.
- **`NA2smol`** is the model. Historical data values still say `RNA2smol`; `reproduction/scripts/normalize_results.py`
  adds `*_canonical` columns mapping legacy labels to manuscript names without rewriting source values.
- External baselines keep their published names (`MolCRAFT`, `Pocket2Mol`, `ED2Mol`).

---

## 10. Current status

See `reproduction/manifests/READINESS.md` for the live status table. In brief: Main Figs. 2–4 and
Supplementary Figs. 1–3 are reproducible from the data in this repository (Supplementary Fig. 2 and the
Fig. 4 funnel have been regenerated end-to-end and match the manuscript). Outstanding items are the ThT
Origin export and the two training-side blockers noted in §4.
