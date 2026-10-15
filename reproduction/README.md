# NA2smol paper reproduction

This directory is the canonical, paper-facing reproduction layer. It is separate from the original research notebooks so that a future reader can run a stable sequence without guessing which historical route produced a figure.

## Scope

The release is intended to reproduce every manuscript and Supplementary display item **except ligand-only pretraining that requires the external pretraining LMDB**. The evaluated benchmark archives, per-molecule metric tables, leakage/recovery audit tables, checkpoints, pocket inputs and KRAS G4 case-study files must be placed under the paths in `config/reproduction.toml`.

The manuscript's canonical evaluation data name is **RNA test set**. `SMRTnet` is the method that supplied the source evaluation set, not the dataset name.

## Layout

```
reproduction/
  config/reproduction.toml       canonical relative paths
  README.md                      this document
  scripts/                       stable entry points (added as each route is normalized)
  results/                       generated source tables and figure intermediates
  figures/                       regenerated publication figures
  manifests/                     checksums and input manifests
```


## Path conventions (verified)

Every analysis and figure script now resolves data through repository-relative paths. Machine-specific
prefixes (`/home/glt/github_official/DiffEDMol`, `/home/glt/github_official/RNALigBench`,
`/tmp/opencode/...`) and the external `RNA2smol_paper/figures` output directory have been rewritten:

- figure output -> `reproduction/figures/`
- benchmark tables -> `reproduction/results/benchmark_tables/`
- leakage audit tables -> `reproduction/results/leakage/`
- case-study manifests -> `reproduction/results/kras_g4/`
- temporary caches -> `reproduction/cache/`

`python3 reproduction/scripts/check_figure_paths.py` statically resolves every
`NAME = ROOT / "..."` declaration in the figure scripts and reports whether each one exists, separating
genuine missing inputs from directories that are created at runtime. It currently exits non-zero, and the
remaining gaps reduce to four distinct absent data files:

| Missing input | Blocks |
|---|---|
| `notebooks/output/smrtbench_combined_results.csv` | Main Fig. 2, Fig. 3, SI Fig. 1, SI Fig. 3 |
| `notebooks/output/tables/smrtbench` | the same figure scripts |
| `notebooks/output/chemspace_targeted` (needs `cache_chemspace/*.npy`) | SI Fig. 1 |
| `reproduction/data/tht/` Origin projects + their per-sheet export | case-study assay panel |

These are absent **data**, not broken paths. Every declared path now points at the right place in the
repository.

## Display-item dependencies

| Display item | Required source data | Canonical route |
|---|---|---|
| Main Fig. 1 | model/training counts and architecture constants | `manifests/fig1_workflow.yaml` (to be added) |
| Main Fig. 2 | RNA test-set and DUD-E per-molecule rows; per-target metrics; ECFP4 recovery; funnel counts; scaffold breadth | benchmark export → cross-benchmark figure |
| Main Fig. 3 | four RNA ablation configurations, per-target metrics, strain/PoseCheck, funnel counts | benchmark export → fine-grained ablation figure |
| Main Fig. 4 | BER102 generation, fragment/variant/product manifests, docking and RNAsmol votes, Tier-1/PubChem tables | `casestudy/kras_g4/denovo2/run_chain_ber102.py` and a canonical assembly wrapper |
| Main Tables 1–2 | split manifest; overlap/recovery audit tables | `manifests/` and leakage audit tables |
| Supplementary Figs. 1–3 | chemical-space, MolCRAFT mapping-control, property-landscape tables | benchmark export and audit scripts |
| Supplementary Fig. 4 | 728 relaxed products, 293 Tier-1 candidates and ranking metrics | KRAS G4 case-study source tables |
| Supplementary Tables 1–11 | configuration, recovery, scaffold subsampling, ID exclusions, modality/split, benchmark and PoseCheck tables | frozen CSV/JSON source tables in `results/` |
| Supplementary Algorithms/Notes | algorithmic specification in the manuscript and model code | source code plus config/checkpoints |

## Required input groups

1. **Training manifests**: PDB, CrossDocked2020 and nucleic-acid split/index files. The authoritative counts are PDB 45,871 train + 5,000 validation, CrossDocked 81,028 train, nucleic-acid 356 train + 382 validation; total training/validation 127,255/5,382. The large pretraining LMDB is excluded by design.
2. **Benchmark raw tables**: one row per generated molecule with target, method/configuration, SMILES, affinity, QED, SA, RMSD, strain/PoseCheck, ECFP4 recovery and scaffold identifiers. The historical raw tables are `notebooks/smrtnet_stat_all.csv` and `notebooks/dude_stat_all.csv`, but the canonical copies should be staged in `reproduction/results/benchmarks/`.
3. **RNA/DUD-E source structures and reference actives**: pocket PDB/SDF inputs, DUD-E actives/proteins, RNA test-set pocket and ligand archives.
4. **Recovery/leakage tables**: target-level recovery, exact/ECFP4/ECFP4-TS overlap, matched scaffold subsampling, ID-exclusion sensitivity and Wilson/bootstrap estimates. Existing audit outputs are under `leakage_audit/tables/`.
5. **KRAS G4**: 7X8M model 1 BER102 pocket, 10,077-generation manifest, fragment/variant/product manifests, docking/RNAsmol scores, Tier-1/PubChem/vendor tables and raw ThT data. The wet-lab protocol remains author-supplied.
6. **Model assets**: fine-tuned, baseline, Pretrain Only and Pretrain+Constrain checkpoints. The ligand-only pretraining LMDB remains external.

## Reproduction order

1. `python -m reproduction.scripts.validate_inputs` — validate paths, schemas, counts and checksums.
2. `python -m reproduction.scripts.export_benchmark_tables` — derive frozen per-target/per-method tables.
3. `python -m reproduction.scripts.run_benchmark_figures` — regenerate Main Figs. 2–3 and Supplementary Figs. 1–3.
4. `python -m reproduction.scripts.run_kras_g4` — regenerate Main Fig. 4 and Supplementary Fig. 4 from the BER102 manifests.
5. `python -m reproduction.scripts.build_manuscript` — compile the manuscript and SI.

The wrappers are deliberately named after the paper's display items. Historical exploratory notebooks remain outside this layer until their inputs and outputs have been audited.

## Known blockers

- No canonical source-table directory currently exists in the remote checkout: `notebooks/output/` is absent, while the figure scripts expect it. The raw tables and compressed `paperdata/*.tar.gz` archives are present but need staging and checksum manifests.
- Several historical plotting scripts still write to `RNA2smol_paper/`, an absolute/obsolete path relative to this repository. They must be patched to use `reproduction/figures/`.
- `assemble_cross_benchmark_figure.py` expects table names from `export_tables.py`, while the manuscript's current Fig. 2 and the README describe different historical panel layouts. The current manuscript PDF is the source of truth; do not silently reuse the old layout.
- `baseline_scratch_train.py` calls `baseline_loss_one_step`, which is not defined in the released model module; baseline retraining is therefore not reproducible until fixed. Frozen released outputs can still reproduce metrics.
- The complete BER102 figure assembly script referenced in historical documentation is not present in the remote tree; only `g4tasks/audit_v2/09_panel_kras_inhib.py` is available. Do not claim full Fig. 4 regeneration until the missing assembly route is recovered.
- ThT assay re-analysis requires the collaborator's complete protocol and raw data interpretation; the manuscript still contains one assay-protocol placeholder.

## Naming and safety rules

- Use `SMRTnet` only for the method and `RNA test set` for the evaluation data.
- Use `NA2smol` as the model name; do not create `RNA2smol`/`RNAsmol` aliases in new files.
- Do not delete raw input data or checkpoints during cleanup. First copy or move them into a clearly named `archive/` area and record the reason in a manifest.
- Do not commit pretraining LMDB or private credentials.
