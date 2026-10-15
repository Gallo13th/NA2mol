# Reproduction readiness ledger

Honest state of the paper-reproduction cleanup. Statuses:

- **VERIFIED** — staged source data reproduces the manuscript values, checked numerically.
- **PASS** — source present and staged; figure not yet re-rendered from it.
- **BLOCKED** — a required input or script is not in the checkout.
- **PARTIAL** — some inputs present, provenance not closed.

## Verified against the manuscript

`reproduction/results/kras_g4/paper_funnel.json` reproduces every funnel number in the Main Fig. 4
caption and Methods: 10,077 cold-start unique → 9,278 fragments → 717 pooled → 1,588 variants →
216,092 port candidates → 9,812 assemblies → 1,721 unique products → 785 QED/SA pass → 728 resolved →
293 Tier-1; 34 PubChem-registered, 14 purchasable, candidate CID 74755.

`reproduction/results/leakage/recovery_manuscript_authoritative.csv` reproduces Main Table 2 and
Supplementary Table 11 exactly. Recovery change under the fine-tuning and pretraining analogue filters:
baseline −0.077/−0.077, Pretrain Only −0.017/−0.014, Pretrain+Constrain −0.016/−0.099, full model
−0.077/−0.198; the full model's pretraining-filter effect exceeds the baseline by −0.121. Eligible-target
counts 24/18/19/19/26/14 (RNA test set) and 84/81 (DUD-E) also match.

`reproduction/results/benchmark_tables/dude/` regenerates DUD-E ECFP4 recovery and the four-tier
screening funnel from `notebooks/dude_stat_all.csv` (438,543 rows, 5 methods) with a recorded log.


`reproduction/figures/molcraft_mapping_control.pdf` was regenerated end-to-end from
`notebooks/dude_stat_all.csv` by `reproduction/scripts/render_mapping_control.py`. The run produced the
100 paired DUD-E targets, all eight metrics, ECFP4 recovery, the four-tier funnel and the paired
bootstrap/Wilcoxon-BH effect table that Supplementary Fig. 2 and its source tables require. This is
Supplementary Fig. 2 reproduced from repository data, not a copy of a published asset.

## Stage status

| Stage | Status | Evidence / next action |
|---|---|---|
| Main Fig. 1 | BLOCKED | Workflow figure lives in the paper package, not this repo. Requires an editable source decision. |
| Main Fig. 2 | BLOCKED | Needs the merged RNA/DUD-E benchmark export; the figure scripts' retention-integral radar is not what the legacy DUD-E radar exports compute. |
| Main Fig. 3 | BLOCKED | Same merged export; the four ablation configurations must be present as methods in it. |
| Main Fig. 4 | PARTIAL | Source funnel verified (above); the final assembly script is absent from the checkout. |
| Main Tables 1–2 | VERIFIED (Table 2) / PARTIAL (Table 1) | Table 2 reproduced; Table 1 needs a frozen split manifest. |
| Supplementary Fig. 2 | **VERIFIED** | Rendered from repository data by `render_mapping_control.py`; 100 paired targets, 8 metrics, bootstrap CIs and BH-corrected Wilcoxon reproduced. Outputs under `reproduction/figures/`. |
| Supplementary Figs. 1, 3 | BLOCKED | Chemical-space reference cache (`cache_chemspace/*.npy`) absent. |
| Supplementary Fig. 4 | PARTIAL | KRAS ranking manifests present; assembly script absent. |
| Supplementary Tables 1–11 | PARTIAL | Freeze one CSV/JSON source per table under `reproduction/results/`. |
| Training / inference | PARTIAL | Model code, configs and checkpoints present; pretraining LMDB excluded by design. `baseline_loss_one_step` is undefined, so baseline retraining is not reproducible. |
| Assay panel | BLOCKED | Complete collaborator protocol and raw-data path absent; manuscript still carries one placeholder. |

## Required uploads to close reproduction

1. Merged RNA benchmark table (`notebooks/output/smrtbench_combined_results.csv`) plus
   `output/tables/{smrtbench,dude}/`, or a deterministic command that rebuilds it from the `paperdata/`
   SDF archives. The legacy `notebooks/smrtnet_stat_all.csv` holds only four methods and does not include
   the ablation configurations.
2. `notebooks/output/cache_chemspace/{smiles_bkg_na,smiles_bkg_protein,bkg_na_fps,bkg_protein_fps}.npy`
   and `output/chemspace_targeted/*.csv` (Supplementary Fig. 1).
3. The current Main Fig. 4 and Supplementary Fig. 4 assembly scripts.
4. ThT raw data and the completed assay protocol.
5. Final Fig. 1 editable source, or an explicit statement that it is not regenerable.

Until these are uploaded, legacy notebooks and raw archives must be retained: they are the only
provenance for the missing tables.

## Environment notes for reproduction

Matplotlib's font-cache build is extremely slow on the shared `/home/data` mount (observed: minutes of
stalled I/O with `wchan=ertp_qos_handle_new_event`). All rendering entry points therefore pin
`MPLCONFIGDIR` to local storage (default `/tmp/na2smol_mplconfig`, override with `NA2SMOL_MPLCONFIGDIR`)
and limit BLAS threads via `OMP_NUM_THREADS`. Render with the environment set, or the first plot of a
session will appear to hang.

Large CSV reads on this mount are also slow; `notebooks/dude_stat_all.csv` is 121 MB and takes about a
minute to read.
