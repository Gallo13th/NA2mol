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
screening funnel from `data/benchmark/dude_stat_all.csv` (438,543 rows, 5 methods) with a recorded log.


`reproduction/figures/molcraft_mapping_control.pdf` was regenerated end-to-end from
`data/benchmark/dude_stat_all.csv` by `reproduction/scripts/render_mapping_control.py`. The run produced the
100 paired DUD-E targets, all eight metrics, ECFP4 recovery, the four-tier funnel and the paired
bootstrap/Wilcoxon-BH effect table that Supplementary Fig. 2 and its source tables require. This is
Supplementary Fig. 2 reproduced from repository data, not a copy of a published asset.

## Stage status

| Stage | Status | Evidence / next action |
|---|---|---|
| Main Fig. 1 | PASS | Adopted asset versioned at `reproduction/assets/main_workflow_rev2.pdf` (hand-laid-out, not script-generated). |
| Main Fig. 2 | PASS | Merged benchmark table (392,887 rows, all four configurations + 3 baselines) and both `output/tables/{smrtbench,dude}/` present; all paths resolve. |
| Main Fig. 3 | PASS | All four configurations present in the merged table; `t2_ablation_comparison.csv` and `t8_ablation_radar_profile.csv` exported. |
| Main Fig. 4 | PARTIAL | Source funnel verified (above); the final assembly script is absent from the checkout. |
| Main Tables 1–2 | VERIFIED (Table 2) / PARTIAL (Table 1) | Table 2 reproduced; Table 1 needs a frozen split manifest. |
| Supplementary Fig. 2 | **VERIFIED** | Rendered from repository data by `render_mapping_control.py`; 100 paired targets, 8 metrics, bootstrap CIs and BH-corrected Wilcoxon reproduced. Outputs under `reproduction/figures/`. |
| Supplementary Figs. 1, 3 | PASS | `chemspace_targeted/` staged (11 files incl. `scaffold_breadth_table.csv`); property-landscape inputs resolved. |
| Supplementary Fig. 4 | PARTIAL | KRAS ranking manifests present; assembly script absent. |
| Supplementary Tables 1–11 | PARTIAL | Freeze one CSV/JSON source per table under `reproduction/results/`. |
| Training / inference | PASS | Model code, configs and checkpoints present; pretraining LMDB unpacked (2,368,986 entries). The loss ladder is complete: `baseline_loss_one_step` (4-tuple), `loss_one_step` (5), `constrain_loss_one_step` (9), and the four configurations differ by which auxiliary objective is active. Blocker now OpenBabel only. |
| Assay panel | BLOCKED | Complete collaborator protocol and raw-data path absent; manuscript still carries one placeholder. |

## Required uploads to close reproduction

1. Merged RNA benchmark table (`data/benchmark/output/smrtbench_combined_results.csv`) plus
   `output/tables/{smrtbench,dude}/`, or a deterministic command that rebuilds it from the `paperdata/`
   SDF archives. The legacy `data/benchmark/smrtnet_stat_all.csv` holds only four methods and does not include
   the ablation configurations.
2. `data/benchmark/output/cache_chemspace/{smiles_bkg_na,smiles_bkg_protein,bkg_na_fps,bkg_protein_fps}.npy`
   and `output/chemspace_targeted/*.csv` (Supplementary Fig. 1).
3. The current Main Fig. 4 and Supplementary Fig. 4 assembly scripts.
4. The two Origin ThT projects (copied into `reproduction/data/tht/`) plus their
   per-sheet export, and the completed assay protocol. `荧光数据汇总.csv` is not used: it
   aggregates all sheets with an extra leading column, so the script reads the wrong one.
5. Final Fig. 1 editable source, or an explicit statement that it is not regenerable.

Until these are uploaded, legacy notebooks and raw archives must be retained: they are the only
provenance for the missing tables.

## Environment notes for reproduction

Matplotlib's font-cache build is extremely slow on the shared `<data-root>` mount (observed: minutes of
stalled I/O with `wchan=ertp_qos_handle_new_event`). All rendering entry points therefore pin
`MPLCONFIGDIR` to local storage (default `/tmp/na2smol_mplconfig`, override with `NA2SMOL_MPLCONFIGDIR`)
and limit BLAS threads via `OMP_NUM_THREADS`. Render with the environment set, or the first plot of a
session will appear to hang.

Large CSV reads on this mount are also slow; `data/benchmark/dude_stat_all.csv` is 121 MB and takes about a
minute to read.
