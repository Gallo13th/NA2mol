# audit/

Ligand-leakage and reference-recovery audit. Produces the overlap and recovery tables behind Main Table 2
and Supplementary Tables 2, 4, 10 and 11.

## Chain

```bash
cd scripts && bash run_chain.sh
```

| Stage | Scripts | Purpose |
|---|---|---|
| build | `build_sources.py`, `build_lib.py`, `build_index.py`, `build_reference_actives.py`, `build_generated_table.py`, `build_provenance.py` | assemble fingerprints, reference actives and generated-molecule provenance |
| experiments | `exp1_reference_overlap.py`, `exp2_summary.py`, `exp2a_generated_ft_overlap.py`, `exp2b_generated_pretrain_sample.py`, `exp3_recovery_sensitivity.py`, `exp4_fixed_target_sensitivity.py`, `exp5_ablation_recovery.py` | overlap and recovery analyses |
| report | `summarize_reference.py`, `make_supp_recovery_table.py`, `sync_manuscript_recovery.py`, `assemble_document.py` | summaries and the LaTeX recovery table |

`lka_common.py` holds shared identity/standardisation utilities and the audit paths; `target_ids.py`
validates join keys between tables.

## Independently verified

`tables/recovery_manuscript_authoritative.csv` reproduces Main Table 2 exactly. Recovery change under the
fine-tuning and pretraining analogue filters:

| Configuration | FT filter | PT filter |
|---|---|---|
| Baseline | −0.077 | −0.077 |
| Pretrain Only | −0.017 | −0.014 |
| Pretrain+Constrain | −0.016 | −0.099 |
| Full model | −0.077 | −0.198 |

The full model's pretraining-filter effect exceeds the baseline by −0.121. Eligible-target counts are
24/18/19/19/26/14 for the RNA test set and 84/81 for DUD-E.

## Note

The benchmark label in these tables is `SMRT-Net`; the manuscript calls the data the **RNA test set**.
`reproduction/scripts/normalize_results.py` adds `benchmark_canonical` columns mapping the legacy label
without rewriting source values.
