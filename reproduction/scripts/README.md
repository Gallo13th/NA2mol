# Paper-facing canonical entry points

Historical scripts under `data/benchmark/` and `audit/` are the provenance layer. New work must go
through these entry points, use repository-relative paths, and write only under `reproduction/results/`
or `reproduction/figures/`.

## Runnable now

| Command | What it does | Verified |
|---|---|---|
| `python3 reproduction/scripts/validate_inputs.py` | Checks every staged prerequisite by manifest; exits non-zero on any gap. `--items fig2 fig3` limits scope. `--report PATH` writes JSON. | yes |
| `python3 reproduction/scripts/export_available_sources.py` | Copies audited tables into canonical paths with SHA256 verification. | yes (12 artifacts) |
| `python3 reproduction/scripts/export_dude_tables.py` | Rebuilds DUD-E ECFP4 recovery and the four-tier screening funnel from per-molecule rows. | yes (438,543 rows) |
| `python3 reproduction/scripts/normalize_results.py` | Adds `*_canonical` columns mapping legacy labels to manuscript names. Lossless. | yes |
| `python3 reproduction/scripts/render_mapping_control.py` | Supplementary Fig. 2 route. `--tables-only` skips rendering. | render pending |

`normalize_results.py` must run after the export scripts, because the exports rewrite the CSVs.

## Not yet runnable

| Command | Display items | Blocker |
|---|---|---|
| `export_benchmark_tables` | Main Fig. 2–3, SI Tables | merged RNA benchmark export absent |
| `run_benchmark_figures` | Main Fig. 2–3, SI Fig. 1, 3 | same, plus chemical-space cache |
| `run_kras_g4` | Main Fig. 4, SI Fig. 4 | assembly script absent |
| `build_manuscript` | manuscript/SI | manuscript source lives in the paper package |

## Naming rules for new files

- `SMRTnet` = the method. `RNA test set` = the evaluation data. Never use one for the other.
- `NA2smol` = the model. Legacy data values say `RNA2smol`; they are mapped by `normalize_results.py`,
  not silently rewritten.
- External baselines keep their published names (`MolCRAFT`, `Pocket2Mol`, `ED2Mol`), except where case
  variants of the same tool exist (`ed2mol` → `ED2Mol`).
- Prefer paper-facing names over pipeline codenames: `render_mapping_control.py`, not
  `analysis/sfig2_mapping_control.py`.
- No absolute `/home/...` or `/tmp/...` paths in new code.
