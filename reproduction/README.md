# reproduction/

Validators, manifests and canonical entry points. **The full guide is
[../REPRODUCIBILITY.md](../REPRODUCIBILITY.md)**; this file covers only what lives in this directory.

## Layout

```
config/reproduction.toml     canonical path declarations
manifests/
  inputs.json                what each display item requires
  data_files.json            bulk inputs (not in git) with SHA256
  display_items.yaml         display-item inventory
  READINESS.md               live status and blockers
  naming_normalization.json  legacy -> canonical label mapping
  quarantine.json            what was moved out of the repository and why
results/                     staged source tables and figure intermediates
figures/                     regenerated publication figures
assets/                      non-regenerable art (Main Fig. 1)
data/tht/                    authoritative ThT Origin projects
scripts/                     see scripts/README.md
```

## Validators

| Command | Checks |
|---|---|
| `scripts/validate_inputs.py` | every prerequisite in `manifests/inputs.json` exists with the expected schema and methods; exits non-zero on any gap |
| `scripts/check_figure_paths.py` | resolves every `ROOT / "..."` data path declared in a figure script; separates missing inputs from runtime-created outputs |
| `scripts/smoke_test.sh` | all of the above plus `py_compile` over every script |

## Support scripts

| Script | Purpose |
|---|---|
| `validate_inputs.py` | prerequisite validator (supports `--items` and `--report`) |
| `check_figure_paths.py` | figure-script data-path resolver |
| `check_tht_origin.py` | validates the Origin per-sheet export schema for the ThT panels |
| `export_available_sources.py` | copies audited tables to canonical paths, SHA256-verified |
| `export_dude_tables.py` | rebuilds DUD-E recovery and funnel tables from per-molecule rows |
| normalize_results.py | adds `*_canonical` columns mapping legacy labels to manuscript names (lossless) |
| `manifest_data_files.py` | records bulk inputs and checksums |
| `render_mapping_control.py` | Supplementary Fig. 2 route |
| `relativize_paths.py` | one-off: rewrites machine-specific paths to repository-relative |

## Rules

- New code uses repository-relative paths only. No `/home/...` or `/tmp/...`.
- Run `smoke_test.sh` before claiming a reproduction step works.
- Never report success when a prerequisite is absent: the validators are designed to fail loudly.
- `normalize_results.py` maps labels without rewriting source values, so the audit trail stays intact.
