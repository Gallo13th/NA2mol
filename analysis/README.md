# analysis/

Evaluation metrics, table export and every figure script. These are the scripts that produce published
numbers and display items; historical exploratory code is not kept here.

## Scripts

| Script | Produces |
|---|---|
| `export_tables.py` | **all** numerical tables behind the figures, from the merged benchmark CSV |
| `prepare_benchmark_table.py` | per-molecule metrics for the four ablation configurations |
| `prepare_benchmark_table_ar.py` | metrics for the AR variant |
| `prepare_strain_energy.py` | PoseCheck strain energy for the main output set |
| `prepare_strain_energy_baseline.py` | strain energy for the baseline output set |
| `prepare_chemical_space.py` | chemical-space audit inputs (reference overlap, ligand-space bias, diversity) |
| `fig2_cross_benchmark.py` | **Main Fig. 2** — paired RNA test set / DUD-E benchmark |
| `fig3_ablation.py` | **Main Fig. 3** — four-configuration ablation |
| `fig4g_tht_panel.py` | **Main Fig. 4g** — ThT fluorescence competition panels |
| `sfig1_chemical_space.py` | **Supplementary Fig. 1** — chemical-space audit |
| `sfig2_mapping_control.py` | **Supplementary Fig. 2** — MolCRAFT random-mapping control |
| `sfig3_property_landscape.py` | **Supplementary Fig. 3** — QED/SA and docking/strain landscapes |
| `style.py` | shared publication style (colours, fonts, panel labels, `save_all`) |

## Order

```bash
python prepare_benchmark_table.py     # per-molecule metrics
python export_tables.py               # every derived table  (rewrites CSVs)
python prepare_chemical_space.py      # SI Fig. 1 inputs
python fig2_cross_benchmark.py
python fig3_ablation.py
python sfig1_chemical_space.py
python sfig2_mapping_control.py
python sfig3_property_landscape.py
python fig4g_tht_panel.py             # needs data/tht/export/ first
```

`export_tables.py` must run after any `prepare_*` script, because it rewrites the CSVs those scripts read.

## Conventions

- All data paths are repository-relative (`ROOT = Path(__file__).resolve().parents[1]`). No absolute paths.
- Data lives under `data/benchmark/`; figures are written to `reproduction/figures/` by every figure
  script, alongside a copy in the script's own output directory.
- `style.py` is the single source of plot styling; import from it rather than redefining colours.
- The canonical evaluation data name is **RNA test set**; legacy tables may still label it `SMRT-Net`, and
  `reproduction/scripts/normalize_results.py` maps those labels without altering source values.
- Colors are the Wong colour-blind-safe palette, so series remain distinguishable in print.

## Verification

```bash
python3 ../reproduction/scripts/check_figure_paths.py
```

resolves every declared data path in these scripts and fails if an input is missing.
