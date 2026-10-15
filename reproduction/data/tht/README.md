# ThT competition assay — authoritative data

## The data files

The authoritative experimental data for the KRAS G-quadruplex ThT competition assay are the two
OriginLab projects:

| File | Content |
|---|---|
| `KRAS-THT-Compatative.opj` | KRAS 100 nM + ThT 100 nM competition series |
| `THT-Compatative.opju` | KRAS 1 µM + ThT 1 µM competition series |

**`荧光数据汇总.csv` is not used.** It aggregates every sheet into a single table with an extra leading
`文件` column, so the column offsets do not line up with what the panel script reads. It was the wrong
file for this purpose; do not reintroduce it.

## Why an export step exists

Origin's `.opj` / `.opju` formats are proprietary binary and cannot be read by the Python pipeline. The
plotting code therefore consumes a **per-sheet CSV export** placed in `export/`:

```
reproduction/data/tht/export/{PREF}_ex{N}_sheet2.csv
```

`PREF` is `THT` (experiments `ex0`–`ex5`) or `10X` (`ex0`–`ex4`). Sheet 2 is the equilibrated second
reading; the 1 µM first reading is dominated by ThT aggregation and is deliberately not used.

Each file must satisfy:

- `skiprows=1, header=0` — one title row, then a header row
- **column index 4 contains `S1 / R1`**
- at least 3 numeric wells in that column

## Export procedure (Origin GUI)

1. Open the project in OriginLab.
2. Select the workbook for experiment `ex{N}`, sheet 2.
3. `File → Export → ASCII` (or right-click the worksheet → `Export`).
4. Save as `export/{PREF}_ex{N}_sheet2.csv`.
5. Repeat for every experiment listed above.

## Verify

```bash
python3 reproduction/scripts/check_tht_origin.py
```

This reports which projects are present, which export files exist, and whether each matches the schema
above. It exits non-zero until the export is complete, so it can gate the figure build. It never invents
or substitutes values.

Once the export validates, the panels are produced by:

```bash
python3 analysis/fig4g_tht_panel.py
```

which writes `panel_kras_tht_inhib100nM.{pdf,svg,png}` and `panel_kras_tht_inhib1uM.{pdf,svg,png}`.

## Status

The Origin projects must be copied into this directory (they are small: ~1 MB and ~62 KB). The export
files are not yet present, so the ThT panels cannot be regenerated from this checkout alone.
