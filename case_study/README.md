# case study — KRAS promoter G-quadruplex (BER102, 7X8M model 1)

Generates and triages a focused library against the BER102 site of the KRAS promoter G-quadruplex.

## Pipeline

```bash
python kras_g4/run_generation_ber102.py        # 4 streams x 2,500 -> 10,077 unique molecules
python kras_g4/denovo2/run_chain_ber102.py     # fragment cut -> mutation -> port assembly -> triage
python kras_g4/polish_figure4.py               # Main Fig. 4
python kras_g4/assemble_extended_data_fig_kras.py   # Supplementary Fig. 4
python ../analysis/fig4g_tht_panel.py          # Main Fig. 4g (needs data/tht/export/)
```

## Funnel (verified against the manuscript)

10,077 cold-start unique → 9,278 fragments → 717 pooled → 1,588 variants → 216,092 port candidates →
9,812 assemblies → 1,721 unique products → 785 QED/SA pass → 728 resolved → **293 Tier-1**;
34 PubChem-registered, 14 purchasable, candidate CID 74755.
Source of record: `kras_g4/denovo2/paper_funnel.json`.

## Requirements

- **rDock 2013.1** — `run_chain_ber102.py` reads `RBT_ROOT` (set by `setup_env.sh`). Docking uses the
  BER102 system in `kras_g4/ber102_rdock/`; score-only parameter files in `kras_g4/rdock_r4/`.
- **smina** — `util/smina.static` is bundled.
- **RNAsmol ensemble weights** under `fmn/RNAsmol/saved_model/` (not tracked; see
  `fmn/RNAsmol/README.md`). Scoring wrappers: `fmn/rnasmol_score.py`, `fmn/rnasmol_score_gpu.py`.

The rDock receptor and cavity are reproducible from `kras_g4/rdock_protocol/` (`prepare_receptor.sh`,
`make_cavity.sh`), which also carries a redocking validation (`redock_validation.py`).

## Third-party code

`fmn/RNAsmol/` is an upstream package and keeps its own license (`fmn/RNAsmol/LICENSE`).
