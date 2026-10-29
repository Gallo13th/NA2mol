# NA2smol

Code and data layout for the NA2smol manuscript.

**NA2smol** is a pocket-conditioned 3D molecular generator built on a Bayesian Flow Network. It combines
ligand-only electron-density pretraining with a teacher–student alignment objective and a shared
physicochemical pocket encoder to generate small molecules inside nucleic-acid binding sites. The model
name in the manuscript is **NA2smol**; the repository keeps its historical name `rna2smol`.

> **Reproducing the paper?** Start with **[REPRODUCIBILITY.md](REPRODUCIBILITY.md)**. It maps every
> figure and table to the data it needs and the command that regenerates it.

---

## Quick start

```bash
bash setup_env.sh          # creates the `rna2smol` conda env from environment.yml
conda activate rna2smol    # REQUIRED: the system Python lacks torch_scatter and openbabel

# 1. check that the inputs this checkout needs are present
python3 reproduction/scripts/validate_inputs.py

# 2. confirm every figure script's data paths resolve
python3 reproduction/scripts/check_figure_paths.py

# 3. run all checks at once (fails loudly on any gap)
bash reproduction/scripts/smoke_test.sh
```

## Layout

```
model/          BFN score model and samplers
dataloader/     featurisation, datasets, atom-count tables
util/           molecule reconstruction, refinement, scoring

training/       pretraining, fine-tuning, baseline and ablation entry points
generation/     library generation from a pocket

analysis/       evaluation metrics, table export, every figure script
audit/          ligand-leakage and recovery audit
case_study/     KRAS promoter G-quadruplex (BER102) case study
reproduction/   validators, manifests and canonical entry points
```

See [REPRODUCIBILITY.md §2](REPRODUCIBILITY.md) for the annotated tree.

## The four training configurations

| Configuration | Entry point | Objective |
|---|---|---|
| Baseline | `training/baseline_scratch.py` | coordinate + type loss only, from scratch |
| Pretrain Only | `training/ablation_pretrain_only.py` | pretrained weights, no teacher |
| Pretrain+Constrain | `training/ablation_pretrain_constrain.py` | pretrained weights + teacher–student alignment |
| Full model | `training/finetune_full.py` | both paths, plus the temporal contrastive term |

`analysis/fig3_ablation.py` renders the comparison.

## Data

Bulk inputs (benchmark result tables, training LMDBs, checkpoints) are **not** in git; see
`reproduction/manifests/data_files.json`. The ligand-only pretraining archive (`data/pretrain.tar.gz`, 5.5 GB) is fetched from the authors' AutoDL storage and unpacked into `data/pretrain/merged/` with `reproduction/scripts/finalize_pretrain.sh`.

The only experimental data versioned here are the ThT assay Origin projects in `data/tht/`; see
`data/tht/README.md` for the export step the plotting code requires.

## Terminology

`SMRTnet` is the method that supplied the source evaluation set; the evaluation data is called the
**RNA test set** throughout. `NA2smol` is the model. See
[REPRODUCIBILITY.md §9](REPRODUCIBILITY.md) for the full convention.

## Known limitations

* **OpenBabel** is required for molecule reconstruction and is missing from the live environment.
- The ThT Origin export must be produced in Origin before `analysis/fig4g_tht_panel.py` can run.
- Live status and blockers: `reproduction/manifests/READINESS.md`.

## License

MIT — see [LICENSE](LICENSE). The bundled RNAsmol scoring package keeps its upstream license
(`case_study/fmn/RNAsmol/LICENSE`).
