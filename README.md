# NA2smol

Code and data layout for the NA2smol paper release. NA2smol is a pocket-conditioned 3D molecular generator based on a Bayesian Flow Network, with ligand-only electron-density pretraining and teacher–student alignment.

## Paper-facing reproduction layer

Start with [`reproduction/README.md`](reproduction/README.md). The canonical entry points, relative paths, display-item manifest and readiness ledger are there. They are deliberately separate from the historical exploratory notebooks.

```bash
python3 reproduction/scripts/validate_inputs.py
```

At the current checkout this validator is expected to report missing inputs: the canonical merged benchmark table, chemical-space cache, current KRAS figure assembly sources, ThT raw data and Fig. 1 editable asset still need to be staged. The validator reports the exact missing paths. The ligand-only pretraining LMDB is intentionally excluded and will be uploaded separately.

## Repository layout

```
reproduction/                 paper-facing paths, manifests and validators
model/                        BFN model implementation
dataloader/                  featurisation, atom-count and data utilities
util/                         molecule reconstruction/refinement/scoring
train_scripts/                pretraining, fine-tuning and ablation entry points
scripts/                      training launchers
paperdata/                    benchmark input archives
notebooks/                    historical metric exports and figure-analysis scripts
leakage_audit/                recovery/leakage audit code and tables
casestudy/kras_g4/            BER102 case-study inputs, manifests and protocol
```

`notebooks/` and `leakage_audit/` contain historical provenance needed to close source-table gaps; do not delete their raw tables until the readiness ledger says the corresponding canonical tables are staged. Old case-study routes have been quarantined outside the repository, not deleted.

## Key facts

- `SMRTnet` is the method; the evaluation data is always named **RNA test set**.
- Fine-tuning: PDB 45,871 train + 5,000 validation; CrossDocked2020 81,028 train; nucleic-acid 356 train + 382 validation. Total train/validation records: 127,255/5,382.
- The pretraining LMDB is not part of this release by design.
- The release checkpoints are large binary inputs; do not silently replace them with newly trained checkpoints.
- The baseline training script still calls an undefined `baseline_loss_one_step`; this is documented as a reproduction blocker rather than hidden.

## Environment

```bash
source setup_env.sh
conda activate rna2smol
```

The model and generation scripts remain available as `train_scripts/*.py` and `genbypkt.py`. The exact benchmark and figure commands are being normalized under `reproduction/scripts/`; no command is advertised as complete until its source tables and output are validated.

## Data and privacy

Do not commit private credentials, the pretraining LMDB before its intended upload, or unrelated historical case-study data. Large staged inputs should have checksums in `reproduction/manifests/`.
