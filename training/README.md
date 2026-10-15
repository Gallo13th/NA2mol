# training/

Every training entry point. Launchers live in `launchers/`.

| Script | Configuration |
|---|---|
| `pretrain_ed.py` | ligand-only electron-density pretraining (teacher checkpoint) |
| `finetune_full.py` | mixed-domain fine-tuning; produces the released checkpoint |
| `baseline_scratch.py` | no pretraining, no teacher, no contrastive term |
| `ablation_pretrain_only.py` | pretrained weights only |
| `ablation_pretrain_constrain.py` | pretrained weights + teacher–student alignment |
| `exped_train.py` | experimental-structures-only training variant |
| `prepare_filter_expdb.py` | removes PDB entries whose pockets resemble DUD-E test pockets |
| `prepare_crossdock_indices.py` | rebuilds CrossDocked index files |

## Configurations

The ablation pair is controlled by `--ckpt_path_teacher`:

- **Pretrain Only** — omit it. No teacher is loaded and the alignment/contrastive terms are zeroed.
- **Pretrain+Constrain** — set it to the pretrained teacher checkpoint.

Both scripts guard the load and zero the four alignment/contrastive terms, so an unset teacher yields a
valid run rather than a crash.

## Launchers

`launchers/*.sh` set the documented hyperparameters and resolve the repository root from their own
location, so they work regardless of the current directory. Set `CUDA_VISIBLE_DEVICES` to override GPU
selection.

## Known limitation

`baseline_scratch.py` calls `baseline_loss_one_step`, which is **not defined** in the released
`model/bfn4edmol.py`. Baseline retraining is therefore not reproducible as shipped; the baseline's
reported metrics are reproducible from its frozen outputs. This is a documented blocker, not a silent gap.
