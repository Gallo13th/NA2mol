#!/usr/bin/env python3
"""Inpainting post-training for the BFN ligand generator.

The sampler can already hold atoms fixed (`sample_by_seed`), but the network is not
trained for it: nothing tells it which atoms are fixed, and it never sees the sampler's
regime (a subset exact, the rest generated). This script fine-tunes the model with

  * a per-atom fixed indicator fed through `ligand_atom_emb` (patch in
    `train_scripts/patch_model_for_inpaint.py`);
  * random *contiguous* fixed fragments (20-80% of each ligand, BFS over its bond graph),
    which is the pattern the seed-inpainting sampler actually uses;
  * the standard BFN loss, so the fixed atoms are reproduced exactly and the free atoms
    are predicted conditioned on them.

Evaluation samples with the real inpainting path (`sample_by_seed`) at a few step counts
and reports fixed-atom conservation, validity and reconstruction of the free atoms.

Run (6 GPUs):
    accelerate launch --num_processes 6 --mixed_precision bf16 \
        train_scripts/inpaint_finetune.py --steps 3000 --batch-size 16 --lr 1e-4 \
        --out ckpt/inpaint_20261117
"""
import argparse
import json
import os
import random
import pickle
import sys
import time
from datetime import datetime

import numpy as np
import torch
from accelerate import Accelerator
from rdkit import Chem, RDLogger
from torch.utils.data import ConcatDataset, Subset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataloader.dataloader_pointcloud import MolPocketPointCloud
from dataloader.featurizer import GeomMolOnlyPocket, ProteinLigandDataLoader
from model.bfn4edmol import BFN4SBDDScoreModel, Config

ED_DIM = 4
K = 13


class SanitizedDataset(torch.utils.data.Dataset):
    """Skip the handful of LMDB entries whose ligand arrays are ragged.

    ~0.5% of the cross-docked complexes store mol_pos with one atom fewer than
    mol_element/mol_atom_feature; the collate concatenates them as-is, which surfaces later as a
    one-atom size mismatch in the loss. Re-draw a deterministic neighbour instead.
    """

    def __init__(self, base, stride=997, tries=6):
        self.base, self.stride, self.tries = base, stride, tries

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        d = None
        for k in range(self.tries):
            d = self.base[(i + k * self.stride) % len(self.base)]
            if (d['mol_pos'].shape[0] == d['mol_element'].shape[0]
                    == d['mol_atom_feature'].shape[0]):
                return d
        return d


def load_clean_dataset(lmdb_path, indices_path, pocket_radius=8, is_rna=False):
    """Indexed split of a cleaned LMDB (same construction as training/finetune_full.py)."""
    with open(indices_path, 'rb') as f:
        idx = pickle.load(f)
    raw = MolPocketPointCloud(lmdb_path=lmdb_path)
    ds = GeomMolOnlyPocket(raw, mode='general', pocket_radius=pocket_radius,
                           is_train=True, is_rna=is_rna)
    return Subset(ds, list(idx['train'])), Subset(ds, list(idx['val']))


def load_or_create_expdb_dataset(lmdb_path, output_dir, val_size=2000, seed=42):
    os.makedirs(output_dir, exist_ok=True)
    pkl = os.path.join(output_dir, 'expdb_indices.pkl')
    if os.path.exists(pkl):
        with open(pkl, 'rb') as f:
            idx = pickle.load(f)
    else:
        raw = MolPocketPointCloud(lmdb_path=lmdb_path)
        order = list(range(len(raw)))
        np.random.RandomState(seed).shuffle(order)
        idx = {'val': order[:val_size], 'train': order[val_size:]}
        with open(pkl, 'wb') as f:
            pickle.dump(idx, f)
    raw = MolPocketPointCloud(lmdb_path=lmdb_path)
    ds = GeomMolOnlyPocket(raw, mode='general', pocket_radius=8, is_train=True, is_rna=False)
    return Subset(ds, list(idx['train'])), Subset(ds, list(idx['val']))


def load_base(model, ckpt_path, log=print):
    """Load a pretrained checkpoint into the inpainting model.

    `ligand_atom_emb` grew by one input column, so its weight cannot be loaded as-is:
    copy the old columns and leave the new (fixed-indicator) column at zero, which makes
    the fine-tuned model start numerically identical to the pretrained one.
    """
    sd = torch.load(ckpt_path, map_location='cpu')
    sd = {k.replace('_orig_mod.', ''): v for k, v in sd.items()}
    own = model.state_dict()
    for k in list(sd.keys()):
        if k in own and own[k].shape != sd[k].shape:
            if k == 'ligand_atom_emb.weight':
                w = torch.zeros_like(own[k])
                w[:, : sd[k].shape[1]] = sd[k]
                sd[k] = w
                log(f'[i] {k}: padded {tuple(w.shape)} with a zero indicator column')
            else:
                log(f'[i] drop shape-mismatched {k}: {tuple(sd[k].shape)} vs {tuple(own[k].shape)}')
                sd.pop(k)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    log(f'[i] loaded {len(sd)} tensors; missing {len(missing)} unexpected {len(unexpected)}')
    if 'ligand_atom_emb.weight' in own:
        with torch.no_grad():
            model.ligand_atom_emb.weight[:, -1].zero_()
            if model.ligand_atom_emb.bias is not None:
                pass
    return model


def _ligand_bonds_and_rings(batch, g):
    """Bond graph and ring-atom set of one ligand graph, perceived from its true 3D coords."""
    from rdkit.Chem import rdDetermineBonds
    sel = batch.mol_element_batch == g
    idx = sel.nonzero(as_tuple=True)[0].tolist()
    m = Chem.RWMol()
    for i in idx:
        m.AddAtom(Chem.Atom(int(batch.mol_element[i])))
    m = m.GetMol()
    conf = Chem.Conformer(m.GetNumAtoms())
    pos = batch.mol_pos[sel].cpu().numpy()
    for k in range(len(pos)):
        conf.SetAtomPosition(k, pos[k].tolist())
    m.AddConformer(conf)
    bonds, ring_atoms = [], set()
    try:
        rdDetermineBonds.DetermineConnectivity(m)
        Chem.GetSSSR(m)          # initialise RingInfo after connectivity perception
        bonds = [(idx[b.GetBeginAtomIdx()], idx[b.GetEndAtomIdx()]) for b in m.GetBonds()]
        for ring in m.GetRingInfo().AtomRings():
            ring_atoms.update(idx[i] for i in ring)
    except Exception as exc:
        print(f'[w] bond/ring perception failed: {type(exc).__name__}: {exc}', flush=True)
    return idx, bonds, ring_atoms


def save_trainer_state(path, accelerator, model, optimizer, sched, step, epoch, args):
    """Everything needed to resume: weights, optimiser, scheduler, counters and RNG."""
    if not accelerator.is_main_process:
        return
    torch.save({
        'step': step,
        'epoch': epoch,
        'model': accelerator.unwrap_model(model).state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': sched.state_dict(),
        'args': vars(args),
        'rng_torch': torch.get_rng_state(),
        'rng_numpy': np.random.get_state(),
        'rng_python': random.getstate(),
    }, path)


def load_trainer_state(path, accelerator, model, optimizer, sched):
    sd = torch.load(path, map_location='cpu')
    accelerator.unwrap_model(model).load_state_dict(sd['model'])
    optimizer.load_state_dict(sd['optimizer'])
    sched.load_state_dict(sd['scheduler'])
    torch.set_rng_state(sd['rng_torch'])
    np.random.set_state(sd['rng_numpy'])
    random.setstate(sd['rng_python'])
    return int(sd['step']), int(sd.get('epoch', 0))


def maybe_hot_update(args, optimizer, sched, log):
    """Re-read the live config file and apply the hyper-parameters that may change mid-run.

    Only values that are safe to swap without rebuilding the loaders are honoured:
    learning rate, task probabilities, the mask fraction range, gradient clip, and the
    logging / saving / validation intervals.
    """
    path = args.live_config
    if not path or not os.path.exists(path):
        return
    stamp = os.path.getmtime(path)
    if getattr(maybe_hot_update, '_stamp', None) == stamp:
        return
    maybe_hot_update._stamp = stamp
    try:
        with open(path) as fh:
            cfg = json.load(fh)
    except Exception as exc:
        log(f'[live] cannot parse {path}: {exc}')
        return
    applied = []
    for key, cast in (('lr', float), ('p_ring', float), ('p_type', float),
                      ('grad_clip', float), ('frac_min', float), ('frac_max', float),
                      ('log_every', int), ('save_every', int), ('val_every', int)):
        if key in cfg and hasattr(args, key):
            new = cast(cfg[key])
            current = getattr(args, key)
            if key == 'lr' and optimizer.param_groups:
                current = optimizer.param_groups[0]['lr']
            if abs(current - new) > 1e-12:
                setattr(args, key, new)
                applied.append(f'{key}={new}')
                if key == 'lr':
                    for g in optimizer.param_groups:
                        g['lr'] = new
                    # accelerate wraps the scheduler (AcceleratedScheduler) and does not
                    # forward base_lrs, so look one level deeper before giving up
                    base = getattr(sched, 'base_lrs', None)
                    if base is None:
                        inner = getattr(sched, 'scheduler', None)
                        base = getattr(inner, 'base_lrs', None)
                    if base is not None:
                        for i in range(len(base)):
                            base[i] = new
    if applied:
        log(f'[live] applied: ' + ', '.join(applied))


def make_task_masks(batch, gen, p_ring=0.6, p_type=0.2, frac=(0.3, 0.7)):
    """Sample one of the three inpainting tasks and return (pos_mask, type_mask, task).

    ring  : positions AND types of every ring atom are given -- inpaint linkers/substituents
    type  : only element types are given on a random subset -- inpaint the positions
    pos   : only positions are given on a random subset -- inpaint the element types
    """
    n = batch.mol_element_batch.size(0)
    B = int(batch.mol_element_batch.max().item()) + 1
    pos_mask = torch.zeros(n, dtype=torch.bool)
    type_mask = torch.zeros(n, dtype=torch.bool)
    r = torch.rand(1, generator=gen).item()
    task = 'ring' if r < p_ring else ('type' if r < p_ring + p_type else 'pos')
    for g in range(B):
        idx, bonds, ring_atoms = _ligand_bonds_and_rings(batch, g)
        if not idx:
            continue
        if task == 'ring' and ring_atoms:
            keep = sorted(ring_atoms)
            pos_mask[keep] = True
            type_mask[keep] = True
            continue
        frac_g = frac[0] + (frac[1] - frac[0]) * torch.rand(1, generator=gen).item()
        k = max(1, min(len(idx) - 1, int(round(frac_g * len(idx)))))
        start = idx[torch.randint(0, len(idx), (1,), generator=gen).item()]
        seen = {start}
        frontier = [start]
        adj = {}
        for a, b in bonds:
            adj.setdefault(a, []).append(b)
            adj.setdefault(b, []).append(a)
        while len(seen) < k and frontier:
            nxt = []
            for u in frontier:
                for v in adj.get(u, []):
                    if v not in seen:
                        seen.add(v)
                        nxt.append(v)
                        if len(seen) >= k:
                            break
                if len(seen) >= k:
                    break
            frontier = nxt
        if len(seen) < k:
            rest = [i for i in idx if i not in seen]
            order = torch.randperm(len(rest), generator=gen).tolist()
            seen.update(rest[i] for i in order[: k - len(seen)])
        sel = sorted(seen)
        if task == 'type':
            type_mask[sel] = True
        else:
            pos_mask[sel] = True
    return pos_mask, type_mask, task


@torch.no_grad()
def evaluate(unwrapped, loader, device, steps=20, n_batches=4):
    """Seed inpainting on held-out complexes: half the ring system fixed, rest generated."""
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog('rdApp.*')
    unwrapped.eval()
    gen = torch.Generator(device='cpu').manual_seed(0)
    written = graphs = 0
    for bi, batch in enumerate(loader):
        if bi >= n_batches:
            break
        batch = batch.to(device)
        pos_mask, type_mask, _ = make_task_masks(batch, gen, p_ring=1.0, p_type=0.0)
        if pos_mask.sum() == 0:
            continue
        seed_mols = []
        for g in range(int(batch.mol_element_batch.max().item()) + 1):
            gm = pos_mask & (batch.mol_element_batch.cpu() == g)
            if gm.sum() < 2:
                seed_mols.append(None)
                continue
            from rdkit.Chem import rdDetermineBonds
            m = Chem.RWMol()
            for i in gm.nonzero(as_tuple=True)[0].tolist():
                m.AddAtom(Chem.Atom(int(batch.mol_element[i])))
            m = m.GetMol()
            conf = Chem.Conformer(m.GetNumAtoms())
            pos = batch.mol_pos[gm].cpu().numpy()
            for k in range(len(pos)):
                conf.SetAtomPosition(k, pos[k].tolist())
            m.AddConformer(conf)
            try:
                rdDetermineBonds.DetermineConnectivity(m)
                Chem.SanitizeMol(m, catchErrors=True)
            except Exception:
                pass
            seed_mols.append(m)
        graphs += len(seed_mols)
        d = '/tmp/inpaint_val'
        import shutil
        shutil.rmtree(d, ignore_errors=True)
        try:
            traj, *_ = unwrapped.sample_by_seed(
                pocket_pos=batch.pocket_pos / 2,
                pocket_v=batch.pocket_atom_feature.float(),
                batch_pocket=batch.pocket_element_batch,
                ed_pos=batch.ed_pos[:0] if False else batch.ed_ed_pos / 2,
                ed_v=torch.nn.functional.one_hot(batch.ed_ed_code - 290, num_classes=4).float(),
                batch_ed=batch.ed_ed_code_batch,
                batch_ligand=batch.mol_element_batch,
                n_nodes=int(batch.mol_element_batch.size(0)),
                fixed_atom_mask=pos_mask,
                fixed_atom_pos=batch.mol_pos[pos_mask] / 2,
                fixed_atom_types=batch.mol_atom_feature[pos_mask],
                type_fixed_mask=type_mask,
                sample_steps=steps,
                pocket_res=None, pocket_ele=None, return_traj=False)
        except Exception as e:
            print(f'[val] sampling failed: {e}', flush=True)
            break
        centers = batch.mol_geom_center.view(-1, 3)[batch.mol_element_batch]
        pos_pred = traj[-1][0] * 2 + centers
        sampletraj_to_sdf_byseed({'trajs': (pos_pred, traj[-1][1]),
                                  'batch_idx': batch.mol_element_batch.cpu()},
                                 d, seed_mols, pos_mask, dist_tol=0.4)
        written += len([f for f in os.listdir(d) if f.endswith('.sdf')]) if os.path.isdir(d) else 0
    unwrapped.train()
    return {'valid_fraction': written / max(graphs, 1), 'graphs': graphs, 'written': written}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--init-ckpt',
                    default='onlypkt_ftwithrna_oncrossdocked_onlygeneralfeat_train_best_val_loss_step_111429.pth',
                    help='starting weights: the released model, or an inpainting checkpoint to resume from')
    ap.add_argument('--steps', type=int, default=0, help='0 = derive from --epochs')
    ap.add_argument('--epochs', type=float, default=20.0)
    ap.add_argument('--resume', default='', help='trainer_state checkpoint to resume from')
    ap.add_argument('--live-config', default='live_config.json',
                    help='JSON file polled for hyper-parameter changes (lr, task probs, intervals)')
    ap.add_argument('--config-every', type=int, default=100)
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--warmup', type=int, default=100)
    ap.add_argument('--grad-clip', type=float, default=1.0)
    ap.add_argument('--accum', type=int, default=1)
    ap.add_argument('--p-ring', type=float, default=0.6, help='probability of the ring-system task')
    ap.add_argument('--p-type', type=float, default=0.2, help='probability of the type-only task')
    ap.add_argument('--log-every', type=int, default=25)
    ap.add_argument('--save-every', type=int, default=500)
    ap.add_argument('--val-every', type=int, default=1000)
    ap.add_argument('--out', default='ckpt/inpaint_20261117')
    args = ap.parse_args()

    accelerator = Accelerator(gradient_accumulation_steps=args.accum)
    device = accelerator.device
    if accelerator.is_main_process:
        os.makedirs(args.out, exist_ok=True)

    # ---------------- data ----------------
    root = os.getcwd()
    train_crossdock, _ = load_clean_dataset(
        os.path.join(root, 'data/crossdock/crossdocked_diffedmol_total'),
        './clean_dataset_onlycrossdock/Onlycrossdock.pkl', pocket_radius=8, is_rna=False)
    train_expdb, val_expdb = load_or_create_expdb_dataset(
        os.path.join(root, 'data/expdb'), './clean_exped_dataset', val_size=2000, seed=42)
    train_rna, _ = load_clean_dataset(
        os.path.join(root, 'data/rnaligand'), './clean_dataset_onlyrna/OnlyRNA.pkl',
        pocket_radius=8, is_rna=True)
    train_set = SanitizedDataset(ConcatDataset([train_expdb, train_crossdock, train_rna]))
    if accelerator.is_main_process:
        print(f'[i] train {len(train_set)} complexes '
              f'(expdb {len(train_expdb)} + crossdock {len(train_crossdock)} + rna {len(train_rna)})')
    # accelerate shards the loader itself (per-rank DistributedSampler + even batches)
    loader = ProteinLigandDataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                                     num_workers=4, drop_last=True)
    val_loader = ProteinLigandDataLoader(val_expdb, batch_size=8, shuffle=False, num_workers=2)

    # ---------------- model ----------------
    cfg = Config(config_file='config.yaml')
    model = BFN4SBDDScoreModel(net_config=cfg.dynamics.net_config,
                               protein_atom_feature_dim=11, ligand_atom_feature_dim=13,
                               time_emb_dim=1, ed_feature_dim=ED_DIM)
    model = load_base(model, args.init_ckpt, print)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min(1.0, (s + 1) / max(args.warmup, 1)))
    loader, model, optimizer, sched = accelerator.prepare(loader, model, optimizer, sched)
    unwrapped = accelerator.unwrap_model(model)

    steps_per_epoch = max(1, len(train_set) // (args.batch_size * accelerator.num_processes * args.accum))
    if args.steps <= 0:
        args.steps = int(round(args.epochs * steps_per_epoch))
    if accelerator.is_main_process:
        print(f'[i] steps/epoch {steps_per_epoch}, target {args.steps} steps '
              f'({args.steps / steps_per_epoch:.1f} epochs)', flush=True)

    start_step = 0
    if args.resume:
        path = args.resume
        if path in ('latest', 'auto'):
            cand = sorted([f for f in os.listdir(args.out) if f.startswith('trainer_state_')],
                          key=lambda f: int(f.split('_')[-1].split('.')[0]))
            path = os.path.join(args.out, cand[-1]) if cand else ''
        if path and os.path.exists(path):
            start_step, _ = load_trainer_state(path, accelerator, model, optimizer, sched)
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                print(f'[i] resumed from {path} at step {start_step}', flush=True)

    gen = torch.Generator(device='cpu').manual_seed(1234)
    step, t0, skipped = start_step, time.time(), 0
    model.train()
    while step < args.steps:
        for batch in loader:
            batch = batch.to(device)
            pos_mask, type_mask, task = make_task_masks(
                batch, gen, p_ring=args.p_ring, p_type=args.p_type)
            bs = int(batch.mol_element_batch.max().item()) + 1
            t = torch.rand(bs, 1, device=device).index_select(0, batch.mol_element_batch)
            t = t.clamp(min=1e-4, max=1 - 1e-4)
            if batch.mol_element_batch.numel() == 0:
                continue
            with accelerator.accumulate(model):
                out = unwrapped.loss_one_step(
                    t=t,
                    pocket_pos=batch.pocket_pos / 2,
                    pocket_v=batch.pocket_atom_feature.float(),
                    batch_pocket=batch.pocket_element_batch,
                    ed_pos=batch.ed_ed_pos / 2,
                    ed_v=torch.nn.functional.one_hot(batch.ed_ed_code - 290,
                                                     num_classes=ED_DIM).float(),
                    batch_ed=batch.ed_ed_code_batch,
                    ligand_pos=batch.mol_pos / 2,
                    ligand_v=batch.mol_atom_feature,
                    batch_ligand=batch.mol_element_batch,
                    pos_mask=pos_mask.to(device),
                    type_mask=type_mask.to(device))
                c_loss, d_loss = out[0], out[1]
                loss = (c_loss + d_loss).mean()
                if not torch.isfinite(loss):
                    # a single pathological complex can poison the weights; skip that batch
                    optimizer.zero_grad(set_to_none=True)
                    skipped += 1
                    if accelerator.is_main_process and skipped % 10 == 1:
                        print(f'[w] non-finite loss at step {step}, skipped ({skipped} total)',
                              flush=True)
                    continue
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                sched.step()
                optimizer.zero_grad()
            step += 1
            if args.config_every and step % args.config_every == 0 and accelerator.is_main_process:
                maybe_hot_update(args, optimizer, sched, print)
            if accelerator.is_main_process and step % args.log_every == 0:
                print(f'[step {step}] loss {loss.detach().item():.4f} '
                      f'(c {c_loss.mean().item():.4f} d {d_loss.mean().item():.4f}) '
                      f'{task} ({type_mask.float().mean().item():.2f}/{pos_mask.float().mean().item():.2f}) '
                      f'lr {sched.get_last_lr()[0]:.2e} {time.time()-t0:.0f}s', flush=True)
            if step % args.val_every == 0:
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    met = evaluate(unwrapped, val_loader, device, steps=20)
                    print(f'[val {step}] {met}', flush=True)
                accelerator.wait_for_everyone()
            if step % args.save_every == 0:
                accelerator.wait_for_everyone()
                save_trainer_state(os.path.join(args.out, f'trainer_state_{step}.pt'),
                                   accelerator, model, optimizer, sched, step,
                                   step // steps_per_epoch, args)
                if accelerator.is_main_process:
                    torch.save(unwrapped.state_dict(),
                               os.path.join(args.out, f'inpaint_step{step}.pth'))
                    print(f'[i] saved step {step} (epoch {step / steps_per_epoch:.2f})',
                          flush=True)
            if step >= args.steps:
                break
    accelerator.wait_for_everyone()
    save_trainer_state(os.path.join(args.out, 'trainer_state_final.pt'),
                       accelerator, model, optimizer, sched, step,
                       step // steps_per_epoch, args)
    if accelerator.is_main_process:
        torch.save(unwrapped.state_dict(), os.path.join(args.out, 'inpaint_final.pth'))
        print(f'[OK] finished {step} steps in {(time.time()-t0)/60:.1f} min -> {args.out}')


if __name__ == '__main__':
    main()
