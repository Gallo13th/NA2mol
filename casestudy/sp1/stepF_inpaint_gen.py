#!/usr/bin/env python3
"""sp1 flavone inpainting generation.

Mirrors t2_repro/stepC_inpaint_gen.py (same monkey-patches, same model/pocket
featurization, same decode/reconstruct/neutralize pipeline) with sp1 pocket +
the 4 flavone seeds. The scoring is vs. all three target flavones (BM = Murcko
or chromone substructure match; Tanimoto best-of-3).

Usage:
  CUDA_VISIBLE_DEVICES=0 python casestudy/sp1/stepF_inpaint_gen.py --run A \
      --natom_lo 20 --natom_hi 22 --rounds 12 --batch 80
  (deliberately wider --free_extra to let the model also grow/shrink size a bit)
"""
import os, sys, glob, random, time, collections, json, argparse

PROJECT_ROOT = '/home/data/users/gaoletian/rna2smol'
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch

from accelerate import Accelerator
print('[ok] imports ready', flush=True)

from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import DataStructs
from rdkit.Chem import rdFingerprintGenerator
import textwrap, inspect
import dataloader.featurizer as _featurizer

POCKET = 'casestudy/sp1/sp1_recp_pocket.pdb'
CKPT = 'ckpt/onlypkt_ftwithrna_oncrossdocked_onlygeneralfeat_train_best_val_loss_step_111429.pth'
SEED_DIR = 'casestudy/sp1/inpaint_seeds'
TARGETS = {
    'm1': Chem.MolFromSmiles('O=c1c(O)c(-c2ccc(O)cc2)oc2cc(O)cc(O)c12'),
    'm2': Chem.MolFromSmiles('O=c1c(O)c(-c2ccc(O)cc2O)oc2cc(O)cc(O)c12'),
    'm3': Chem.MolFromSmiles('O=c1c(O)c(-c2ccc(O)c(O)c2)oc2cc(O)cc(O)c12'),
}
TGT_CANON = {k: Chem.MolToSmiles(m) for k, m in TARGETS.items()}
TGT_BM = {k: Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(m)) for k, m in TARGETS.items()}
# chromone substructure = BM core (covers all three)
CHROMONE = Chem.MolFromSmarts('O=c1cc(-c2ccccc2)oc2ccccc12')
mfpg = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
print('[ok] targets:', TGT_CANON, flush=True)

from model.bfn4edmol import Config, BFN4SBDDScoreModel
from genbypkt import load_compiled_model
from dataloader.featurizer import (GeomSeedMolOnlyPocket as _G, ProteinLigandDataLoader,
                                   MAP_INDEX_TO_ATOM_TYPE_AROMATIC as FEAT_TAB)
from torch.utils.data import ConcatDataset
from util.mol_utils import reconstruct_from_generated_with_seed, _neutralize_or_reject


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=str, default='A')
    ap.add_argument('--rounds', type=int, default=12)
    ap.add_argument('--batch', type=int, default=80)
    ap.add_argument('--natom_lo', type=int, default=20)
    ap.add_argument('--natom_hi', type=int, default=22)
    ap.add_argument('--seed0', type=int, default=42)
    ap.add_argument('--dist_tol', type=float, default=0.4)
    ap.add_argument('--free_extra', type=int, default=0)
    ap.add_argument('--seeds_glob', type=str, default=SEED_DIR + '/seed_*.sdf')
    ap.add_argument('--outdir', type=str, default='')
    args = ap.parse_args()
    OUTDIR = args.outdir or f"casestudy/sp1/gen_run{args.run}"
    os.makedirs(OUTDIR, exist_ok=True)

    def _fixed_sample_atom_num(space_size):
        return random.randint(args.natom_lo, args.natom_hi)
    _featurizer.sample_atom_num = _fixed_sample_atom_num
    print(f'[ok] sample_atom_num -> randint({args.natom_lo},{args.natom_hi})', flush=True)

    if args.free_extra != 6:
        _src = textwrap.dedent(inspect.getsource(_G.__getitem__))
        _new = _src.replace('total_n = max(origin_n + 6, sample_atom_num(self.pocket_size))',
                            f'total_n = max(origin_n + {int(args.free_extra)}, sample_atom_num(self.pocket_size))')
        assert f'origin_n + {int(args.free_extra)}' in _new, 'free_extra patch failed'
        _ns = dict(vars(_featurizer))
        _ns['sample_atom_num'] = _fixed_sample_atom_num
        exec(_new, _ns)
        _G.__getitem__ = _ns['__getitem__']
        print(f'[ok] GeomSeedMolOnlyPocket.__getitem__ patched: total_n = max(origin_n + {args.free_extra}, randint)', flush=True)

    seed_files = sorted(glob.glob(args.seeds_glob))
    assert seed_files, 'no seeds'
    print(f'[ok] {len(seed_files)} seeds', flush=True)

    args_dict = {
        'config_file': 'config.yaml', 'exp_name': f'sp1_inp_{args.run}',
        'revision': 'default', 'debug': False, 'resume': False,
        'wandb_resume_id': None, 'empty_folder': False, 'test_only': False,
        'seed': args.seed0, 'no_wandb': True, 'logging_level': 'warning',
        'random_rot': False, 'pos_noise_std': 0, 'pos_normalizer': 2.0,
        'batch_size': 8, 'epochs': 15, 'v_loss_weight': 1, 'lr': 5e-4,
        'scheduler': 'plateau', 'weight_decay': 0, 'max_grad_norm': 'Q',
        'sigma1_coord': 0.03, 'beta1': 1.5, 't_min': 0.0001,
        'use_discrete_t': True, 'discrete_steps': 1000,
        'destination_prediction': True, 'sampling_strategy': 'end_back_pmf',
        'time_emb_mode': 'simple', 'time_emb_dim': 1, 'pos_init_mode': 'zero',
        'ckpt_path': 'best', 'num_samples': 5, 'sample_steps': 100,
        'sample_num_atoms': 'ref', 'visual_chain': False,
        'docking_mode': 'vina score', 'output_prefix': f'sp1_inp_{args.run}',
        'feat_mode': 'generalfeat', 'use_cavity_volume': False,
        'volume_prob_table_path': None, 'use_probabilistic': False,
        'prob_top_n': 5, 'prob_beam_width': 100,
    }
    cfg = Config(**args_dict)
    model = BFN4SBDDScoreModel(
        net_config=cfg.dynamics.net_config,
        protein_atom_feature_dim=11, ligand_atom_feature_dim=13,
        time_emb_dim=1, ed_feature_dim=4,
    )
    accelerator = Accelerator(mixed_precision='bf16', gradient_accumulation_steps=1)
    print(f'[diag] device={accelerator.device}', flush=True)
    state_dict = torch.load(CKPT, map_location='cpu')
    model.load_state_dict(load_compiled_model(model, state_dict), strict=False)
    model = accelerator.prepare(model)
    model.eval()
    unwrapped = accelerator.unwrap_model(model)
    print('[ok] model ready', flush=True)

    ds_by_seed = {}
    for sf in seed_files:
        ds_by_seed[sf] = _G(pocket=POCKET, seed=sf, mode='general',
                                               extra_range=(-8, 8), remain_seed=True)
    seed_mols = {sf: next(Chem.SDMolSupplier(sf)) for sf in seed_files}
    print('[ok] per-seed datasets built', flush=True)

    writer = Chem.SDWriter(f'{OUTDIR}/valid_mols.sdf')
    hits_writer = Chem.SDWriter(f'{OUTDIR}/bm_hits.sdf')
    n_bm_hits = [0]
    fail_all = collections.Counter()
    results = []
    t_start = time.time()

    for r in range(args.rounds):
        seed = args.seed0 + r
        random.seed(seed); np.random.seed(seed)
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

        datasets = []
        i = 0
        while len(datasets) < args.batch:
            datasets.append(ds_by_seed[seed_files[i % len(seed_files)]])
            i += 1
        batch_dataset = ConcatDataset(datasets)
        dataloader = ProteinLigandDataLoader(batch_dataset, batch_size=args.batch, shuffle=False)
        dataloader = accelerator.prepare(dataloader)
        batch = next(iter(dataloader)).to(accelerator.device)

        t0 = time.time()
        with torch.no_grad():
            theta_traj, sample_traj, y_traj, final_e, final_ei = unwrapped.sample_by_seed(
                pocket_pos=batch.pocket_pos / 2,
                pocket_v=batch.pocket_atom_feature.float(),
                batch_pocket=batch.pocket_element_batch,
                ed_pos=torch.empty((0, 3), device=accelerator.device),
                ed_v=torch.empty((0, 4), device=accelerator.device),
                batch_ed=batch.ed_ed_code_batch,
                batch_ligand=batch.mol_element_batch,
                n_nodes=args.batch,
                fixed_atom_mask=batch.mol_fixed_atom_mask,
                fixed_atom_pos=batch.mol_fixed_atom_pos / 2,
                fixed_atom_types=batch.mol_fixed_atom_types,
                sample_steps=100,
                pocket_res=None, pocket_ele=None,
                return_traj=False,
            )
        t_sample = time.time() - t0

        new_traj = (sample_traj[-1][0] * 2 + batch.mol_geom_center[:3], sample_traj[-1][1])
        xyz = new_traj[0]
        atomtype = new_traj[1]
        graph_batch_idx = batch.mol_element_batch.cpu()
        fixed_mask_all = batch.mol_fixed_atom_mask.cpu()
        seed_paths = list(batch.mol_seed_path)

        fail = collections.Counter()
        r_valid = 0
        t0 = time.time()
        n_b = int(graph_batch_idx.max().item()) + 1
        for i in range(n_b):
            mask = graph_batch_idx == i
            sub_xyz = xyz[mask].cpu().numpy()
            sub_at = atomtype[mask].argmax(-1)
            pairs = [FEAT_TAB.get(int(j.item()), (None, None)) for j in sub_at]
            sub_atomnum = [p[0] for p in pairs]
            sub_arom = [p[-1] for p in pairs]
            if any(a is None for a in sub_atomnum):
                fail['unknown_class'] += 1
                continue
            sf = seed_paths[i]
            try:
                rd_mol = reconstruct_from_generated_with_seed(
                    sub_xyz, sub_atomnum, fixed_mask_all[mask].numpy().astype(bool),
                    seed_mols[sf], sub_arom, dist_tol=args.dist_tol)
            except Exception as e:
                msg = str(e).lower()
                if 'kekul' in msg:
                    fail['kekulize'] += 1
                elif 'valen' in msg:
                    fail['valence'] += 1
                elif 'disconn' in msg:
                    fail['disconnected'] += 1
                else:
                    fail['recon_other'] += 1
                continue
            name = f'sp1run{args.run}_r{r:02d}_i{i:03d}_s{seed}'
            try:
                smi = Chem.MolToSmiles(rd_mol)
            except Exception:
                fail['smiles_fail'] += 1
                continue
            if '.' in smi:
                fail['disconnected'] += 1
                continue
            neutral = None
            try:
                neutral = _neutralize_or_reject(rd_mol)
            except Exception:
                neutral = None
            if neutral is None:
                fail['charged_reject'] += 1
                continue
            r_valid += 1
            smi = Chem.MolToSmiles(neutral)
            try:
                Chem.SanitizeMol(neutral)
            except Exception:
                fail['sanitize_fail'] += 1
                r_valid -= 1
                continue
            # scoring vs targets
            bm = None
            try:
                bm = Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(neutral))
            except Exception:
                bm = None
            is_bm = (bm is not None and bm in set(TGT_BM.values()))
            has_core = neutral.HasSubstructMatch(CHROMONE)
            tcis = {k: round(DataStructs.TanimotoSimilarity(mfpg.GetFingerprint(neutral),
                                                            mfpg.GetFingerprint(t)), 4)
                    for k, t in TARGETS.items()}
            exact = [k for k, c in TGT_CANON.items() if c == smi]
            results.append({'name': name, 'smiles': smi, 'natoms': neutral.GetNumAtoms(),
                            'bm_match': bool(is_bm), 'bm': bm,
                            'has_core': bool(has_core),
                            'tc': tcis, 'exact': exact,
                            'seed': os.path.basename(sf)})
            if is_bm or has_core or exact:
                n_bm_hits[0] += 1
                print(f'  [HIT{"-EXACT" if exact else ""}] {name} bm={is_bm} core={has_core} '
                      f'tc={tcis} {smi[:95]}', flush=True)
                try:
                    neutral.SetProp('_Name', name)
                    hits_writer.write(neutral)
                except Exception:
                    pass
            try:
                neutral.SetProp('_Name', name)
                writer.write(neutral)
            except Exception:
                fail['sdf_write_skip'] += 1
        fail_all.update(fail)
        n_exact = sum(1 for x in results if x['exact'])
        n_bm = sum(1 for x in results if x['bm_match'])
        print(f'[run{args.run} r{r:02d}] seed={seed} valid={r_valid}/{n_b} '
              f'cum={len(results)} exact={n_exact} bm={n_bm} '
              f'sample={t_sample:.1f}s recon={time.time()-t0:.1f}s el={time.time()-t_start:.0f}s '
              f'fail={dict(fail)}', flush=True)

        del batch, dataloader, batch_dataset, theta_traj, sample_traj, y_traj
        torch.cuda.empty_cache()

    writer.close()
    hits_writer.close()
    with open(f'{OUTDIR}/results.json', 'w') as f:
        json.dump({'results': results, 'fail_total': dict(fail_all),
                   'config': vars(args)}, f, indent=2)

    n_exact = [x for x in results if x['exact']]
    n_bm = [x for x in results if x['bm_match'] or x['has_core']]
    print('\n' + '=' * 60, flush=True)
    print(f'run {args.run} DONE | valid {len(results)} | exact targets: {len(n_exact)} '
          f'| BM/core hits: {len(n_bm)}', flush=True)
    if n_exact:
        for x in n_exact[:15]:
            print('  EXACT:', x['name'], x['exact'], x['seed'], flush=True)
    print('top Tanimoto (best of 3):')
    for x in sorted(results, key=lambda y: -max(y['tc'].values()))[:12]:
        print(f"  {x['tc']} bm={x['bm_match']} core={x['has_core']} {x['smiles'][:85]}", flush=True)
    print('fail total:', dict(fail_all), flush=True)
    print('=' * 60, flush=True)


if __name__ == '__main__':
    main()
