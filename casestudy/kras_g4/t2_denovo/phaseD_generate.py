#!/usr/bin/env python3
"""t2-DENOVO Phase D: inpainting generation from the Phase-BC two-benzothiazole
anchor seeds (18 atoms, 2 fragments, free = 1) in the BER102 pocket.

Machinery identical to t2_repro/stepC_inpaint_gen.py (sample_by_seed +
reconstruct_with_seed, correct featurizer 13-class decode). Seeds come from the
pure de novo chain: library-isostere context -> A6 pose-scan probe endorsement ->
N->S replacement (Phase B) + generic placement (Phase C). No t2 topology,
conformation or fragment ever entered the seeds; t2 SMILES is used ONLY as the
retrieval criterion at the end.

Usage: CUDA_VISIBLE_DEVICES=1 python casestudy/kras_g4/t2_denovo/phaseD_generate.py \
          --run D1 --rounds 16
"""
import os, sys, glob, random, time, collections, json, argparse

PROJECT_ROOT = '/home/data/users/gaoletian/rna2smol'
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch

from accelerate import Accelerator

from dataloader.featurizer import GeomSeedMolOnlyPocket as _G
_orig_init = _G.__init__
def _patched_init(self, pocket, seed, mode='general', pocket_radius=None,
                  extra_range=(0, 6), remain_seed=False, **kwargs):
    _orig_init(self, pocket=pocket, seed=seed, mode=mode, pocket_radius=pocket_radius,
               extra_range=extra_range, remain_seed=remain_seed)
_G.__init__ = _patched_init

from model.bfn4edmol import BFN4SBDDScoreModel as _M
_orig_sbs = _M.sample_by_seed
def _patched_sbs(self, pocket_pos, pocket_v, batch_pocket, ed_pos, ed_v, batch_ed,
                 batch_ligand, n_nodes, fixed_atom_mask, fixed_atom_pos, fixed_atom_types,
                 sample_steps=1000, desc='', ligand_pos=None, pocket_res=None,
                 pocket_ele=None, **kwargs):
    return _orig_sbs(self, pocket_pos=pocket_pos, pocket_v=pocket_v,
                     batch_pocket=batch_pocket, ed_pos=ed_pos, ed_v=ed_v,
                     batch_ed=batch_ed, batch_ligand=batch_ligand, n_nodes=n_nodes,
                     fixed_atom_mask=fixed_atom_mask, fixed_atom_pos=fixed_atom_pos,
                     fixed_atom_types=fixed_atom_types, sample_steps=sample_steps,
                     desc=desc, ligand_pos=ligand_pos, pocket_res=pocket_res,
                     pocket_ele=pocket_ele)
_M.sample_by_seed = _patched_sbs
print('[ok] monkey-patches applied', flush=True)

from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')
from rdkit import DataStructs
from rdkit.Chem import rdFingerprintGenerator
import inspect

POCKET = 'casestudy/kras_g4/ber102_rdock/ber102_pocket.pdb'
CKPT = 'ckpt/onlypkt_ftwithrna_oncrossdocked_onlygeneralfeat_train_best_val_loss_step_111429.pth'
SEED_DIR = 'casestudy/kras_g4/t2_denovo/phaseBC_seeds'
T2_CANON = Chem.MolToSmiles(Chem.MolFromSmiles('c1ccc2sc(Cc3nc4ccccc4s3)nc2c1'))
mfpg = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

from model.bfn4edmol import Config, BFN4SBDDScoreModel
from genbypkt import load_compiled_model
from dataloader.featurizer import (GeomSeedMolOnlyPocket, ProteinLigandDataLoader,
                                   MAP_INDEX_TO_ATOM_TYPE_AROMATIC as FEAT_TAB)
from torch.utils.data import ConcatDataset
from util.mol_utils import reconstruct_from_generated_with_seed, _neutralize_or_reject
import dataloader.featurizer as _featurizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=str, default='D1')
    ap.add_argument('--rounds', type=int, default=16)
    ap.add_argument('--batch', type=int, default=100)
    ap.add_argument('--natom', type=int, default=19)
    ap.add_argument('--seed0', type=int, default=3042)
    ap.add_argument('--seeds_glob', type=str, default=SEED_DIR + '/seedBC_*.sdf')
    ap.add_argument('--dist_tol', type=float, default=0.4)
    ap.add_argument('--free_extra', type=int, default=0)
    args = ap.parse_args()
    OUTDIR = f'casestudy/kras_g4/t2_denovo/gen_run{args.run}'
    os.makedirs(OUTDIR, exist_ok=True)

    def _fixed_sample_atom_num(space_size):
        return args.natom
    _featurizer.sample_atom_num = _fixed_sample_atom_num
    print(f'[ok] sample_atom_num -> {args.natom} (fixed; free = {args.natom - 18})', flush=True)

    if args.free_extra != 6:
        import textwrap
        import dataloader.featurizer as _fmod
        _src = textwrap.dedent(inspect.getsource(_G.__getitem__))
        _new = _src.replace('total_n = max(origin_n + 6, sample_atom_num(self.pocket_size))',
                            f'total_n = max(origin_n + {int(args.free_extra)}, sample_atom_num(self.pocket_size))')
        assert f'origin_n + {int(args.free_extra)}' in _new, 'free_extra patch failed'
        _ns = dict(vars(_fmod))
        _ns['sample_atom_num'] = _fixed_sample_atom_num
        exec(_new, _ns)
        _G.__getitem__ = _ns['__getitem__']
        print(f'[ok] total_n = max(origin_n + {args.free_extra}, {args.natom})', flush=True)

    seed_files = sorted(glob.glob(args.seeds_glob))
    seed_files = [f for f in seed_files if not f.endswith('_pre.sdf')]
    assert seed_files, 'no seeds'
    print(f'[ok] {len(seed_files)} seeds', flush=True)

    args_dict = {
        'config_file': 'config.yaml', 'exp_name': f'denovo_{args.run}',
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
        'docking_mode': 'vina score', 'output_prefix': f'denovo_{args.run}',
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
        ds_by_seed[sf] = GeomSeedMolOnlyPocket(pocket=POCKET, seed=sf, mode='general',
                                               extra_range=(-8, 8), remain_seed=True)
    seed_mols = {sf: next(Chem.SDMolSupplier(sf)) for sf in seed_files}

    writer = Chem.SDWriter(f'{OUTDIR}/valid_mols.sdf')
    hits_writer = Chem.SDWriter(f'{OUTDIR}/frag_hits.sdf')
    n_frag_hits = [0]
    fail_all = collections.Counter()
    per_seed = collections.Counter()
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
            name = f'denovo{args.run}_r{r:02d}_i{i:03d}_s{seed}'
            try:
                smi = Chem.MolToSmiles(rd_mol)
            except Exception:
                fail['smiles_fail'] += 1
                continue
            if '.' in smi:
                frags = smi.split('.')
                if T2_CANON in frags:
                    rd_mol.SetProp('_Name', name)
                    hits_writer.write(rd_mol)
                    n_frag_hits[0] += 1
                    print(f'  [FRAG-HIT t2] {name} frags={smi[:90]}', flush=True)
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
            nS = sum(1 for a in neutral.GetAtoms() if a.GetSymbol() == 'S')
            ringS = sum(1 for a in neutral.GetAtoms() if a.GetSymbol() == 'S' and a.GetIsAromatic())
            tani = DataStructs.TanimotoSimilarity(mfpg.GetFingerprint(neutral),
                                                   mfpg.GetFingerprint(Chem.MolFromSmiles(T2_CANON)))
            results.append({'name': name, 'smiles': smi, 'nS': nS, 'ringS': ringS,
                            'natoms': neutral.GetNumAtoms(), 'tani_t2': round(tani, 4),
                            'seed': os.path.basename(sf)})
            per_seed[os.path.basename(sf)] += 1
            try:
                neutral.SetProp('_Name', name)
                writer.write(neutral)
            except Exception:
                fail['sdf_write_skip'] += 1
        fail_all.update(fail)
        n_exact = sum(1 for x in results if x['smiles'] == T2_CANON)
        print(f'[run{args.run} r{r:02d}] seed={seed} valid={r_valid}/{n_b} '
              f'cum={len(results)} exact_t2={n_exact} ringS_mols={sum(1 for x in results if x["ringS"] > 0)} '
              f'sample={t_sample:.1f}s recon={time.time()-t0:.1f}s el={time.time()-t_start:.0f}s '
              f'fail={dict(fail)}', flush=True)

        del batch, dataloader, batch_dataset, theta_traj, sample_traj, y_traj
        torch.cuda.empty_cache()

    writer.close()
    hits_writer.close()
    exact = [x for x in results if x['smiles'] == T2_CANON]
    with open(f'{OUTDIR}/results.json', 'w') as f:
        json.dump({'results': results, 'fail_total': dict(fail_all),
                   'per_seed_valid': dict(per_seed), 'config': vars(args),
                   'n_exact_t2': len(exact)}, f, indent=2)

    print('\n' + '=' * 60)
    print(f'run {args.run} DONE | valid {len(results)} | EXACT t2: {len(exact)} | frag-hits: {n_frag_hits[0]}')
    if exact:
        w = Chem.SDWriter(f'{OUTDIR}/t2_exact_hits.sdf')
        supp = [m for m in Chem.SDMolSupplier(f'{OUTDIR}/valid_mols.sdf', removeHs=True)]
        for x in exact:
            for m in supp:
                if m.GetProp('_Name') == x['name']:
                    w.write(m)
                    break
        w.close()
        for x in exact[:10]:
            print('  EXACT:', x['name'], x['seed'])
    by_seed = collections.Counter(x['seed'] for x in exact)
    print('exact by seed:', dict(by_seed))
    best = sorted(results, key=lambda x: -x['tani_t2'])[:8]
    print('top Tanimoto vs t2:')
    for x in best:
        print(f"  {x['tani_t2']:.3f} ringS={x['ringS']} {x['smiles'][:80]}")
    print(f'fail total: {dict(fail_all)}')
    print('=' * 60, flush=True)


if __name__ == '__main__':
    main()
