#!/usr/bin/env python3
"""Re-probe ambiguous S atoms from the ablation dataset.

For each molecule containing an ambiguous S atom (decoding-table collision
between idx9=aromatic-P and idx10=aliphatic-S, further confounded by the
legacy neutrality/aromaticity fallbacks), we fix the pocket coordinates, lock
all unambiguous atoms to their certain element classes, and run a masked
probe at the ambiguous position to read the model's true 13-class
distribution.

Protocol identical to the validated probes (t=1, belief=softmax(beta1*onehot)
for locked atoms, uniform 1/13 for the masked site, coords/2, generalfeat).
"""
import os, sys, json, glob
sys.path.insert(0, "/home/data/users/gaoletian/rna2smol")
os.chdir("/home/data/users/gaoletian/rna2smol")

import argparse
import numpy as np
import torch
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from accelerate import Accelerator
from model.bfn4edmol import Config, BFN4SBDDScoreModel
from genbypkt import load_compiled_model
from dataloader.featurizer import GeomSeedMolOnlyPocket as G, ProteinLigandDataLoader
from dataloader.featurizer import GeomSeedMolOnlyPocket

_orig_init = G.__init__
G.__init__ = lambda self, *a, **k: _orig_init(
    self, *a, **{x: y for x, y in k.items()
                 if x in ('pocket', 'seed', 'mode', 'pocket_radius',
                          'extra_range', 'remain_seed')})

# featurizer (training-truth) 13-class table
from dataloader.featurizer import MAP_ATOM_TYPE_AROMATIC_TO_INDEX as TRUE_TABLE
IDX2CLASS = {v: k for k, v in TRUE_TABLE.items()}
CLASS_NAMES = {0:'H',1:'C',2:'C*',3:'N',4:'N*',5:'O',6:'O*',7:'F',8:'P',9:'P*',10:'S',11:'S*',12:'Cl'}

TARGETS = [
    # (sdf, tag, ambiguous-atom SMILES-level marker)
    ("/tmp/opencode/fix_ablation/ablation_pretrain_only/8d2b_B_QAX_ligand/8d2b_B_QAX_ligand.sdf", "8d2b_B_QAX", "sulfonium C=C/S"),
    ("/tmp/opencode/fix_ablation/ablation_pretrain_constrain_only/3bnq_C_PAR_ligand/3bnq_C_PAR_ligand.sdf", "3bnq_C_PAR", "thioether"),
    ("/tmp/opencode/fix_ablation/ablation_pretrain_constrain_only/3irw_R_C2E_ligand/3irw_R_C2E_ligand.sdf", "3irw_C2E_con", "sulfonium"),
    ("/tmp/opencode/fix_ablation/baseline_merged/6wzr_A_UG1_ligand.sdf", "6wzr_UG1", "sulfonic O3"),
    ("/tmp/opencode/fix_ablation/baseline_merged/6wzs_B_UG4_ligand.sdf", "6wzs_UG4", "sulfonic O3"),
]

pockets = {
    "8d2b": "/tmp/opencode/fix_ablation/ablation_pretrain_only/8d2b_B_QAX_ligand",  # conf sdf in same dir
    "3bnq": "/tmp/opencode/fix_ablation/ablation_pretrain_constrain_only/3bnq_C_PAR_ligand",
    "3irw": "/tmp/opencode/fix_ablation/ablation_pretrain_constrain_only/3irw_R_C2E_ligand",
    "6wzr": "/tmp/opencode/fix_ablation/baseline_merged",
    "6wzs": "/tmp/opencode/fix_ablation/baseline_merged",
}


def find_pocket_for(tag):
    """Ablation runs targeted RNA pockets; the original pocket PDBs are in
    paperdata/smrtnet_pocket_pdbs_clean.rar (not yet extracted). Fallback:
    any pocket PDB next to the ligand; else use BER102 pocket as generic
    probe context (element preference is mostly intramolecular+local)."""
    key = tag.split('_')[0]
    d = pockets.get(key, "")
    cands = glob.glob(os.path.join(d, "*pocket*.pdb")) if d else []
    if cands:
        return cands[0]
    # try global smrtnet pocket dir
    cands = glob.glob(f"/tmp/opencode/fix_ablation/**/{key}*", recursive=True)
    for c in cands:
        if c.endswith('.pdb'):
            return c
    return "casestudy/kras_g4/ber102_rdock/ber102_pocket.pdb"


def lock_belief(mol, amb_idx):
    """Build one-hot beliefs for all atoms except amb_idx (uniform)."""
    n = mol.GetNumAtoms()
    theta = torch.full((n, 13), 1.0 / 13)
    for a in mol.GetAtoms():
        i = a.GetIdx()
        if i == amb_idx:
            continue
        key = (a.GetAtomicNum(), a.GetIsAromatic())
        if key not in TRUE_TABLE:
            key = (a.GetAtomicNum(), False)
            if key not in TRUE_TABLE:
                key = (6, False)  # last-resort: treat as C
        onehot = torch.zeros(13); onehot[TRUE_TABLE[key]] = 1.0
        theta[i] = torch.softmax(1.5 * torch.log(onehot + 1e-9), dim=0) * 0.98 + 0.02/13
        # simpler: near-onehot
        theta[i] = onehot * 0.98 + 0.02 / 13
    return theta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', default='3')
    ap.add_argument('--out', default='/tmp/opencode/fix_ablation/ambig_reprobe.json')
    args = ap.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    args_dict = {'config_file': 'config.yaml', 'exp_name': 't', 'revision': 'r', 'debug': True,
                 'resume': False, 'wandb_resume_id': None, 'empty_folder': False, 'test_only': True,
                 'seed': 42, 'no_wandb': True, 'logging_level': 'warning', 'random_rot': False,
                 'pos_noise_std': 0, 'pos_normalizer': 2.0, 'batch_size': 8, 'epochs': 15,
                 'v_loss_weight': 1, 'lr': 5e-4, 'scheduler': 'plateau', 'weight_decay': 0,
                 'max_grad_norm': 'Q', 'sigma1_coord': 0.03, 'beta1': 1.5, 't_min': 0.0001,
                 'use_discrete_t': True, 'discrete_steps': 1000, 'destination_prediction': True,
                 'sampling_strategy': 'end_back_pmf', 'time_emb_mode': 'simple',
                 'time_emb_dim': 1, 'pos_init_mode': 'zero', 'ckpt_path': 'best',
                 'num_samples': 5, 'sample_steps': 100, 'sample_num_atoms': 'ref',
                 'visual_chain': False, 'docking_mode': 'vina_score'}
    cfg = Config(**args_dict)
    model = BFN4SBDDScoreModel(net_config=cfg.dynamics.net_config,
                               protein_atom_feature_dim=11,
                               ligand_atom_feature_dim=13, time_emb_dim=1,
                               ed_feature_dim=4)
    accelerator = Accelerator(mixed_precision='bf16')
    sd = torch.load('ckpt/onlypkt_ftwithrna_oncrossdocked_onlygeneralfeat_train_best_val_loss_step_111429.pth',
                    map_location='cpu')
    model.load_state_dict(load_compiled_model(model, sd), strict=False)
    model = accelerator.prepare(model)
    model.eval()
    dev = accelerator.device
    sigma1 = torch.tensor(0.03, dtype=torch.float32, device=dev)
    beta1 = 1.5

    results = {}
    for sdf, tag, note in TARGETS:
        m = Chem.SDMolSupplier(sdf, removeHs=True)[0]
        if m is None or not m.GetNumConformers():
            results[tag] = {"error": "parse/noconf"}; continue
        amb_atoms = [a.GetIdx() for a in m.GetAtoms() if a.GetSymbol() == 'S']
        if not amb_atoms:
            results[tag] = {"error": "no S"}; continue
        pocket = find_pocket_for(tag)
        # write mol to temp sdf for featurizer
        tmp = f'/tmp/opencode/fix_ablation/_tmp_{tag}.sdf'
        w = Chem.SDWriter(tmp); w.write(m); w.close()
        try:
            ds = GeomSeedMolOnlyPocket(pocket=pocket, seed=tmp, mode='general',
                                       extra_range=(-8, 8), remain_seed=True)
            data = ds[0]
        except Exception as e:
            results[tag] = {"error": f"featurize: {e}"}; continue
        # map dataset heavy-atom order back to mol indices via coordinates
        conf = data['mol_pos']  # centered coords tensor
        molc = m.GetConformer().GetPositions() - m.GetConformer().GetPositions().mean(0)
        n_h = m.GetNumAtoms()
        import numpy as np
        d = np.linalg.norm(molc[:, None, :] - conf.numpy()[:n_h][None, :, :], axis=-1)
        order = d.argmin(axis=1)  # mol atom i -> data atom order[i]

        batch = ProteinLigandDataLoader([data], batch_size=1)
        for b in batch:
            b = b.to(dev)
            with torch.no_grad():
                um = accelerator.unwrap_model(model)
                # fixed atoms = all except ambiguous ones
                mask = torch.ones(b.mol_element.shape[0], dtype=torch.bool, device=dev)
                for ai in amb_atoms:
                    mask[order[ai]] = False
                # beliefs
                theta = torch.full((b.mol_element.shape[0], 13), 1.0 / 13, device=dev)
                # locked atoms: use their current class (from featurizer parse of the SDF)
                feat = b.mol_element  # already class idx via featurizer
                for j, cls in enumerate(feat.tolist()):
                    if mask[j]:
                        onehot = torch.zeros(13, device=dev); onehot[cls] = 1.0
                        theta[j] = onehot * 0.98 + 0.02 / 13
                mu = b.mol_pos / 2  # already centered; gamma scaling at t=1
                gamma = 1 - sigma1 ** 2
                t = torch.ones((b.mol_element.shape[0], 1), device=dev)
                coord_pred, p0, k_hat, _, _ = um.interdependency_modeling(
                    time=t, pocket_pos=b.pocket_pos / 2,
                    pocket_v=b.pocket_atom_feature.float(),
                    batch_pocket=b.pocket_element_batch,
                    ed_pos=torch.empty((0, 3), device=dev),
                    ed_v=torch.empty((0, 4), device=dev),
                    batch_ed=b.ed_ed_code_batch,
                    batch_ligand=b.mol_element_batch,
                    theta_h_t=theta, mu_pos_t=mu * gamma, gamma_coord=gamma,
                    pocket_res=None, pocket_ele=None)
                probs = torch.softmax(p0, dim=-1) if p0.shape and p0.max() > 5 else p0
                probs = probs.cpu().numpy()
        entry = {"note": note, "pocket_used": pocket, "sites": {}}
        for ai in amb_atoms:
            j = order[ai]
            p = probs[j]
            top = np.argsort(-p)[:4]
            entry["sites"][f"mol_atom{ai}"] = {
                "top4": [(CLASS_NAMES[int(k)], round(float(p[k]), 4)) for k in top],
                "P_S_aliph": round(float(p[10]), 4),
                "P_S_arom": round(float(p[11]), 4),
                "P_P_arom": round(float(p[9]), 4),
                "verdict": CLASS_NAMES[int(np.argmax(p))],
            }
        results[tag] = entry
        print(f"[done] {tag}: " + json.dumps(entry["sites"], ensure_ascii=False)[:200])

    with open(args.out, 'w') as f:
        json.dump(results, f, indent=1, ensure_ascii=False)
    print(f"\nsaved -> {args.out}")

if __name__ == '__main__':
    main()
