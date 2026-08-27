#!/usr/bin/env python3
"""Chain-conditional probe for the T1 bridge region (positions 5-8).

Fixes the combination-artifact of per-site independent argmax assembly:
instead of masking ONE site at a time with all others locked to TARGET
elements, we sweep the bridge sequentially — each newly decided site is
locked to its *decided* element before probing the next, so the conditional
structure (joint distribution correlations) is preserved.

Protocols:
  A. chain sweep 5→6→7→8 : each step masks only the current site, all
     previous bridge sites locked to their chain-decided element, the rest
     locked to target elements. Record argmax + full 13-class probs.
  B. joint best-of-bridge : exhaustive enumeration of element assignments
     for the 4 bridge carbons over the top-2 per-site candidates from (A)
     (pruned), score each full assignment by SUM of log P(site|rest) using
     sequential factorization  P(6)P(7|6)P(8|6,7)... — approximation of the
     joint via chain rule.
  C. control : all-C bridge (the target assignment) scored the same way, so
     we can compare "model joint preference" of C-chain vs any N-mixed chain.
"""
import os, sys, json, itertools
sys.path.insert(0, "/home/data/users/gaoletian/rna2smol")
os.chdir("/home/data/users/gaoletian/rna2smol")
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')

import numpy as np
import torch
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
from accelerate import Accelerator
from model.bfn4edmol import Config, BFN4SBDDScoreModel
from genbypkt import load_compiled_model
from dataloader.featurizer import GeomSeedMolOnlyPocket, ProteinLigandDataLoader
from dataloader.featurizer import MAP_ATOM_TYPE_AROMATIC_TO_INDEX as TRUE_TABLE

CLS = {0:'H',1:'C',2:'C*',3:'N',4:'N*',5:'O',6:'O*',7:'F',8:'P',9:'P*',10:'S',11:'S*',12:'Cl'}
BRIDGE = [5, 6, 7, 8]          # bridge atoms in skeleton indexing (5=C,6,7,8 chain)
T1 = "CCN1C=C/C(=C/C=C/c2cc[n+]3ccccc3n2)c2ccccc21"
SKEL_SDF = 'casestudy/kras_g4/ber102_rdock/skeleton_aromatic_ber102.sdf'
POCKET = 'casestudy/kras_g4/ber102_rdock/ber102_pocket.pdb'
OUT = 'casestudy/kras_g4/ber102_rdock/bridge_chainprobe_results.json'


def class_of(atomnum, aromatic):
    return TRUE_TABLE.get((atomnum, aromatic), TRUE_TABLE.get((atomnum, False), 1))


def build_target_classes(skel, target):
    """target molecule atoms are index-aligned with the skeleton (verified)."""
    tgt_cls = []
    for a in target.GetAtoms():
        z, ar = a.GetAtomicNum(), a.GetIsAromatic()
        # pyridinium N+ and quinolinium N+ both parse as aromatic N
        tgt_cls.append(class_of(z, ar))
    return tgt_cls


def probe(model, accelerator, dev, sigma1, seed_sdf, pocket, assignments, mask_sites):
    """assignments: dict site->class_idx (locked). mask_sites: list of sites to mask
    (uniform). Returns probs [n_atoms, 13] numpy."""
    ds = GeomSeedMolOnlyPocket(pocket=pocket, seed=seed_sdf, mode='general',
                               extra_range=(-8, 8), remain_seed=True)
    data = ds[0]
    batch = ProteinLigandDataLoader([data], batch_size=1)
    for b in batch:
        b = b.to(dev)
        with torch.no_grad():
            um = accelerator.unwrap_model(model)
            n = b.mol_element.shape[0]
            theta = torch.full((n, 13), 1.0 / 13, device=dev)
            # dataset heavy-atom order == mol order (verified in prior probes);
            # b.mol_element holds atomic numbers
            mol_atomic = b.mol_element.tolist()
            skel_atomic = assignments['_atomic']  # list of atomic numbers per site
            arom = assignments['_arom']
            for j in range(n):
                if j in mask_sites:
                    continue
                cls = assignments.get(j, class_of(mol_atomic[j], False))
                onehot = torch.zeros(13, device=dev)
                onehot[cls] = 1.0
                theta[j] = onehot * 0.98 + 0.02 / 13
            mu = b.mol_pos / 2
            gamma = 1 - sigma1 ** 2
            t = torch.ones((n, 1), device=dev)
            _, p0, _, _, _ = um.interdependency_modeling(
                time=t, pocket_pos=b.pocket_pos / 2,
                pocket_v=b.pocket_atom_feature.float(),
                batch_pocket=b.pocket_element_batch,
                ed_pos=torch.empty((0, 3), device=dev),
                ed_v=torch.empty((0, 4), device=dev),
                batch_ed=b.ed_ed_code_batch,
                batch_ligand=b.mol_element_batch,
                theta_h_t=theta, mu_pos_t=mu * gamma, gamma_coord=gamma,
                pocket_res=None, pocket_ele=None)
            probs = p0 if p0.max() <= 5 else torch.softmax(p0, -1)
            return probs.cpu().numpy()
    return None


def main():
    skel = Chem.SDMolSupplier(SKEL_SDF, removeHs=True)[0]
    target = Chem.MolFromSmiles(T1)
    tgt_cls = build_target_classes(skel, target)
    # write skeleton as seed (coords fixed)
    tmp = '/tmp/opencode/_chainprobe_seed.sdf'
    w = Chem.SDWriter(tmp); w.write(skel); w.close()

    args_dict = {'config_file': 'config.yaml', 'exp_name': 't', 'revision': 'r',
                 'debug': True, 'resume': False, 'wandb_resume_id': None,
                 'empty_folder': False, 'test_only': True, 'seed': 42, 'no_wandb': True,
                 'logging_level': 'warning', 'random_rot': False, 'pos_noise_std': 0,
                 'pos_normalizer': 2.0, 'batch_size': 8, 'epochs': 15,
                 'v_loss_weight': 1, 'lr': 5e-4, 'scheduler': 'plateau',
                 'weight_decay': 0, 'max_grad_norm': 'Q', 'sigma1_coord': 0.03,
                 'beta1': 1.5, 't_min': 0.0001, 'use_discrete_t': True,
                 'discrete_steps': 1000, 'destination_prediction': True,
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

    atomic = [a.GetAtomicNum() for a in skel.GetAtoms()]
    arom = [a.GetIsAromatic() for a in skel.GetAtoms()]

    def mk_assign(bridge_cls):
        A = {'_atomic': atomic, '_arom': arom}
        for i, c in enumerate(tgt_cls):
            A[i] = c
        for s, c in bridge_cls.items():
            A[s] = c
        return A

    # ---------- Protocol A: chain sweep 5->6->7->8 ----------
    results = {'protocol_A_chain': {}, 'protocol_B_joint': {}, 'protocol_C_control': {}}
    decided = {}
    for s in BRIDGE:
        A = mk_assign(decided)
        probs = probe(model, accelerator, dev, sigma1, tmp, POCKET, A, [s])
        p = probs[s]
        top = np.argsort(-p)[:5]
        results['protocol_A_chain'][f'site{s}'] = {
            'context': {f'site{k}': CLS[v] for k, v in decided.items()},
            'top5': [(CLS[int(k)], round(float(p[k]), 4)) for k in top],
        }
        best = int(top[0])
        decided[s] = best
        print(f"[A] site{s}: top={CLS[best]} {p[best]*100:.1f}%  (context {decided})")

    # ---------- Protocol B: joint enumeration (chain-rule score) ----------
    # candidate set per site: top-2 from chain sweep + target class (C for all)
    cand = {}
    for s in BRIDGE:
        top5 = results['protocol_A_chain'][f'site{s}']['top5']
        cs = {TRUE_TABLE[(6, False)], TRUE_TABLE[(6, True)]}  # target = C classes
        for name, pr in top5[:2]:
            # map class name back to idx
            for idx, nm in CLS.items():
                if nm == name:
                    cs.add(idx); break
        cand[s] = sorted(cs)
    scores = {}
    for combo in itertools.product(*[cand[s] for s in BRIDGE]):
        assign = dict(zip(BRIDGE, combo))
        logp = 0.0
        # chain-rule: sequential factorization in bridge order
        for i, s in enumerate(BRIDGE):
            ctx = {k: v for k, v in list(assign.items())[:i]}
            A = mk_assign(ctx)
            probs = probe(model, accelerator, dev, sigma1, tmp, POCKET, A, [s])
            p = probs[s, assign[s]] + 1e-9
            logp += float(np.log(p))
        key = '+'.join(CLS[c] for c in combo)
        scores[key] = round(logp, 3)
    results['protocol_B_joint']['candidates'] = {f'site{s}': [CLS[c] for c in cand[s]] for s in BRIDGE}
    results['protocol_B_joint']['chain_rule_logp'] = dict(sorted(scores.items(), key=lambda x: -x[1])[:12])
    print("\n[B] joint chain-rule top-12:")
    for k, v in list(sorted(scores.items(), key=lambda x: -x[1])[:12]):
        print(f"    {k}: {v}")

    # ---------- Protocol C: control (all-C bridge = target) ----------
    ctrl_assign = {s: TRUE_TABLE[(6, False)] for s in BRIDGE}
    logp = 0.0
    detail = {}
    for i, s in enumerate(BRIDGE):
        ctx = {k: v for k, v in list(ctrl_assign.items())[:i]}
        A = mk_assign(ctx)
        probs = probe(model, accelerator, dev, sigma1, tmp, POCKET, A, [s])
        p = float(probs[s, ctrl_assign[s]] + 1e-9)
        detail[f'site{s}'] = round(p, 4)
        logp += np.log(p)
    results['protocol_C_control'] = {'assignment': 'C+C+C+C (target bridge)',
                                     'per_site_p': detail, 'chain_rule_logp': round(logp, 3)}
    print(f"\n[C] all-C bridge (target) logp = {logp:.3f}")

    with open(OUT, 'w') as f:
        json.dump(results, f, indent=1, ensure_ascii=False)
    print(f"\nsaved -> {OUT}")


if __name__ == '__main__':
    main()
