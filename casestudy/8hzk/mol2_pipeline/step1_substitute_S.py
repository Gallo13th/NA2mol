"""Step1: Element substitution (NH/O -> S) to obtain benzothiophene scaffold.

Reads dummy_ligand.sdf, locates indole ([nH]) / benzofuran (o) heteroatoms,
replaces with S, sanitizes, verifies benzothiophene SMARTS match.
"""
import sys, os
from rdkit import Chem
from rdkit.Chem import AllChem

DUMMY = 'output/8hzk_output/dummy_ligand/dummy_ligand.sdf'
OUT_SDF = 'casestudy/8hzk/mol2_pipeline/step1_benzothiophene.sdf'

INDOLE = Chem.MolFromSmarts('c1ccc2[nH]ccc2c1')
BENZOFURAN = Chem.MolFromSmarts('c1ccc2occc2c1')
BENZOTHIO = Chem.MolFromSmarts('c1ccc2sccc2c1')

TARGET_IDX = [5133, 7785, 8700, 9882]


def replace_heteroatom_with_S(mol, hetero_idx):
    """Replace atom at hetero_idx with S, fix Hs, sanitize."""
    rw = Chem.RWMol(mol)
    atom = rw.GetAtomWithIdx(hetero_idx)
    old_sym = atom.GetSymbol()
    atom.SetAtomicNum(16)           # sulfur
    atom.SetNoImplicit(False)
    atom.SetNumExplicitHs(0)
    atom.SetIsAromatic(True)
    mol2 = rw.GetMol()
    try:
        Chem.SanitizeMol(mol2)
    except Exception as e:
        # try without aromaticity then re-perceive
        Chem.SanitizeMol(mol2, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^
                         Chem.SanitizeFlags.SANITIZE_KEKULIZE)
        Chem.SetAromaticity(mol2, Chem.AromaticityModel.AROMATICITY_RDKIT)
    return mol2, old_sym


def main():
    sup = Chem.SDMolSupplier(DUMMY)
    mols = [m for m in sup if m is not None]
    print(f'Loaded {len(mols)} mols')

    writer = Chem.SDWriter(OUT_SDF)
    success = []
    for idx in TARGET_IDX:
        m = mols[idx]
        mi = m.GetSubstructMatch(INDOLE)
        mf = m.GetSubstructMatch(BENZOFURAN)
        hetero_idx = None
        kind = None
        if mi:
            hetero_idx = mi[4]      # [nH] position in SMARTS
            kind = 'indole-NH'
        elif mf:
            hetero_idx = mf[4]      # o position in SMARTS
            kind = 'benzofuran-O'
        if hetero_idx is None:
            continue
        sym = m.GetAtomWithIdx(hetero_idx).GetSymbol()
        n_h = m.GetAtomWithIdx(hetero_idx).GetTotalNumHs()
        print(f'idx={idx} {kind} hetero_atom={hetero_idx} sym={sym} H={n_h}')

        new, old_sym = replace_heteroatom_with_S(m, hetero_idx)
        csmi = Chem.MolToSmiles(new)
        has_bt = len(new.GetSubstructMatch(BENZOTHIO)) > 0
        print(f'   -> SMI: {csmi}')
        print(f'   -> benzothiophene match: {has_bt}')
        if has_bt:
            new.SetProp('_Name', f'dummy_idx_{idx}_{kind}_to_S')
            new.SetProp('origin_idx', str(idx))
            new.SetProp('edit', f'{kind}->{kind.split("-")[0]}->S')
            new.SetProp('smiles', csmi)
            writer.write(new)
            success.append((idx, csmi, kind))
    writer.close()
    print(f'\nSuccess: {len(success)}/{len(TARGET_IDX)}')
    for s in success:
        print('  ', s)


if __name__ == '__main__':
    main()
