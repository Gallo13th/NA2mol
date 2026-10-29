#!/usr/bin/env python
"""SPEC 04 — verify functional fragment libraries against mol4 (independent check).

Reads known_mols.txt (ONLY here, not in build).
Checks:
 1. mol4's three units D=N(C)CCO / A=C=C(C#N) / pi=stilbene are each in the
    corresponding library AND can substruct-match mol4.
 2. Single-role purity spot-check: D-top fragments must not contain C#N / C=S / nitro;
    A-top fragments must not contain tertiary amine.
"""
import os
import pickle

from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

ROOT = "/home/data/users/gaoletian/rna2smol"
LIB_DIR = os.path.join(ROOT, "fragment_library", "functional_blocks")
KNOWN_PATH = os.path.join(ROOT, "casestudy", "8hzk", "known_mols.txt")

ROLE_FILE = {
    "donor": "donors.pkl",
    "acceptor": "acceptors.pkl",
    "bridge": "bridges.pkl",
}


def load_lib(role):
    with open(os.path.join(LIB_DIR, ROLE_FILE[role]), "rb") as f:
        return pickle.load(f)


def smiles_set(role):
    return {it["smiles"]: it for it in load_lib(role)}


# acceptor contamination patterns (should NOT appear in D library)
A_CONTAM_PATTERNS = {
    "cyano": Chem.MolFromSmarts("[#6]#[#7]"),
    "thiocarbonyl": Chem.MolFromSmarts("C=S"),
    "nitro": Chem.MolFromSmarts("[N+](=O)[O-]"),
    "rhodanine": Chem.MolFromSmarts("O=C1NC(=S)SC1"),
}
# donor contamination pattern (should NOT appear in A library).
# Use a NON-amide tertiary amine: require all three N-attached carbons to be
# non-carbonyl / non-thiocarbonyl, so acceptor ring N (e.g. rhodanine amide N,
# barbituric N) is not mis-flagged as a donor.
D_CONTAM_PATTERNS = {
    "tert_amine_nonamide": Chem.MolFromSmarts(
        "[NX3;H0]"
        "([#6;!$(C=O);!$(C=S)])"
        "([#6;!$(C=O);!$(C=S)])"
        "[#6;!$(C=O);!$(C=S)]"
    ),
}


def main():
    print("=" * 60)
    print("Verification of functional fragment libraries")
    print("=" * 60)

    # ---- load known mols ----
    with open(KNOWN_PATH) as f:
        known = [ln.strip() for ln in f if ln.strip()]
    mol4_smi = known[3]  # 4th line: D-pi-A with cyanoethylene + stilbene
    mol4 = Chem.MolFromSmiles(mol4_smi)
    assert mol4 is not None, f"mol4 parse failed: {mol4_smi}"
    print(f"[mol4] {mol4_smi}")
    print(f"[mol4] canonical: {Chem.MolToSmiles(mol4)}")

    D = smiles_set("donor")
    A = smiles_set("acceptor")
    P = smiles_set("bridge")
    print(f"[lib sizes] D={len(D)}  A={len(A)}  pi={len(P)}")

    # ---- mol4 unit acceptance ----
    # D unit: N(C)CCO  (tertiary amine + CCO + methyl)
    d_unit = Chem.MolFromSmiles("CN(C)CCO")
    d_canon = Chem.MolToSmiles(d_unit)
    # A unit: C=C(C#N) cyanoethylene
    a_unit = Chem.MolFromSmiles("C=C(C#N)")
    a_canon = Chem.MolToSmiles(a_unit)
    # pi unit: stilbene
    pi_unit = Chem.MolFromSmiles("C(=Cc1ccccc1)c1ccccc1")
    pi_canon = Chem.MolToSmiles(pi_unit)

    print("\n--- mol4 acceptance ---")
    print(f"target D unit canonical : {d_canon}")
    print(f"target A unit canonical : {a_canon}")
    print(f"target pi unit canonical: {pi_canon}")

    def check(role_lib, role_name, unit, unit_canon, target_smi):
        # library membership: exact smiles or substruct match in library
        exact = unit_canon in role_lib
        # also: any library fragment that substruct-matches mol4 AND the unit
        # substruct-matches the same set of atoms
        unit_in_mol4 = mol4.HasSubstructMatch(unit)
        # check at least one lib fragment that equals unit and matches mol4
        lib_match_mol4 = None
        for smi, it in role_lib.items():
            fmol = Chem.MolFromSmiles(smi)
            if fmol is None:
                continue
            if fmol.HasSubstructMatch(unit) and mol4.HasSubstructMatch(fmol):
                lib_match_mol4 = smi
                break
        print(f"\n[{role_name}] target={target_smi}")
        print(f"  exact in {role_name} lib : {exact}")
        print(f"  unit matches mol4       : {unit_in_mol4}")
        if lib_match_mol4:
            print(f"  lib fragment matches mol4: {lib_match_mol4}  (freq={role_lib[lib_match_mol4]['freq']})")
        else:
            print(f"  lib fragment matches mol4: NONE")
        return exact or (lib_match_mol4 is not None)

    ok_d = check(D, "D", d_unit, d_canon, "N(C)CCO")
    ok_a = check(A, "A", a_unit, a_canon, "C=C(C#N)")
    ok_pi = check(P, "pi", pi_unit, pi_canon, "stilbene")

    # ---- single-role purity spot check ----
    print("\n--- single-role purity spot check (top-15 each) ---")

    def top_smiles(role, n=15):
        return [it["smiles"] for it in sorted(
            load_lib(role), key=lambda x: -x["freq"])[:n]]

    d_viol, a_viol = [], []
    for smi in top_smiles("donor"):
        m = Chem.MolFromSmiles(smi)
        if m is None:
            continue
        for pname, pat in A_CONTAM_PATTERNS.items():
            if m.HasSubstructMatch(pat):
                d_viol.append((smi, pname))
                break
    for smi in top_smiles("acceptor"):
        m = Chem.MolFromSmiles(smi)
        if m is None:
            continue
        for pname, pat in D_CONTAM_PATTERNS.items():
            if m.HasSubstructMatch(pat):
                a_viol.append((smi, pname))
                break

    print(f"D top-15 acceptor-contamination violations: {len(d_viol)}")
    for smi, p in d_viol:
        print(f"  VIOL  {smi}  contains {p}")
    print(f"A top-15 donor-contamination violations: {len(a_viol)}")
    for smi, p in a_viol:
        print(f"  VIOL  {smi}  contains {p}")

    # ---- summary ----
    print("\n=== SUMMARY ===")
    print(f"mol4 D  accepted: {ok_d}")
    print(f"mol4 A  accepted: {ok_a}")
    print(f"mol4 pi accepted: {ok_pi}")
    purity_ok = (len(d_viol) == 0) and (len(a_viol) == 0)
    print(f"single-role purity (top-15): {purity_ok}")
    overall = ok_d and ok_a and ok_pi and purity_ok
    print(f"OVERALL: {'PASS' if overall else 'FAIL'}")
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
