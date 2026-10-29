#!/usr/bin/env python
"""Extract the full RNA receptor (all ATOM records, exclude HETATM/FMN)
from MODEL 1 of 1FMN.pdb -> 1FMN_receptor_full.pdb (no hydrogens)."""
import os
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "1FMN.pdb")
OUT = os.path.join(HERE, "1FMN_receptor_full.pdb")

in_m1 = False
n = 0
with open(SRC) as fh, open(OUT, "w") as out:
    out.write("MODEL        1\n")
    for ln in fh:
        rec = ln[0:6].strip()
        if rec == "MODEL":
            in_m1 = (ln.split()[1] == "1")
            continue
        if rec == "ENDMDL" and in_m1:
            out.write("ENDMDL\n")
            break
        if not in_m1:
            continue
        if rec != "ATOM":
            continue
        # skip hydrogens (element col 77-78)
        element = ln[76:78].strip() if len(ln) >= 78 else ""
        if element == "H":
            continue
        out.write(ln if ln.endswith("\n") else ln + "\n")
        n += 1
print(f"[extract] {n} RNA heavy atoms from MODEL 1 -> {OUT}")
