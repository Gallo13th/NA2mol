# DUDE cold-start verification pockets

Protein pockets extracted from `paperdata/DUDE_proteins.zip` (14 A around the
co-crystal ligand) used to verify cold-start reconstruction yield.

Reproduce (100 samples, from the repository root):

    python - << 'PY'
    # pocket preparation is done; _pocket.pdb files already exist
    PY
    # yield test driver (see repo history) or via genbypkt cold start:
    python genbypkt.py --ckpt ckpt/onlypkt_*.pth \
        --pocket casestudy/_dude_test/1qw6_protein_wo_ligand.pdb \
        --ligand casestudy/_dude_test/dude_h_1qw6.sdf \
        --output_prefix dude_1qw6 --total_target 50

Reference yields (2026-08-27, n_heads=16 fix): sp1 93/100, 1qw6 93/100,
1c8k 77/100.
