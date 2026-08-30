# Reproduction-route molecules: 2D figures and SDFs

| figure | molecule | role |
|---|---|---|
| stage1_gen2k1369 | cold-start hit (phenanthroline-type core) | Stage-1 library representative |
| stage2a_dimer_allN | all-N isoelectronic dimer | Stage-2 precursor; red = ring N sites of the endorsed N->S conversion |
| stage2b_seed_twoBT | two benzothiazole fragments (18 atoms) | Stage-2 anchor seed; red = converted S |
| stage3_nh_bridged | NH-bridged bis-benzothiazole | Stage-3 inpainting product; green = bridge atom |
| stage4_t2_product | `c1ccc2sc(Cc3nc4ccccc4s3)nc2c1` | Stage-4 final product (SMILES == target); green = N->C bridge |
| target_T2 | same | reference target |
| sp1_m1 / sp1_m2 / sp1_m3 | flavones | sp1 targets (m1, m2 reproduced exactly) |

Highlights: cyan = first fused ring system (5+6 benzothiazole / 6+6 chromene core);
blue = second ring system (standalone six-membered ring); red = N->S conversion
sites; green = bridge atom (N->C conversion); orange = hydroxyl oxygens.

Each molecule is also written as a single-molecule SDF next to its figure.
